"""CPU pilot: checkpoint projection and learned driver-to-history cross-attention."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from maraboupy import Marabou, MarabouCore as M
from sklearn.metrics import roc_auc_score
from torch import nn


def equation(query, terms, rhs):
    eq = M.Equation()
    for variable, coefficient in terms:
        eq.addAddend(float(coefficient), int(variable))
    eq.setScalar(float(rhs))
    query.addEquation(eq)


def solve(query):
    status, values, _ = M.solve(query, Marabou.createOptions(timeoutInSeconds=30, verbosity=0), "")
    return str(status).lower(), values


def download(repo, kind, suffix, revision):
    names = HfApi().list_repo_files(repo, repo_type=kind, revision=revision)
    matches = [n for n in names if n == suffix or n.endswith("/" + suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one file ending in {suffix}: {matches}")
    return hf_hub_download(repo, matches[0], repo_type=kind, revision=revision)


def large(out):
    repo = "stanford-star/rt-v1"
    revision = HfApi().model_info(repo).sha
    path = download(repo, "model", "pretrain_rel-f1_driver-dnf.pt", revision)
    obj = torch.load(path, map_location="cpu", weights_only=True)
    def matrices(tree, prefix=""):
        for k, v in tree.items():
            key = prefix + str(k)
            if isinstance(v, dict):
                yield from matrices(v, key + ".")
            elif isinstance(v, torch.Tensor) and v.ndim == 2 and key.endswith("weight"):
                yield key, v
    choices = [(k, w) for k, w in matrices(obj) if w.shape[1] == 256]
    if not choices:
        raise ValueError("No width-256 weight matrix found")
    key, matrix = min(choices, key=lambda pair: ("proj" not in pair[0].lower(), pair[0]))
    w = matrix[0].double().numpy()
    radius = 0.01
    exact = radius * np.abs(w).sum()
    assert exact > 0
    rows = []
    for factor in (0.9, 1.1):
        n = len(w)
        query = M.InputQuery()
        query.setNumberOfVariables(n + 1)
        for i in range(n):
            query.setLowerBound(i, -radius)
            query.setUpperBound(i, radius)
        equation(query, [(i, a) for i, a in enumerate(w)] + [(n, -1)], 0)
        query.setLowerBound(n, factor * exact)
        status, values = solve(query)
        record = dict(status=status, factor=factor, delta=float(factor * exact))
        if status == "sat":
            witness = np.array([values[i] for i in range(n)])
            assert np.max(np.abs(witness)) <= radius + 1e-7
            assert w @ witness >= factor * exact - 1e-7
            record["witness"] = witness.tolist()
        rows.append(record)
    (out / "projection.json").write_text(json.dumps(dict(revision=revision, file=Path(path).name,
        matrix=key, row=0, radius=radius, exact=float(exact), results=rows), indent=2))
    assert [r["status"] for r in rows] == ["sat", "unsat"], rows


class HistoryRT(nn.Module):
    """Driver query attends to its previous results via the driver foreign key."""
    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.randn(16))
        self.key = nn.Linear(2, 16)
        self.value = nn.Linear(3, 16)
        self.hidden = nn.Linear(16, 32)
        self.output = nn.Linear(32, 1)

    def attention(self, x):
        return ((self.key(x[..., :2]) @ self.query) / 4).softmax(-1)

    def forward(self, x):
        pooled = (self.attention(x).unsqueeze(-1) * self.value(x)).sum(-2)
        return self.output(self.hidden(pooled).relu()).squeeze(-1)


def verify(model, x, radius):
    # x[-1,2] = most recent historical laps / 100. Keys depend only on age and grid.
    weights = model.attention(x)
    base = model.hidden((weights[:, None] * model.value(x)).sum(0))
    slope = model.hidden.weight @ model.value.weight[:, 2] * weights[-1] / 100
    nominal = float(model(x))
    n = len(base)
    y = 2*n + 1
    q = M.InputQuery()
    q.setNumberOfVariables(y + 1)
    low = max(-radius, -100 * float(x[-1, 2]))
    q.setLowerBound(0, low)
    q.setUpperBound(0, radius)
    for i in range(n):
        equation(q, [(1+i, 1), (0, -slope[i])], base[i])
        M.addReluConstraint(q, 1+i, 1+n+i)
    equation(q, [(y, 1)] + [(1+n+i, -w) for i,w in enumerate(model.output.weight[0])],
             model.output.bias[0])
    # Ask whether the signed margin can reach zero (ties count as unstable).
    if nominal > 0:
        q.setUpperBound(y, 0)
    else:
        q.setLowerBound(y, 0)
    status, values = solve(q)
    result = dict(radius=radius, nominal=nominal, status=status)
    offsets = torch.arange(np.ceil(low), radius + 1, dtype=x.dtype)
    variants = x.repeat(len(offsets), 1, 1)
    variants[:, -1, 2] += offsets / 100
    scores = model(variants)
    crossed = scores <= 0 if nominal > 0 else scores >= 0
    result["integer_witnesses"] = offsets[crossed].tolist()
    if status == "unsat" and crossed.any():
        raise RuntimeError("Solver verdict conflicts with integer replay")
    if status == "sat":
        d = values[0]
        edited = x.clone()
        edited[-1, 2] += d / 100
        score = float(model(edited))
        result.update(delta_laps=d, replay_logit=score)
        if not low <= d <= radius or (score > 0 if nominal > 0 else score < 0):
            result["status"] = "unknown-replay"
    return result


def small(out, seed):
    repo = "stanford-star/relbench-v1"
    revision = HfApi().dataset_info(repo).sha
    def table(suffix):
        return pd.read_parquet(download(repo, "dataset", "rel-f1/" + suffix, revision))
    history = table("db/results.parquet").dropna(subset=["laps", "grid", "date"])
    history = history.sort_values(["date","resultId"])
    grouped = dict(tuple(history.groupby("driverId")))
    def examples(split):
        labels = table(f"tasks/driver-dnf/{split}.parquet")
        target = "did_not_finish"
        xs, ys, ids = [], [], []
        for row in labels.to_dict("records"):
            group = grouped.get(row["driverId"])
            if group is None:
                continue
            past = group[group.date < row["date"]].tail(4)
            if len(past) != 4:
                continue
            age = (row["date"] - past.date).dt.days
            xs.append(np.column_stack((np.log1p(age)/8, past.grid/30, past.laps/100)))
            ys.append(row[target])
            ids.append([int(row["driverId"]), str(row["date"]), int(past.resultId.iloc[-1])])
        return torch.tensor(np.stack(xs)).double(), torch.tensor(ys).double(), ids
    x, y, _ = examples("train")
    v, labels, ids = examples("val")
    torch.manual_seed(seed)
    model = HistoryRT().double()
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(20):
        for ix in torch.randperm(len(x)).split(128):
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(model(x[ix]), y[ix]).backward()
            opt.step()
    model.eval()
    torch.save(model.state_dict(), out/"small.pt")
    with torch.no_grad():
        auc = roc_auc_score(labels.numpy(), model(v).numpy())
        indices = np.random.default_rng(seed).permutation(len(v))[:30]
        with (out/"queries.jsonl").open("w") as f:
            for i in indices:
                for radius in (0, 1, 5, 10):
                    r = verify(model, v[i], radius)
                    print(json.dumps(dict(query=ids[i], **r)), file=f, flush=True)
    (out/"metrics.json").write_text(json.dumps(dict(auc=auc, revision=revision, seed=seed,
                                                  train=len(y), validation=len(labels))))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["large","small"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    root = Path("results") / (args.mode + "-" + os.getenv("SLURM_JOB_ID","local"))
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.getenv("SLURM_CPUS_PER_TASK", "1")))
    if args.mode == "large":
        large(root)
    else:
        small(root, args.seed)

