# Sourced inside a single-node allocation. Marabou pilot supports CPython 3.10/3.11.
hostname
echo "partition=${SLURM_JOB_PARTITION:-unknown}"
scratch=${SLURM_TMPDIR:-/tmp}
storage_report() {
    for dest in "$scratch" "$PWD" "$HOME"; do
        df -h "$dest" >&2 || true
        df -i "$dest" >&2 || true
    done
    if command -v quota >/dev/null; then quota -s >&2 || true; fi
}
JOB_TMP=$(mktemp -d "$scratch/rt-pilot.XXXXXXXX") || { storage_report; exit 1; }
cleanup_runtime() {
    rc=$?
    if (( rc != 0 )); then storage_report; fi
    rm -rf -- "$JOB_TMP"
    exit "$rc"
}
trap cleanup_runtime EXIT
export TMPDIR="$JOB_TMP" PYTHONNOUSERSITE=1 PIP_CACHE_DIR="$JOB_TMP/pip-cache"
export HF_HOME="$PWD/cache/huggingface"
unset PYTHONPATH
echo "scratch=$JOB_TMP"
try_python() {
    local path
    path=$(command -v -- "$1") && [[ -x "$path" ]] || return 1
    "$path" -c 'import sys; print(sys.executable, sys.version, file=sys.stderr); sys.exit(not (sys.implementation.name=="cpython" and sys.version_info[:2] in ((3,10),(3,11))))' || return 1
    PYTHON_BIN=$("$path" -c 'import os,sys; print(os.path.abspath(sys.executable))')
}
find_python() {
    local name
    for name in "$PWD/.venv/bin/python" python3.11 python3.10 python3 python; do
        if try_python "$name"; then return 0; fi
    done
    return 1
}
if [[ -n ${CLUSTER_PYTHON+x} ]]; then
    try_python "$CLUSTER_PYTHON" || { echo "CLUSTER_PYTHON inválido: '$CLUSTER_PYTHON' (requer CPython 3.10/3.11)" >&2; exit 2; }
elif ! find_python; then
    if ! type module >/dev/null 2>&1; then
        for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
            if [[ -r "$init" ]]; then source "$init"; break; fi
        done
    fi
    if type module >/dev/null 2>&1; then module load python/3.10.13 || true; fi
    if ! find_python; then
        if [[ ${RT_ALLOW_UV:-0} != 1 ]]; then
            echo "Sem CPython compatível. Indique CLUSTER_PYTHON ou autorize RT_ALLOW_UV=1." >&2
            exit 2
        fi
        export UV_PYTHON_INSTALL_DIR="$JOB_TMP/python" UV_CACHE_DIR="$JOB_TMP/uv-cache"
        if command -v uv >/dev/null; then uv_bin=$(command -v uv)
        else
            curl --fail --location --retry 3 https://github.com/astral-sh/uv/releases/download/0.7.13/uv-x86_64-unknown-linux-gnu.tar.gz -o "$JOB_TMP/uv.tgz"
            tar xzf "$JOB_TMP/uv.tgz" -C "$JOB_TMP"
            uv_bin="$JOB_TMP/uv-x86_64-unknown-linux-gnu/uv"
        fi
        "$uv_bin" python install 3.11
        try_python "$("$uv_bin" python find --managed-python 3.11)"
    fi
fi
check_runtime() {
    "$PYTHON_BIN" -c 'import pilot, pyarrow; from maraboupy import MarabouCore; MarabouCore.InputQuery(); print("MarabouCore", MarabouCore.__file__)'
}
tag=$("$PYTHON_BIN" -c 'import sys; print(sys.implementation.cache_tag)')
hash=$(sha256sum requirements.txt | cut -c1-16)
previous="$HOME/pylibs/rt-marabou-$tag-$hash"
if [[ -f "$previous/.complete" ]]; then export PYTHONPATH="$previous"; fi
if ! check_runtime; then
    unset PYTHONPATH
    if ! "$PYTHON_BIN" -m pip --version; then
        wheel="$PWD/wheels/pip-25.0.1-py3-none-any.whl"
        if [[ ! -r "$wheel" ]]; then
            wheel="$JOB_TMP/pip-25.0.1-py3-none-any.whl"
            curl -fLsS --retry 3 https://files.pythonhosted.org/packages/py3/p/pip/pip-25.0.1-py3-none-any.whl -o "$wheel"
        fi
        export PYTHONPATH="$wheel"
        "$PYTHON_BIN" -m pip --version
    fi
    "$PYTHON_BIN" -m pip install --ignore-installed --target "$JOB_TMP/site" --find-links "$PWD/wheels" -r requirements.txt
    export PYTHONPATH="$JOB_TMP/site"
    check_runtime
fi
