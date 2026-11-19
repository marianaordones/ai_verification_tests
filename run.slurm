#!/usr/bin/env bash
#SBATCH --job-name=rt-marabou
#SBATCH --partition=new_p_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mode=${1:?Pass small or large}
case "$mode" in small|large) ;; *) exit 2;; esac
hostname
export PYTHONNOUSERSITE=1
unset PYTHONPATH
source "$SLURM_SUBMIT_DIR/select_python.sh"
ver=$("$PYTHON_BIN" -c 'import sys; print(sys.implementation.cache_tag)')
hash=$(sha256sum requirements.txt | cut -c1-16)
libs="$HOME/pylibs/rt-marabou-$ver-$hash"
mkdir -p "$HOME/pylibs"
export HF_HOME="$PWD/cache/huggingface"
tmp=$(mktemp -d)
stage=""
trap 'rm -rf -- "$tmp"; if [[ -n "$stage" ]]; then rm -rf -- "$stage"; fi' EXIT
exec 9>"$libs.lock"
flock 9
if [[ ! -f "$libs/.complete" ]]; then
    if ! "$PYTHON_BIN" -m pip --version; then
        curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$tmp/get-pip.py"
        "$PYTHON_BIN" "$tmp/get-pip.py" --target "$tmp/bootstrap" 'pip==25.0.1'
        export PYTHONPATH="$tmp/bootstrap"
    fi
    stage=$(mktemp -d "$libs.stage.XXXXXX")
    "$PYTHON_BIN" -m pip install --target "$stage" -r requirements.txt
    PYTHONPATH="$stage" "$PYTHON_BIN" -c 'import pilot; print(pilot.M.InputQuery())'
    touch "$stage/.complete"
    mv -T "$stage" "$libs"
    stage=""
fi
flock -u 9
export PYTHONPATH="$libs"
srun "$PYTHON_BIN" -u pilot.py "$@"

