# Sourced by the Slurm launcher in the job shell.
compatible() {
    "$1" -c 'import sys; sys.exit(not (sys.implementation.name=="cpython" and sys.version_info[:2] in ((3,10),(3,11))))' 2>/dev/null
}
find_python() {
    local candidate
    for candidate in python3.11 python3.10 python3 python; do
        if command -v "$candidate" >/dev/null && compatible "$candidate"; then
            PYTHON_BIN=$(command -v "$candidate")
            return 0
        fi
    done
    return 1
}
if [[ -n ${CLUSTER_PYTHON:-} ]]; then
    PYTHON_BIN=$CLUSTER_PYTHON
    compatible "$PYTHON_BIN" || { echo "CLUSTER_PYTHON=$PYTHON_BIN precisa executar CPython 3.10 ou 3.11" >&2; exit 2; }
elif ! find_python; then
    if ! type module >/dev/null 2>&1; then
        for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
            if [[ -r $init ]]; then source "$init"; break; fi
        done
    fi
    if type module >/dev/null 2>&1; then
        module load python/3.10.13 || true
    fi
    if ! find_python; then
        echo "Procurando CPython 3.11 gerenciado pelo uv" >&2
        export UV_PYTHON_INSTALL_DIR="$HOME/pylibs/rt-managed-python"
        mkdir -p "$UV_PYTHON_INSTALL_DIR"
        exec 8>"$UV_PYTHON_INSTALL_DIR.lock"
        flock 8
        uv_bin="$UV_PYTHON_INSTALL_DIR/uv"
        if command -v uv >/dev/null; then
            uv_bin=$(command -v uv)
        elif [[ ! -x "$uv_bin" ]]; then
            bootstrap=$(mktemp -d)
            curl --retry 3 -fsSL https://github.com/astral-sh/uv/releases/download/0.7.13/uv-x86_64-unknown-linux-gnu.tar.gz -o "$bootstrap/uv.tar.gz"
            tar xzf "$bootstrap/uv.tar.gz" -C "$bootstrap"
            install -m 755 "$bootstrap/uv-x86_64-unknown-linux-gnu/uv" "$uv_bin.tmp"
            mv "$uv_bin.tmp" "$uv_bin"
            rm -rf -- "$bootstrap"
        fi
        "$uv_bin" python install 3.11
        PYTHON_BIN=$("$uv_bin" python find --managed-python 3.11)
        compatible "$PYTHON_BIN"
        flock -u 8
    fi
fi
"$PYTHON_BIN" -c 'import sys; print(sys.executable, sys.version, flush=True)'
