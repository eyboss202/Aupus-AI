#!/usr/bin/env bash
# Chat with Aupus 0.1 Instruct in the browser, from a fresh clone or a ready machine:
#
#   bash aupus.sh                 # then the browser opens http://127.0.0.1:8000
#   bash aupus.sh --seed 42       # extra arguments go to scripts/chat_web.py
#
# Safe to run every time: each step is skipped when it is already done.
#   1. Python >= 3.11 and the virtual environment (.venv)
#   2. PyTorch: the CUDA build when an NVIDIA GPU is found, otherwise the CPU build (Apple: the default build)
#   3. Aupus and its dependencies (pip install -e .)
#   4. the model: release/aupus-0.1-instruct, exported from runs/aupus-0.1-instruct/best.pt if missing
#   5. the browser chat (Ctrl-C to quit)
#
# Settings (environment variables): MODEL (release/aupus-0.1-instruct), PORT (8000), VENV (.venv),
# TORCH_INDEX (a PyTorch wheel index URL, to override the automatic choice), NO_BROWSER=1 (don't open it).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-release/aupus-0.1-instruct}"
PORT="${PORT:-8000}"
VENV="${VENV:-.venv}"
CHECKPOINT=runs/aupus-0.1-instruct/best.pt
URL="http://127.0.0.1:${PORT}"
cd "$REPO"

die() { echo "error: $*" >&2; exit 1; }

find_python() {
    # The first Python >= 3.11 on PATH. `py` is the Windows launcher; the Microsoft Store
    # "python" stub fails the version check, so it is skipped.
    local cmd
    for cmd in python3 python "py -3"; do
        if $cmd -c 'import sys; sys.exit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
            echo "$cmd"
            return
        fi
    done
    return 1
}

venv_python() {
    if [ -x "$VENV/Scripts/python.exe" ]; then echo "$VENV/Scripts/python.exe"   # Windows
    elif [ -x "$VENV/bin/python" ]; then echo "$VENV/bin/python"                 # Linux, macOS
    else return 1
    fi
}

ensure_venv() {
    if VPY="$(venv_python)"; then
        "$VPY" -c 'import sys' >/dev/null 2>&1 \
            || die "$VENV is broken (was its Python moved or uninstalled?). Delete the $VENV folder and run this again."
        return
    fi
    local py
    py="$(find_python)" || die "Python 3.11 or newer is needed: https://www.python.org/downloads/"
    echo "creating the virtual environment in $VENV ($($py --version))..."
    $py -m venv "$VENV"
    VPY="$(venv_python)" || die "could not create $VENV"
    "$VPY" -m pip install --quiet --upgrade pip
}

torch_index() {
    # The PyTorch wheel index for this machine. CUDA 13.0 wheels need an NVIDIA driver >= 580.
    if [ -n "${TORCH_INDEX:-}" ]; then echo "$TORCH_INDEX"; return; fi
    [ "$(uname -s)" = Darwin ] && return   # macOS: PyPI's default build (CPU + Apple GPU)
    if command -v nvidia-smi >/dev/null 2>&1; then
        local driver
        driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
        driver="${driver%%.*}"
        if [ -n "$driver" ] && [ "$driver" -ge 580 ] 2>/dev/null; then
            echo https://download.pytorch.org/whl/cu130
        else
            echo https://download.pytorch.org/whl/cu126
        fi
        return
    fi
    echo https://download.pytorch.org/whl/cpu
}

ensure_torch() {
    "$VPY" -I -c 'import torch' >/dev/null 2>&1 && return
    local index
    index="$(torch_index)"
    echo "installing PyTorch (${index:-PyPI default}; the CUDA build is a ~2.5 GB download)..."
    if [ -n "$index" ]; then
        "$VPY" -m pip install torch --index-url "$index"
    else
        "$VPY" -m pip install torch
    fi
}

ensure_aupus() {
    # -I: don't let the current folder satisfy `import aupus`; the scripts need it installed.
    "$VPY" -I -c 'import aupus, numpy, yaml, safetensors, tqdm' >/dev/null 2>&1 && return
    echo "installing Aupus and its dependencies..."
    "$VPY" -m pip install --quiet -e .
}

ensure_model() {
    [ -e "$MODEL" ] && return
    [ "$MODEL" = release/aupus-0.1-instruct ] && [ -f "$CHECKPOINT" ] \
        || die "model not found: $MODEL (train it first: docs/TRAINING.md, or set MODEL=...)"
    echo "exporting $CHECKPOINT to $MODEL..."
    "$VPY" scripts/export_release.py --checkpoint "$CHECKPOINT" --out "$MODEL"
}

port_open() {
    "$VPY" -c "import socket, sys; s = socket.socket(); s.settimeout(0.5); sys.exit(s.connect_ex(('127.0.0.1', $PORT)))" 2>/dev/null
}

open_browser() {
    if command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$URL" || true   # Windows (also WSL)
    elif [ "$(uname -s)" = Darwin ]; then open "$URL"
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1
    fi
}

open_when_ready() {
    # Loading the model takes a few seconds; open the page once the server answers (give up after 2 min).
    local i
    for i in $(seq 1 240); do
        if port_open; then open_browser; return; fi
        sleep 0.5
    done
}

ensure_venv
ensure_torch
ensure_aupus
ensure_model
"$VPY" -c 'import torch; print("PyTorch", torch.__version__, "|", "GPU: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU found: replies run on the CPU")'

port_open && die "port $PORT is already in use (is Aupus already running? then open $URL), or pick another: PORT=8001 bash aupus.sh"

if [ "${NO_BROWSER:-0}" != 1 ]; then
    open_when_ready &
    BROWSER_WAIT=$!
    trap 'kill "$BROWSER_WAIT" 2>/dev/null || true' EXIT
fi
"$VPY" scripts/chat_web.py --model "$MODEL" --port "$PORT" "$@"
