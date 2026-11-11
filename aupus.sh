#!/usr/bin/env bash
# Chat with Aupus 0.1 Instruct in the browser, on Windows (Git Bash), Linux (also WSL) or macOS:
#
#   bash aupus.sh                 # then the browser opens http://127.0.0.1:8000
#   bash aupus.sh --seed 42       # extra arguments go to scripts/chat_web.py
#
# Safe to run every time: each step is skipped when it is already done, and repaired when it is
# half done.
#   1. Python >= 3.11 and a virtual environment (.venv). Works without Debian/Ubuntu's python3-venv
#      package (pip is fetched from bootstrap.pypa.io instead). A .venv made on another system
#      (Windows vs Linux, e.g. the same folder opened from WSL) is left alone: .venv-linux or
#      .venv-windows is used next to it.
#   2. PyTorch >= 2.4. One already in the environment is kept; one already installed in the system
#      Python is reused instead of downloading it again (USE_SYSTEM_TORCH=0 to turn that off).
#      Otherwise: the CUDA build when an NVIDIA GPU is found (CUDA 13.0 with driver >= 580, else
#      12.6), the CPU build without one, PyPI's default build on macOS.
#   3. Aupus and its dependencies (pip install -e .), again if the folder was moved or copied.
#   4. The model: release/aupus-0.1-instruct, exported from runs/aupus-0.1-instruct/best.pt if
#      missing. Models are not in git (too big): on a new computer, copy that folder over.
#   5. The browser chat (Ctrl-C to quit). In the page, the menu at the top switches models.
#
# Settings (environment variables): MODEL (release/aupus-0.1-instruct), PORT (8000), VENV (.venv),
# TORCH_INDEX (a PyTorch wheel index URL, to override the automatic choice), USE_SYSTEM_TORCH (1),
# NO_BROWSER=1 (don't open the browser).

# Started with `sh aupus.sh` (dash on Debian/Ubuntu)? Run again under bash: the rest needs it.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL=release/aupus-0.1-instruct
MODEL="${MODEL:-$DEFAULT_MODEL}"
PORT="${PORT:-8000}"
CHECKPOINT=runs/aupus-0.1-instruct/best.pt
URL="http://127.0.0.1:${PORT}"
MIN_TORCH=2.4
STEP="starting"
cd "$REPO"

case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) OS=windows ;;
    Darwin) OS=macos ;;
    *) OS=linux ;;
esac

die() { echo "error: $*" >&2; exit 1; }
note() { echo "aupus.sh: $*"; }
trap 'echo "aupus.sh: stopped while $STEP (see the error above)." >&2' ERR

# ---------------------------------------------------------------- 1. Python and the environment

venv_python_path() {  # where this system's Python lives inside a virtual environment
    if [ "$OS" = windows ]; then echo "$1/Scripts/python.exe"; else echo "$1/bin/python"; fi
}

choose_venv() {
    if [ -n "${VENV:-}" ]; then return; fi  # set by the user: use it as it is
    VENV=.venv
    local other
    if [ "$OS" = windows ]; then other=.venv/bin/python; else other=.venv/Scripts/python.exe; fi
    if [ -e "$other" ] && [ ! -e "$(venv_python_path .venv)" ]; then
        VENV=".venv-$OS"
        note ".venv was made on another system, so this one uses $VENV (the other is left as it is)."
    fi
}

python_hint() {
    case "$OS" in
        windows) echo "install it from https://www.python.org/downloads/ (tick \"Add python.exe to PATH\") or run: winget install Python.Python.3.12" ;;
        macos) echo "install it from https://www.python.org/downloads/ or run: brew install python@3.12" ;;
        *) echo "install it with your package manager, e.g. sudo apt install python3 python3-venv (Debian/Ubuntu)" ;;
    esac
}

find_python() {
    # The first Python >= 3.11. `py` is the Windows launcher; the Microsoft Store "python" stub
    # fails the version check, so it is skipped.
    local cmd
    for cmd in python3.13 python3.12 python3.11 python3 python "py -3"; do
        if $cmd -c 'import sys; sys.exit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
            echo "$cmd"
            return
        fi
    done
    return 1
}

torch_status() {
    # What the given Python has: ok, missing, old (< MIN_TORCH) or cpu (CPU-only build, but an
    # NVIDIA GPU is here). -s: ignore the user site, which a virtual environment doesn't see either.
    local status
    status="$("$@" -s -c '
import sys
try:
    import torch
except Exception:
    print("missing"); sys.exit()
major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
want = tuple(int(x) for x in sys.argv[1].split("."))
print("old" if (major, minor) < want else ("cuda" if torch.version.cuda else "cpu"))
' "$MIN_TORCH" 2>/dev/null)" || status=missing
    case "$status" in
        cpu) if [ "$OS" != macos ] && has_nvidia_gpu; then echo cpu; else echo ok; fi ;;
        cuda) echo ok ;;
        *) echo "${status:-missing}" ;;
    esac
}

has_nvidia_gpu() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }

bootstrap_pip() {
    # For a Python without ensurepip (Debian/Ubuntu without python3-venv): the official installer.
    local installer
    installer="$(mktemp)"
    note "fetching pip from https://bootstrap.pypa.io/get-pip.py..."
    if "$VPY" -c 'import sys, urllib.request; urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", sys.argv[1])' "$installer" \
        && "$VPY" "$installer" --quiet; then
        rm -f "$installer"
        return 0
    fi
    rm -f "$installer"
    return 1
}

no_pip_hint() {
    if [ "$OS" = linux ]; then
        local version
        version="$("$VPY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo 3)"
        echo "Install the venv package and run this again: sudo apt install python${version}-venv (Debian/Ubuntu)"
    else
        echo "Check the internet connection and run this again."
    fi
}

create_venv() {
    local py flags=() fresh=0
    [ -e "$VENV" ] || fresh=1  # only a folder made here may be removed again if this fails
    py="$(find_python)" || die "Python 3.11 or newer is needed: $(python_hint)"
    if [ "${USE_SYSTEM_TORCH:-1}" = 1 ] && [ "$(torch_status $py)" = ok ]; then
        note "reusing the PyTorch already installed in $($py -c 'import sys; print(sys.executable)') (USE_SYSTEM_TORCH=0 bash aupus.sh to install a separate one)."
        flags+=(--system-site-packages)
    fi
    note "creating the virtual environment in $VENV ($($py --version 2>&1))..."
    if $py -c 'import ensurepip' >/dev/null 2>&1; then
        $py -m venv ${flags[@]+"${flags[@]}"} "$VENV"
        VPY="$(venv_python_path "$VENV")"
    else
        note "this Python has no ensurepip (on Debian/Ubuntu it comes with the python3-venv package)."
        $py -m venv --without-pip ${flags[@]+"${flags[@]}"} "$VENV"
        VPY="$(venv_python_path "$VENV")"
        if ! bootstrap_pip; then
            # Just created and unusable without pip: don't leave it half made.
            if [ "$fresh" = 1 ]; then rm -rf "$VENV"; fi
            die "could not install pip into $VENV. $(no_pip_hint)"
        fi
    fi
    "$VPY" -m pip install --quiet --upgrade pip
}

ensure_venv() {
    STEP="setting up Python"
    choose_venv
    VPY="$(venv_python_path "$VENV")"
    if [ ! -e "$VPY" ] && [ ! -L "$VPY" ]; then  # -L: a link to a Python that is gone still counts
        [ -e "$VENV" ] && [ ! -f "$VENV/pyvenv.cfg" ] \
            && die "$VENV exists but is not a virtual environment: move it away, or set VENV=another-folder"
        create_venv
        return
    fi
    if ! "$VPY" -c 'import sys' >/dev/null 2>&1; then
        # Its Python was uninstalled or upgraded. The default environment is ours: rebuild it.
        [ "$VENV" = .venv ] || [ "$VENV" = ".venv-$OS" ] \
            || die "the Python in $VENV no longer runs (was it moved or uninstalled?). Delete $VENV or set VENV=another-folder."
        note "the Python in $VENV no longer runs (uninstalled or upgraded?): rebuilding it."
        rm -rf "$VENV"
        create_venv
        return
    fi
    "$VPY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
        || die "$VENV has Python $("$VPY" -c 'import platform; print(platform.python_version())'); Aupus needs 3.11 or newer. Delete $VENV (or set VENV=another-folder) and run this again."
    if ! "$VPY" -m pip --version >/dev/null 2>&1; then
        note "$VENV has no pip (it was left half made): adding it."
        "$VPY" -m ensurepip --upgrade >/dev/null 2>&1 || bootstrap_pip || die "could not install pip into $VENV. $(no_pip_hint)"
    fi
}

# ---------------------------------------------------------------- 2. PyTorch

torch_index() {
    # The PyTorch wheel index for this machine. CUDA 13.0 wheels need an NVIDIA driver >= 580.
    if [ -n "${TORCH_INDEX:-}" ]; then echo "$TORCH_INDEX"; return; fi
    [ "$OS" = macos ] && return  # PyPI's default build (CPU + Apple GPU)
    if has_nvidia_gpu; then
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

warn_low_disk() {  # $1: GB the download and unpacking need
    local free_kb
    free_kb="$(df -Pk "$VENV" 2>/dev/null | awk 'NR == 2 { print $4 }')" || return 0
    if [ -n "$free_kb" ] && [ "$free_kb" -lt $(($1 * 1024 * 1024)) ] 2>/dev/null; then
        note "warning: only $((free_kb / 1024 / 1024)) GB free here; installing PyTorch needs about $1 GB."
    fi
}

install_torch() {  # $@: extra pip options
    local index
    index="$(torch_index)"
    case "$index" in
        "${TORCH_INDEX:-none}") note "installing PyTorch from $index (TORCH_INDEX)..."; warn_low_disk 6 ;;
        *cu1*) note "installing PyTorch for NVIDIA GPUs (${index##*/}; a ~2.5 GB download)..."; warn_low_disk 6 ;;
        *cpu) note "installing PyTorch for the CPU (no NVIDIA GPU found; ~200 MB)..."; warn_low_disk 2 ;;
        *) note "installing PyTorch (${index:-PyPI default})..."; warn_low_disk 2 ;;
    esac
    if [ -n "$index" ]; then
        "$VPY" -m pip install "$@" torch --index-url "$index"
    else
        "$VPY" -m pip install "$@" torch
    fi
}

ensure_torch() {
    STEP="installing PyTorch (check the internet connection)"
    case "$(torch_status "$VPY")" in
        ok) ;;
        missing) install_torch ;;
        old)
            # Upgrade it here, with the right build: otherwise `pip install -e .` would, from PyPI.
            note "PyTorch in $VENV is older than $MIN_TORCH: upgrading it."
            install_torch --upgrade
            ;;
        cpu)
            note "the PyTorch in $VENV runs on the CPU only, but this computer has an NVIDIA GPU, so replies will be slower."
            note "to use the GPU: \"$VPY\" -m pip install --force-reinstall torch --index-url $(torch_index)"
            ;;
    esac
}

# ---------------------------------------------------------------- 3. Aupus

ensure_aupus() {
    STEP="installing Aupus's dependencies (check the internet connection)"
    # -I: the current folder must not satisfy `import aupus`; the scripts need it installed. It must
    # also be this folder's copy: an install from before the folder was moved or copied points elsewhere.
    if "$VPY" -I -c '
import os, sys
import aupus, numpy, safetensors, tqdm, yaml
here = os.path.realpath(os.path.join(os.getcwd(), "aupus"))
sys.exit(os.path.realpath(os.path.dirname(aupus.__file__)) != here)
' >/dev/null 2>&1; then
        return
    fi
    note "installing Aupus and its dependencies..."
    "$VPY" -m pip install --quiet -e .
}

# ---------------------------------------------------------------- 4. the model

check_model() {
    # Before anything is installed, so a computer without the model doesn't download PyTorch first.
    STEP="looking for the model"
    if [ -d "$MODEL" ]; then
        local f missing=()
        for f in model.safetensors model.yaml tokenizer.json release.json; do
            [ -f "$MODEL/$f" ] || missing+=("$f")
        done
        [ ${#missing[@]} -eq 0 ] \
            || die "$MODEL is incomplete (missing: ${missing[*]}). Copy the whole folder again from the computer that has it."
        return
    fi
    [ -f "$MODEL" ] && return  # a checkpoint .pt
    [ "$MODEL" = "$DEFAULT_MODEL" ] && [ -f "$CHECKPOINT" ] && return  # exported once Python is ready
    if [ "$MODEL" = "$DEFAULT_MODEL" ]; then
        die "the model isn't here: $REPO/$MODEL
  Trained models are not in git (release/ and runs/ are too big for it). Copy the folder
  release/aupus-0.1-instruct from the computer that has it into release/ here, and run this again.
  (Or train it: docs/TRAINING.md.)"
    fi
    die "model not found: $MODEL"
}

ensure_model() {
    [ -e "$MODEL" ] && return
    STEP="exporting the model"
    note "exporting $CHECKPOINT to $MODEL..."
    "$VPY" scripts/export_release.py --checkpoint "$CHECKPOINT" --out "$MODEL"
}

# ---------------------------------------------------------------- 5. the chat

report_device() {
    "$VPY" -c '
import subprocess
import torch
if torch.cuda.is_available():
    print("PyTorch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
elif torch.version.cuda:
    try:
        subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=True)
        gpu = True
    except Exception:
        gpu = False
    if gpu:
        print("PyTorch", torch.__version__, f"| the GPU is not usable (this PyTorch needs CUDA {torch.version.cuda}:",
              "update the NVIDIA driver, or reinstall with TORCH_INDEX=https://download.pytorch.org/whl/cu126):",
              "replies run on the CPU")
    else:
        print("PyTorch", torch.__version__, "| no GPU found: replies run on the CPU")
else:
    print("PyTorch", torch.__version__, "| CPU")
'
}

port_open() {
    "$VPY" -c "import socket, sys; s = socket.socket(); s.settimeout(0.5); sys.exit(s.connect_ex(('127.0.0.1', $PORT)))" 2>/dev/null
}

open_browser() {
    if command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$URL" || true  # Windows, and WSL
    elif [ "$OS" = macos ]; then open "$URL"
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 || true
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

main() {
    check_model
    ensure_venv
    ensure_torch
    ensure_aupus
    ensure_model
    STEP="starting the chat"
    report_device
    port_open && die "port $PORT is already in use (is Aupus already running? then open $URL), or pick another: PORT=$((PORT + 1)) bash aupus.sh"
    if [ "${NO_BROWSER:-0}" != 1 ]; then
        open_when_ready &
        BROWSER_WAIT=$!
        trap 'kill "$BROWSER_WAIT" 2>/dev/null || true' EXIT
    fi
    trap - ERR  # from here on errors are the chat's own, and it reports them
    "$VPY" scripts/chat_web.py --model "$MODEL" --port "$PORT" "$@"
}

main "$@"
