#!/usr/bin/env bash
# Prepare a RunPod pod for Vulpes 0.1 Base pretraining (docs/RUNPOD.md). Run it once, inside the pod:
#
#   cd /workspace && git clone https://github.com/eyboss202/Aupus-AI.git && cd Aupus-AI
#   bash scripts/runpod/setup.sh
#
# It runs inside tmux session "vulpes-setup" (an SSH drop doesn't stop it; `tmux attach -t vulpes-setup`
# comes back) and is safe to run again: finished steps are skipped. ~30-45 min, mostly building the training data, at ~0.34 USD/h on an RTX 4090:
#   1. tmux and the GPU            4. training data: download FineWeb-Edu, build the shards, check
#   2. Python env in /workspace       them against the PC's (docs/PLAN.md §16), delete the download
#   3. the test suite (incl. CUDA)  5. speed test: real tok/s, and the hours and USD for 2.5B tokens
#
# Settings (environment variables): PRICE_PER_HOUR (0.34, what the pod costs in USD),
# VENV (/workspace/venv), WORKERS (data-prep processes; default: the pod's CPUs, at most 16).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-/workspace/venv}"
PRICE_PER_HOUR="${PRICE_PER_HOUR:-0.34}"
DATA=data/pretrain/fineweb_edu
RAW=data/raw/fineweb_edu
cd "$REPO"

ensure_tmux() {
    # apt packages live on the container disk, which RunPod wipes on every restart.
    command -v tmux >/dev/null && return
    echo "installing tmux..."
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux >/dev/null
}

if [ -z "${TMUX:-}" ] && [ "${VULPES_NO_TMUX:-0}" != 1 ]; then
    ensure_tmux
    settings="VENV=$(printf %q "$VENV") PRICE_PER_HOUR=$(printf %q "$PRICE_PER_HOUR") WORKERS=$(printf %q "${WORKERS:-}")"
    # -A: if the setup session already exists (e.g. still building the data), attach to it instead.
    inner="cd $(printf %q "$REPO") && env $settings bash scripts/runpod/setup.sh; exec bash"
    exec tmux new-session -A -s vulpes-setup "exec bash -c $(printf %q "$inner")"
fi

step() { printf '\n=== %s ===\n' "$*"; }

pod_cpus() {
    # nproc can report the whole host; the container's real share is its cgroup CPU quota.
    local n quota period
    n=$(nproc)
    if read -r quota period </sys/fs/cgroup/cpu.max 2>/dev/null && [ "$quota" != max ]; then
        quota=$(((quota + period - 1) / period))
        [ "$quota" -lt "$n" ] && n=$quota
    fi
    [ "$n" -gt 16 ] && n=16
    echo "$n"
}

# PyTorch and NumPy start one thread per CPU they *see*: on a pod that is the whole host (e.g. 128),
# while the container may use ~14. That many threads thrash, and CPU work (tests, the tiny models
# in them) runs ~100x slower. Limit them to the pod's real share.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(pod_cpus)}"

step "1/5 tmux and the GPU"
ensure_tmux
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "CPUs for this pod: $(pod_cpus) of $(nproc) visible (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
df -h /workspace | tail -1 | awk '{print "disk /workspace: " $4 " free of " $2}'

step "2/5 Python environment ($VENV)"
if [ ! -x "$VENV/bin/python" ]; then
    # --system-site-packages reuses the template's CUDA PyTorch (saves a ~3 GB download).
    # The venv is on /workspace, so it survives a pod stop.
    python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python - <<'EOF'
import sys

import torch

print(f"python {sys.version.split()[0]} | torch {torch.__version__} | CUDA {torch.version.cuda}")
problems = []
if sys.version_info < (3, 11):
    problems.append("Vulpes needs Python 3.11+: pick a newer RunPod PyTorch template")
if tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2]) < (2, 4):
    problems.append("Vulpes needs PyTorch 2.4+: pick a newer RunPod PyTorch template")
if not torch.cuda.is_available():
    problems.append("PyTorch sees no GPU")
else:
    cc = torch.cuda.get_device_capability()
    print(f"GPU {torch.cuda.get_device_name()} | compute capability {cc[0]}.{cc[1]} | bf16 {cc >= (8, 0)}")
    if cc < (8, 0):
        problems.append("this GPU has no bf16: small-runpod.yaml expects an Ampere-or-newer card")
if problems:
    sys.exit("\n".join(f"ERROR: {p}" for p in problems))
EOF
pip install -q -e ".[data,dev]"

step "3/5 tests"
python -m pytest -q

step "4/5 training data ($DATA)"
if [ -f "$DATA/meta.json" ]; then
    echo "already built (or copied from the PC): checking it"
else
    workers="${WORKERS:-$(pod_cpus)}"
    python scripts/download_fineweb_edu.py --files 4 --out-dir "$RAW"
    echo "building the shards with $workers workers (~15-30 min; it does not resume, so let it finish)"
    python scripts/prepare_data.py --source parquet --input "$RAW" --out "$DATA" --workers "$workers"
fi
# The run may have to finish on the PC (docs/TRAINING.md, "Moving a run to another machine"),
# which needs exactly the same windows. These are the counts of the PC's build (docs/PLAN.md §16).
python - "$DATA" <<'EOF'
import json
import sys
from pathlib import Path

from vulpes.data.loader import PretrainLoader

data = Path(sys.argv[1])
splits = json.loads((data / "meta.json").read_text())["splits"]
found = {
    "train tokens": sum(s["tokens"] for s in splits["train"]),
    "val tokens": sum(s["tokens"] for s in splits["val"]),
    "train shards": len(splits["train"]),
    "train windows of 1,024": PretrainLoader(data, "train", 1, 1024).num_windows,
}
expected = {
    "train tokens": 2_936_424_290,
    "val tokens": 14_644_688,
    "train shards": 30,
    "train windows of 1,024": 2_867_586,
}
for key, want in expected.items():
    mark = "ok" if found[key] == want else f"DIFFERENT (the PC has {want:,})"
    print(f"  {key:<24} {found[key]:>15,}  {mark}")
if found != expected:
    sys.exit(
        "ERROR: these shards differ from the PC's, so the run could not be finished on the PC.\n"
        "Copy the PC's shards here instead (on the PC, from the Aupus-AI folder):\n"
        "  scp -P <port> -i ~/.ssh/id_ed25519 -r data/pretrain/fineweb_edu root@<ip>:/workspace/Aupus-AI/data/pretrain/\n"
        "after deleting this build: rm -rf data/pretrain/fineweb_edu (then run setup.sh again)"
    )
print("same data as the PC")
EOF
if [ -d "$RAW" ]; then
    rm -rf "$RAW"  # 8.6 GB; scripts/download_fineweb_edu.py fetches it again if ever needed
    echo "deleted the download ($RAW)"
fi

step "5/5 speed test (small, bf16, micro-batch 16, torch.compile; ~2 min)"
result=$(python scripts/benchmark_throughput.py --device cuda --presets small --micro-batch 16 --dtype bf16 \
    --compile --steps 20 --warmup 5 --plan-tokens 2.5e9 --usd-per-hour "$PRICE_PER_HOUR" | tail -1)
echo "$result"
python - "$result" "$PRICE_PER_HOUR" <<'EOF'
import json
import sys

r, price = json.loads(sys.argv[1]), float(sys.argv[2])
if "error" in r:
    sys.exit(
        "ERROR: micro-batch 16 doesn't fit on this GPU. Use micro_batch=8 grad_accum=32 eval_batches=100:\n"
        "  bash scripts/runpod/train.sh micro_batch=8 grad_accum=32 eval_batches=100"
    )
hours = r["plan_hours"] * 1.05  # + evaluations, samples and checkpoints
print(
    f"\n  speed       {r['tokens_per_sec']:,.0f} tok/s ({r['achieved_tflops']:.0f} TFLOPS), "
    f"peak {r['peak_vram_gib']:.1f} GiB of GPU memory"
)
print(f"  full run    2.5B tokens ~ {hours:.1f} h ~ {hours * price:.2f} USD at {price} USD/h")
print(
    "  budget      train.sh stops cleanly after MAX_HOURS (default 20 h = "
    f"{20 * price:.2f} USD); set it to what your balance allows, minus ~1 USD"
)
EOF

ip="${RUNPOD_PUBLIC_IP:-<ip>}"
port="${RUNPOD_TCP_PORT_22:-<port>}"
cat <<EOF

=== ready ===
Start training (it runs in tmux; Ctrl-b d leaves it running):
  bash scripts/runpod/train.sh
On the PC, from the Aupus-AI folder, keep a copy of the run as it trains:
  python scripts/runpod/sync.py "ssh root@$ip -p $port -i ~/.ssh/id_ed25519"
EOF
