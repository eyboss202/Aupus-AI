#!/usr/bin/env bash
# Pretrain Aupus 0.1 Base on a RunPod pod, evaluate it, and stop the pod (docs/RUNPOD.md).
# Run after scripts/runpod/setup.sh. The same command starts the run and resumes it after any stop:
#
#   bash scripts/runpod/train.sh                    # start, or continue from the newest checkpoint
#   MAX_HOURS=12 bash scripts/runpod/train.sh       # stop cleanly (saved) after 12 h of training
#   bash scripts/runpod/train.sh max_steps=5767     # key=value overrides go to scripts/train.py
#
# What it does:
#   * runs in a tmux session "aupus" (Ctrl-b d leaves it running; `tmux attach -t aupus` comes back);
#     running this again while training runs just attaches to it;
#   * trains with configs/train/small-runpod.yaml (or --resume runs/aupus-0.1-base), stopping cleanly
#     after MAX_HOURS: a checkpoint is saved, and the run continues later, here or on the PC;
#   * when the run is finished, scores best.pt (val perplexity, HellaSwag, ARC-Easy, PIQA; ~5 min);
#   * writes runs/aupus-0.1-base/POD_DONE, then waits up to WAIT_FOR_PC_MIN minutes for the PC's
#     scripts/runpod/sync.py to copy everything and confirm (it writes PULLED), then stops the pod,
#     so an idle GPU never eats the budget. Ctrl-C during that wait cancels the stop.
#
# Settings (environment variables): MAX_HOURS (20), WAIT_FOR_PC_MIN (120), AUTO_STOP (1; 0 = never
# stop the pod), EVAL (1; 0 = skip the evaluation), VENV (/workspace/venv). CONFIG, RUN_DIR and DATA
# change what is trained (tests/test_runpod.py runs this script on a tiny model with them).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAX_HOURS="${MAX_HOURS:-20}"
WAIT_FOR_PC_MIN="${WAIT_FOR_PC_MIN:-120}"
AUTO_STOP="${AUTO_STOP:-1}"
EVAL="${EVAL:-1}"
VENV="${VENV:-/workspace/venv}"
RUN_DIR="${RUN_DIR:-runs/aupus-0.1-base}"
CONFIG="${CONFIG:-configs/train/small-runpod.yaml}"
DATA="${DATA:-data/pretrain/fineweb_edu}"
OUT="$(dirname "$RUN_DIR")/pod-console.log"  # everything printed, across sessions (train.log has the run's own lines)
LOCK="$(dirname "$RUN_DIR")/.pod-train.lock"  # held while this script runs, so it never runs twice
cd "$REPO"
mkdir -p "$RUN_DIR"

training_running() { ! flock -n "$LOCK" true; }

if [ -z "${TMUX:-}" ] && [ "${AUPUS_NO_TMUX:-0}" != 1 ]; then
    if ! command -v tmux >/dev/null; then  # wiped with the container disk on every pod restart
        apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux >/dev/null
    fi
    if tmux has-session -t aupus 2>/dev/null; then
        if training_running; then
            echo "train.sh is already running: attaching (Ctrl-b d to leave it)"
            exec tmux attach -t aupus
        fi
        tmux kill-session -t aupus
    fi
    settings="MAX_HOURS=$(printf %q "$MAX_HOURS") WAIT_FOR_PC_MIN=$(printf %q "$WAIT_FOR_PC_MIN")"
    settings+=" AUTO_STOP=$(printf %q "$AUTO_STOP") EVAL=$(printf %q "$EVAL") VENV=$(printf %q "$VENV")"
    settings+=" RUN_DIR=$(printf %q "$RUN_DIR") CONFIG=$(printf %q "$CONFIG") DATA=$(printf %q "$DATA")"
    args=""
    [ $# -gt 0 ] && args=$(printf ' %q' "$@")
    # `exec bash -c`: bash (not whatever tmux's default shell is) runs it, so Ctrl-C behaves as tested;
    # the final `exec bash` keeps the window open with a shell when the script ends.
    inner="cd $(printf %q "$REPO") && env $settings bash scripts/runpod/train.sh$args; exec bash"
    exec tmux new-session -s aupus "exec bash -c $(printf %q "$inner")"
fi

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "ERROR: train.sh is already running (tmux attach -t aupus)"
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate" || { echo "ERROR: no Python env at $VENV: run scripts/runpod/setup.sh"; exit 1; }
[ -f "$DATA/meta.json" ] || { echo "ERROR: no training data in $DATA: run scripts/runpod/setup.sh"; exit 1; }

rm -f "$RUN_DIR/POD_DONE" "$RUN_DIR/PULLED"
if compgen -G "$RUN_DIR/step_*.pt" >/dev/null; then
    start=(--resume "$RUN_DIR")
else
    start=(--config "$CONFIG")
fi
export PYTHONUNBUFFERED=1
# One thread per CPU the pod may use, not per CPU of the host (see setup.sh).
if [ -z "${OMP_NUM_THREADS:-}" ]; then
    OMP_NUM_THREADS=$(nproc)
    if read -r quota period </sys/fs/cgroup/cpu.max 2>/dev/null && [ "$quota" != max ]; then
        OMP_NUM_THREADS=$(((quota + period - 1) / period))
    fi
    export OMP_NUM_THREADS
fi

echo "=== training: python scripts/train.py ${start[*]} --stop-after $MAX_HOURS $* ($(date -u '+%F %T') UTC)" | tee -a "$OUT"
# tee -i: Ctrl-C must reach only the trainer (which finishes the step and saves), not the pipe.
python scripts/train.py "${start[@]}" --stop-after "$MAX_HOURS" "$@" 2>&1 | tee -i -a "$OUT"
code=${PIPESTATUS[0]}

case $code in
    0)
        status="finished: the run reached max_steps"
        if [ "$EVAL" = 1 ]; then
            echo "=== evaluating best.pt (~5 min)" | tee -a "$OUT"
            # pipefail: a pipeline fails when python does, even though tee succeeds.
            if python scripts/download_eval_data.py 2>&1 | tee -i -a "$OUT"; then
                python scripts/evaluate.py --model "$RUN_DIR/best.pt" --val-data "$DATA" 2>&1 | tee -i -a "$OUT" ||
                    status+=" (evaluation failed; the model is fine: evaluate it on the PC)"
            else
                status+=" (evaluation data download failed; evaluate on the PC)"
            fi
        fi
        ;;
    130)
        status="stopped before the end (Ctrl-C, --stop-after ${MAX_HOURS} h or SIGTERM), checkpoint saved: continue with bash scripts/runpod/train.sh, or on the PC"
        ;;
    *)
        status="failed with exit code $code: see $RUN_DIR/train.log and $OUT"
        ;;
esac
printf '%s\n%s UTC\n' "$status" "$(date -u '+%F %T')" >"$RUN_DIR/POD_DONE"
echo "=== $status" | tee -a "$OUT"

stop_pod() {
    if [ -z "${RUNPOD_POD_ID:-}" ] || ! command -v runpodctl >/dev/null; then
        echo "not on a RunPod pod (or no runpodctl): stop it yourself in the web console"
        return
    fi
    echo "stopping pod $RUNPOD_POD_ID (the /workspace disk is kept; terminate it once the PC has the run)"
    runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID" ||
        echo "runpodctl failed: stop the pod yourself in the web console (Pods -> Stop)"
}

if [ "$AUTO_STOP" != 1 ]; then
    echo "AUTO_STOP=$AUTO_STOP: leaving the pod running (it costs money until you stop it)"
    exit "$code"
fi
ip="${RUNPOD_PUBLIC_IP:-<ip>}"
port="${RUNPOD_TCP_PORT_22:-<port>}"
cat <<EOF
The pod stops in $WAIT_FOR_PC_MIN min, or as soon as the PC has a checked copy of the run. On the PC:
  python scripts/runpod/sync.py "ssh root@$ip -p $port -i ~/.ssh/id_ed25519"
Ctrl-C now cancels the automatic stop.
EOF
deadline=$((SECONDS + WAIT_FOR_PC_MIN * 60))
while [ ! -f "$RUN_DIR/PULLED" ] && [ "$SECONDS" -lt "$deadline" ]; do
    sleep 20
done
if [ -f "$RUN_DIR/PULLED" ]; then
    echo "the PC confirmed its copy ($(cat "$RUN_DIR/PULLED"))"
else
    echo "no copy from the PC after $WAIT_FOR_PC_MIN min: stopping anyway; start the pod again to copy the run"
fi
stop_pod
exit "$code"
