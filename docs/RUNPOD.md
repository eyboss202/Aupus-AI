# Training Aupus 0.1 Base on RunPod

The full pretraining run (`small`, 125M parameters, 2.5B tokens) on one rented RTX 4090,
for about **5–6 USD of a 10 USD budget**. Three scripts do the work:

| Script | Where | What it does |
|---|---|---|
| `scripts/runpod/setup.sh` | pod, once | tmux, a Python env on `/workspace`, the tests (CUDA ones included), the training data (built on the pod and checked against the PC's), a speed test with the projected hours and cost |
| `scripts/runpod/train.sh` | pod | trains with `configs/train/small-runpod.yaml` (or resumes), stops cleanly after `MAX_HOURS`, evaluates the finished model, then stops the pod once the PC has a checked copy |
| `scripts/runpod/sync.py` | PC | copies the run to the PC every 30 min while it trains, then makes a last copy, checks it with SHA-256 and tells the pod it may stop |

`small-runpod.yaml` is the **same run** as `configs/train/small.yaml` (same name, data, order,
262,144 tokens per step, 9,537 steps, learning-rate curve; a test checks this). Only the memory
settings (16 × 16 sequences instead of 1 × 256, bf16, no checkpointing) and the save/eval/log
frequencies differ. So a run started on the pod can always be finished on the GTX 1650 Super.

## Costs

| | Time on an RTX 4090 | Cost at 0.34 USD/h |
|---|---|---|
| `setup.sh` (data build + tests + speed test) | ~30–45 min | ~0.25 USD |
| Pretraining, 2.5B tokens | ~13–17 h (estimate; `setup.sh` measures it) | ~4.40–5.80 USD |
| Evaluation + waiting for the PC's copy | ~5–20 min | ~0.10 USD |
| **Total** | | **~5–6.50 USD** |

The hours are estimates until `setup.sh` prints the measured speed. RunPod is prepaid: load 10 USD
without auto-reload and you can't spend more. Check the 4090's price on the deploy screen
(Community Cloud, On-Demand) and pass it as `PRICE_PER_HOUR=...` to `setup.sh`.

## Step by step

**Before:** push the repository to GitHub (the pod clones it), and add your SSH public key in
RunPod → Settings → SSH Public Keys *before* creating the pod.

1. **Create the pod:** Pods → Deploy → Community Cloud → RTX 4090 × 1, template "RunPod PyTorch"
   (Python 3.11+, PyTorch 2.4+), Container Disk 20 GB, Volume Disk 40 GB at `/workspace`,
   On-Demand (Spot can be cut off mid-run). Under Connect, check that "SSH over exposed TCP" is
   offered; if not, the machine has no public IP: terminate it and pick another.
2. **Set it up** (in the pod, over SSH):

   ```bash
   cd /workspace && git clone https://github.com/eyboss202/Aupus-AI.git && cd Aupus-AI
   PRICE_PER_HOUR=0.34 bash scripts/runpod/setup.sh
   ```

   It runs in tmux (`Ctrl-b d` leaves it, `tmux attach -t aupus-setup` comes back) and ends with
   the measured speed, the projected hours and cost, and the two commands below.
3. **Train:**

   ```bash
   MAX_HOURS=20 bash scripts/runpod/train.sh
   ```

   `MAX_HOURS` is the budget guard: after that many hours of training the run saves and stops
   (20 h ≈ 6.80 USD). Set it to what your balance allows, minus ~1 USD. Leave tmux with `Ctrl-b d`;
   you can close SSH and switch off the PC.
4. **Copy it to the PC as it trains** (on the PC, in the Aupus-AI folder, with the "SSH over
   exposed TCP" command from Connect):

   ```powershell
   python scripts/runpod/sync.py "ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519"
   ```

   It keeps `runs/aupus-0.1-base/` on the PC up to date (newest checkpoint, `best.pt`, logs,
   metrics, TensorBoard). If the PC is off, nothing breaks; it catches up when it runs again.
5. **The end:** `train.sh` evaluates `best.pt` (results in `runs/aupus-0.1-base/eval_best.json`),
   then waits up to 2 h (`WAIT_FOR_PC_MIN`) for `sync.py` to copy and check everything, then stops
   the pod. When `sync.py` prints "All copied and checked", **terminate the pod** (Pods → ⋮ →
   Terminate): a stopped pod still pays for its disk (40 GB ≈ 0.27 USD per day).

## Stops, restarts and running out of budget

| What happened | What you lose | What to do |
|---|---|---|
| SSH dropped | nothing (tmux) | reconnect, `tmux attach -t aupus` |
| You pressed Ctrl-C once in tmux | nothing: it finishes the step and saves | `bash scripts/runpod/train.sh` |
| `MAX_HOURS` reached | nothing: saved | more budget: `train.sh` again; or finish on the PC (below) |
| Pod stopped (Stop button, balance, RunPod) | steps since the last save (≤ ~10 min) | Start the pod, `bash scripts/runpod/train.sh` (it reinstalls tmux and resumes) |
| Machine gone, or pod terminated | everything after the PC's last copy | new pod, `setup.sh`, copy the run back (`scp -P <port> -r runs/aupus-0.1-base root@<ip>:/workspace/Aupus-AI/runs/`), `train.sh` |

`train.sh` always resumes the newest checkpoint; extra `key=value` arguments go to
`scripts/train.py` (for example `max_steps=...` decided *before* starting, since the learning-rate
curve depends on it).

**Finishing on the PC.** Once `sync.py` has the newest checkpoint, the run continues on the GTX 1650
Super with the same data and order (docs/TRAINING.md, "Moving a run to another machine"):

```powershell
python scripts/train.py --resume runs/aupus-0.1-base micro_batch=1 grad_accum=256 eval_batches=800 grad_checkpointing=true cuda_memory_limit_gib=2.9 save_every=10 eval_every=40 sample_every=200 log_every=1
```

`setup.sh` refuses data that differs from the PC's (the token and window counts of docs/PLAN.md
§16), because the PC resume needs exactly the same windows. If that happens, copy the PC's
`data/pretrain/fineweb_edu/` to the pod instead, as the error message says.

## What happens on the pod, and why

- **Data on the pod:** downloading 8.6 GB in the datacenter takes minutes, faster than uploading
  5.5 GB of shards from home. The build is deterministic, so it gives the PC's shards; the check
  in `setup.sh` confirms it. The download is deleted afterwards.
- **Python env on `/workspace`:** the Volume Disk survives a pod stop; the Container Disk (and
  anything `apt-get` installed, like tmux) doesn't. `--system-site-packages` reuses the template's
  CUDA PyTorch.
- **Clean stops:** `train.py --stop-after` (used by `MAX_HOURS`), Ctrl-C and `kill <pid>`
  (SIGTERM) all finish the current step and save. RunPod's own Stop button kills the process
  without saving, so stop from tmux first when you can.
- **Auto-stop:** the pod stops itself with `runpodctl stop pod $RUNPOD_POD_ID` after the PC's
  checked copy (or after `WAIT_FOR_PC_MIN`), so an idle GPU never eats the budget; also when
  training failed, so an error at night costs at most that wait (2 h ≈ 0.70 USD), not the rest
  of the night. `AUTO_STOP=0` turns it off.
