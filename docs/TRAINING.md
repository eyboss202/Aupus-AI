# Training Aupus yourself

A step-by-step guide to training Aupus 0.1 on your own PC, from a fresh clone to
chatting with the finished model. Everything is a plain Python script in this
repository: no cloud service and no AI assistant is needed.

The numbers below were measured on the reference machine, a Windows 11 desktop with an
**NVIDIA GTX 1650 Super (4 GB)**, an Intel i5-9400F and 16 GB of RAM (docs/PLAN.md §12–§16).
Other NVIDIA GPUs work the same way; faster ones just finish sooner. A CPU works too,
but only for the small test presets.

## The whole pipeline at a glance

| Step | Command | Time on a GTX 1650 Super |
|---|---|---|
| 0. Set up | `pip install ...` (below) | ~10 min |
| 1. Training data | `download_fineweb_edu.py`, then `prepare_data.py` | ~25 min download + ~70 min CPU |
| 2. Pretraining (Base) | `train.py` | **days** (see "How long it takes"); stops and resumes freely |
| 3. Evaluation | `evaluate.py` | under an hour |
| 4. Fine-tuning (Instruct) | `prepare_sft_data.py`, then `sft.py` | a few hours |
| 5. Release and chat | `export_release.py`, then `chat.py` | minutes |

Every command below is run from the repository folder, in a terminal where the virtual
environment is active.

## 0. Set up (once per PC)

You need Python 3.11 or newer and, for a GPU, a recent NVIDIA driver.

```powershell
git clone https://github.com/eyboss202/Aupus-AI.git
cd Aupus-AI
python -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate

# PyTorch first, matching your hardware (see https://pytorch.org/get-started/locally/):
pip install torch --index-url https://download.pytorch.org/whl/cu130   # NVIDIA GPU
# pip install torch --index-url https://download.pytorch.org/whl/cpu   # no GPU

pip install -e ".[data,dev]"
pytest                                       # ~200 tests, under a minute; all should pass
python scripts/check_env.py                  # what your machine has (GPU, VRAM, RAM, disk)
```

Every new terminal needs `.venv\Scripts\activate` again before the commands below.

**Windows:** if a script prints garbled characters or a `UnicodeEncodeError`, run it as
`python -X utf8 scripts\...`.

**Disk:** about 20 GB free is needed (8.6 GB download, 5.5 GB of token shards, ~6 GB of
checkpoints). The download can be deleted after step 1.

## 1. Build the training data

```powershell
python scripts/download_fineweb_edu.py --files 4
python scripts/prepare_data.py --source parquet --input data/raw/fineweb_edu --out data/pretrain/fineweb_edu --workers 3
```

- The download (8.6 GB) checks every file against its published SHA-256. If it is
  interrupted, run the same command again: finished files are skipped.
- Processing cleans, filters and deduplicates ~2.9M web pages and tokenizes them into
  `data/pretrain/fineweb_edu/` (2.94B tokens). `--workers` sets how many CPU cores it
  uses: more is faster, fewer keeps the PC cooler and quieter. The result is identical
  either way. This step does *not* resume, so an interruption means running it again.

## 2. Pretraining: Aupus 0.1 Base

### Start the run (once)

```powershell
python scripts/train.py --config configs/train/small.yaml
```

Everything the run produces goes to `runs/aupus-0.1-base/`: the log (`train.log`),
metrics, checkpoints (`step_NNNNNN.pt`) and `best.pt`, the checkpoint with the lowest
validation loss.

### Stopping and continuing

You can stop as often as you like; the model ends up exactly the same as in a run that
never stopped (tested bit for bit, docs/PLAN.md §15).

- **To stop:** press **Ctrl-C once** in the training window. It finishes the current
  step, saves a checkpoint and exits. With `small` one step is ~2.5 minutes, so this can
  take up to ~3 minutes: wait for the `saved ...` line. (Pressing it twice aborts at once
  without saving, losing up to `save_every` steps, ~25 min.)
- **A crash, power cut or forced shutdown** is also fine: you only lose the steps since
  the last checkpoint.
- **To continue:** always the same command, however many times you've stopped:

```powershell
python scripts/train.py --resume runs/aupus-0.1-base
```

Don't use `--config` again for a run that already exists: it refuses, so a run is never
overwritten by accident.

### Moving a run to another machine

A run can start on one machine and finish on another, for example a rented GPU first and
this PC afterwards. Copy the whole run folder (`runs/aupus-0.1-base/`, at least the newest
`step_NNNNNN.pt`, `config.yaml` and `model.yaml`), make sure the new machine has the same
`data/pretrain/fineweb_edu/` (copied, or rebuilt with step 1: the result is identical),
and resume there.

If the new GPU has less memory, give it a smaller `micro_batch` and a bigger `grad_accum`
with the **same product**, so each step still trains on the same 256 sequences in the same
order. Also scale `eval_batches` so validation still reads the same sequences
(`micro_batch × eval_batches` unchanged), and set the memory settings and the save/eval
frequencies back to this card's. A run started at 16 × 16 with `eval_batches=50`
continues on the GTX 1650 Super with:

```powershell
python scripts/train.py --resume runs/aupus-0.1-base micro_batch=1 grad_accum=256 eval_batches=800 grad_checkpointing=true cuda_memory_limit_gib=2.9 save_every=10 eval_every=40 sample_every=200 log_every=1
```

Training refuses a resume that would change the sequences per step or the validation set,
and says which value to pass. The new values are saved in the run's `config.yaml`, so later
resumes on this PC are the plain `--resume` command again. The model after the move isn't
bit-identical to a run that stayed on one machine (the sums run in a different order, and
a newer GPU trains in bf16 where this card uses fp32), but the data, its order and the
learning-rate schedule are exactly the same. Resuming with the same `micro_batch` is
still bit-identical.

### Watching it

In a **second** terminal (activate the environment there too), this shows how far the run
is (% done, time left), the losses, the last checkpoint and what the GPU is doing. It only
reads files, so it never disturbs training:

```powershell
python scripts/train_status.py              # once
python scripts/train_status.py --watch 30   # refresh every 30 s; Ctrl-C here only quits the status
```

For CPU, RAM and disk use, Windows' Task Manager (Ctrl+Shift+Esc) → Performance shows all of
it, including "GPU 0" with its temperature. Training keeps the GPU at ~100% but needs little
else: measured on the reference PC, about a quarter of one CPU core and ~1.4 GB of RAM.

Each log line looks like this:

```
step    160 | loss 6.3414 | lr 1.26e-04 | gnorm  0.545 |    1,743 tok/s | 1.44 TFLOPS | eta 0:12:32
```

| Field | What it means | What's normal |
|---|---|---|
| `loss` | how wrong the model's next-word guesses are (lower is better) | starts at ~10.4, falls fast, then slowly; roughly 3–3.5 at the end of a full `small` run |
| `val loss` | the same on text it never trains on (printed at each evaluation) | close to `loss`. If it rises while `loss` falls, the model is memorising |
| `gnorm` | size of the update | usually below ~1–2; occasional spikes are fine |
| `tok/s` | training speed | ~1,750 on a GTX 1650 Super (with `grad_checkpointing: true`) |
| `eta` | time left for this run | |

Samples of generated text are printed every `sample_every` steps: watch them go from
random words to grammatical sentences. For graphs, run `tensorboard --logdir runs` in
a second terminal and open http://localhost:6006.

### Using the PC while it trains

- **Light use** (browser, music, video) is fine.
- **Games and other GPU-heavy programs: stop training first** (Ctrl-C), then resume
  afterwards. The training needs most of a 4 GB card. When a game takes GPU memory too,
  Windows moves part of the training's memory to system RAM ("shared GPU memory"), and
  both slow down: in the pilot run, training dropped from ~2,200 to ~480 tok/s while
  Rocket League was running, and stayed slow until training was restarted.
- **If `tok/s` drops to less than about half of normal** and nothing else is running,
  press Ctrl-C and resume. A fresh process gets its GPU memory back.

### Heat

On the reference PC the GPU ran at 62–73 °C at ~95 W during training (its target is
83 °C, it slows itself down at 90 °C). Check with `nvidia-smi` in another terminal. If
it ever stays above ~85 °C, stop training and check the case fans and dust. With an
administrator terminal you can also lower the power limit, which costs a little speed
for a lot less heat: `nvidia-smi -pl 80` (the GTX 1650 Super allows 70–100 W; the setting
resets on reboot).

### If something goes wrong

| Problem | What to do |
|---|---|
| `ModuleNotFoundError: No module named 'aupus'` | The virtual environment isn't active in this terminal: run `.venv\Scripts\activate` first (the prompt then starts with `(.venv)`), or call its Python directly: `.venv\Scripts\python.exe scripts\...` |
| `running scripts is disabled on this system` (when activating) | Windows blocks PowerShell scripts by default. Once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again |
| `model not found` / `No such file or directory` | Run the commands from the repository folder (`Aupus-AI`), not from `scripts\` |
| `CUDA out of memory` | Close GPU-heavy programs, or set `grad_checkpointing: true` in the config (or add `grad_checkpointing=true` to the `--resume` command) |
| `loss is nan` and the run stops | Resume from the last checkpoint; if it happens again at the same place, lower the learning rate: `--resume runs/aupus-0.1-base optim.lr=3e-4` |
| `already has checkpoints; pass --resume` | You used `--config` for an existing run: use `--resume` instead |
| `sequences per step ... keep the product the same` or `... pass eval_batches=N` | You changed `micro_batch` when resuming: change `grad_accum` and `eval_batches` with it, as the message says ("Moving a run to another machine") |
| `shards were tokenized with ... sha256` | The tokenizer file changed (often Windows line endings). Run `git checkout -- data/tokenizer/` |
| Very slow (`tok/s` far below normal) | See "Using the PC while it trains" |

Any setting can be changed when resuming by adding `key=value` (for example
`max_steps=12000` to train longer). Model shape settings can't change mid-run.

### How long it takes

Time = tokens ÷ tokens per second. At ~1,750 tok/s, one billion tokens is about 6.6 days
of training time (not counting stops). `configs/train/small.yaml` trains the `small` model
(125M parameters) for 2.5B tokens: 9,537 steps of ~150 s, so **~16–17 days of GPU time**,
3–4 weeks in practice with stops. The `eta` in the log is the remaining training time.
Smaller budgets finish sooner but give a weaker model (`max_steps=...` when starting);
the reasoning is in docs/PLAN.md.

## 3. Evaluate the Base model

```powershell
python scripts/download_eval_data.py      # once, ~7 MB
python scripts/evaluate.py --model runs/aupus-0.1-base/best.pt --val-data data/pretrain/fineweb_edu
```

Prints validation perplexity and zero-shot accuracy on HellaSwag, ARC-Easy and PIQA,
next to the score of random guessing, and saves them as JSON next to the checkpoint.
Training can be stopped (Ctrl-C) to evaluate an intermediate `best.pt`, then resumed.

## 4. Fine-tune into Aupus 0.1 Instruct

```powershell
python scripts/prepare_sft_data.py        # ~80 MB of human-written conversations -> data/sft/
python scripts/sft.py --config configs/sft/aupus-0.1-instruct.yaml
```

Before this, write `data/identity.jsonl`: a few dozen short conversations about what
Aupus is (guide: docs/IDENTITY.md; check it with `python scripts/check_identity.py`). It must be written by a person, like all
of Aupus's training data. Without it, the model won't know its own name.

Fine-tuning stops by itself when validation loss stops improving. Ctrl-C and
`python scripts/sft.py --resume runs/aupus-0.1-instruct` work exactly as in pretraining.

## 5. Export a release and chat

```powershell
python scripts/export_release.py --checkpoint runs/aupus-0.1-instruct/best.pt --out release/aupus-0.1-instruct
python scripts/chat_web.py --model release/aupus-0.1-instruct     # then open http://127.0.0.1:8000
python scripts/chat.py --model release/aupus-0.1-instruct         # the same chat in the terminal
python scripts/generate.py --model runs/aupus-0.1-base/best.pt --prompt "The water cycle is"
```

The browser chat runs only on your computer. Press Ctrl-C in its terminal to quit. Stop
training first: the chat needs the GPU too (or add `--device cpu`, which is slower but works
next to training). In the terminal chat, `/reset` clears the conversation and `/exit` quits.

**Teaching it with your own chats.** Chatting never changes the model. In the browser chat,
"Save chat" writes the conversation to `data/chats/` (never committed to git). Open the file,
rewrite Aupus's replies into the answers you wanted (training on its own unedited replies
would teach it its own mistakes), add the conversations to `data/identity.jsonl` or another
file passed with `--identity`, and run step 4 again. A release directory
holds the weights (`model.safetensors`), config, tokenizer and model card. To publish
it, zip it and attach it to a GitHub Release: weights never go into the git repository.

## Training longer or on more data later

- **Longer:** resume with a higher `max_steps`. The learning rate is a cosine curve that
  ends at `max_steps`, so a higher value stretches the curve: the rate jumps up a little
  and then decays again over the new remaining steps. Deciding early (mid-run) is
  better than after a run has finished, when the rate has already reached its minimum.
- **More data:** download more files (`--files 8`), rebuild the shards into a new
  directory, and start a new run on it.
- **A bigger model:** `configs/model/base.yaml` (348M parameters) needs a GPU with more
  memory, or many weeks on a 4 GB card.
