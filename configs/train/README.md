# Pretraining-run configs

One YAML file per run, loaded into `vulpes.train.TrainConfig` (strict: unknown keys are errors).
Every key and its default is documented in `vulpes/train/config.py`.

| File | Run | Where |
|---|---|---|
| `tiny.yaml` | Milestone 6 end-to-end check (`tiny`, 34M params, ~20M tokens of the M5 sample) | local GPU or CPU |
| `small.yaml` | Milestone 8, Vulpes 0.1 Base (`small`, 125M params, 2.5B tokens), sized for the GTX 1650 Super (docs/PLAN.md decisions 39–41) | desktop GPU |
| `small-runpod.yaml` | The same run (same name, data, order, tokens per step, steps, LR curve; a test checks it) with the memory and logging settings of a 24 GB GPU: 16 × 16, bf16, no checkpointing. Run it with `scripts/runpod/train.sh` (docs/RUNPOD.md) | rented RTX 4090 |

```bash
python scripts/train.py --config configs/train/tiny.yaml                 # new run -> runs/tiny/
python scripts/train.py --config configs/train/tiny.yaml max_steps=100 run_name=tiny-short
python scripts/train.py --resume runs/tiny                               # continue after Ctrl-C / crash
tensorboard --logdir runs
```

Each run saves its fully resolved config (`config.yaml`, `model.yaml`) next to its checkpoints.
Fine-tuning configs live in `configs/sft/` (loaded into `vulpes.train.SFTConfig`, run with `scripts/sft.py`).
