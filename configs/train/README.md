# Training-run configs

One YAML file per run, loaded into `aupus.train.TrainConfig` (strict: unknown keys are errors).
Every key and its default is documented in `aupus/train/config.py`.

| File | Run | Where |
|---|---|---|
| `tiny.yaml` | Milestone 6 end-to-end check (`tiny`, 34M params, ~20M tokens of the M5 sample) | local GPU or CPU |
| `small.yaml` | Milestone 8, Aupus 0.1 Base (`small`, 125M params, 2.5B tokens). Draft until the rented GPU is benchmarked | rented GPU |

```bash
python scripts/train.py --config configs/train/tiny.yaml                 # new run -> runs/tiny/
python scripts/train.py --config configs/train/tiny.yaml max_steps=100 run_name=tiny-short
python scripts/train.py --resume runs/tiny                               # continue after Ctrl-C / crash
tensorboard --logdir runs
```

Each run saves its fully resolved config (`config.yaml`, `model.yaml`) next to its checkpoints.
SFT configs are added in Milestone 10.
