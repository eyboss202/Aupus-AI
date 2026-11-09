"""The pretraining loop.

One optimizer step::

    lr = schedule(step)
    repeat grad_accum times:                 # micro-batches
        x, y = next(loader)
        loss = model(x, y).loss / grad_accum  # under autocast (bf16/fp16) or plain fp32
        backward(loss)                       # gradients add up across micro-batches
    clip global grad norm to grad_clip
    AdamW step, zero grads

Dividing each micro-batch loss by ``grad_accum`` makes the summed gradient
equal to the gradient of the mean loss over the whole effective batch, so
``micro_batch=4, grad_accum=8`` trains exactly like ``micro_batch=32`` (up to
float rounding) using a quarter of the activation memory.

Around that the trainer:
  * evaluates validation loss on a fixed slice of the val split (every
    ``eval_every`` steps, and before the first step);
  * generates text from fixed prompts with a fixed seed (``sample_every``), so
    differences between samples come from the model, not the dice;
  * logs loss, LR, grad norm, tokens/s and achieved TFLOPS to the console,
    ``train.log``, ``metrics.jsonl`` and TensorBoard (``<run_dir>/tb``);
  * saves a full checkpoint every ``save_every`` steps and at the end, keeps
    the newest ``keep_last`` plus ``best.pt``;
  * on Ctrl-C, finishes the current step, saves, and exits cleanly (a second
    Ctrl-C aborts immediately). ``scripts/train.py --resume`` continues the run.
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from aupus.config import save_config, to_dict
from aupus.data.loader import PretrainLoader, eval_batches
from aupus.data.shards import check_tokenizer, file_sha256, open_split
from aupus.model import ModelConfig, Transformer, flops_per_token, param_count, stream
from aupus.tokenizer import EOS, Tokenizer
from aupus.train import checkpoint
from aupus.train.config import TrainConfig, resolve_model_config
from aupus.train.optim import lr_at, make_optimizer, set_lr

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


# ---------------------------------------------------------------- environment


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested, but torch.cuda.is_available() is False")
    return torch.device(name)


def resolve_precision(name: str, device: torch.device) -> str:
    """``auto`` -> bf16 on Ampere+ GPUs, fp32 otherwise (CPU, and Turing/Pascal GPUs).

    bf16 is only fast with bf16 tensor cores (compute capability 8.0+). On older
    GPUs and on most CPUs it is emulated and several times slower than fp32
    (docs/PLAN.md §2 and §13). fp16 is never picked automatically: it needs a
    GradScaler and is only fast on GPUs with fp16 tensor cores (e.g. T4).
    """
    if name != "auto":
        if name != "fp32" and device.type == "cpu":
            raise RuntimeError(f"precision={name} on CPU is not supported (it is far slower than fp32)")
        return name
    if device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 0):
        return "bf16"
    return "fp32"


def compile_unsupported_reason(device: torch.device) -> str | None:
    """Why ``torch.compile`` can't work here, or None if it should.

    Inductor generates Triton kernels on CUDA (no official Windows build) and
    C++ on CPU (needs a compiler on PATH: MSVC ``cl`` on Windows, gcc/clang elsewhere).
    """
    if device.type == "cuda":
        try:
            import triton  # noqa: F401
        except ImportError:
            return "Triton is not installed (no official Windows build)"
        return None
    compilers = ("cl",) if sys.platform == "win32" else ("c++", "g++", "clang++")
    if not any(shutil.which(c) for c in compilers):
        return f"no C++ compiler on PATH (looked for {', '.join(compilers)})"
    return None


# ---------------------------------------------------------------- trainer


class Trainer:
    """Builds model, optimizer and data from a ``TrainConfig`` and runs (or resumes) the loop.

    Args:
        cfg: the run config.
        resume: continue from the newest checkpoint in ``cfg.run_dir``. Without it,
            a run directory that already has checkpoints is an error (no silent overwrites).
        log: where human-readable progress lines go (also appended to ``train.log``).

    Supervised fine-tuning (``aupus/train/sft.py``) reuses this loop. It overrides
    the few methods that depend on the data: ``_resolve_configs``, ``_setup_data``,
    ``_init_weights``, ``_make_loader``, ``evaluate``, ``_sample_prompt``,
    ``_format_sample``, ``_data_metrics`` and ``_describe_data``. Setting
    ``_halt_reason`` ends the run cleanly after the current step (early stopping).
    """

    #: Recorded in checkpoints, so a model's history is visible (``pretrain`` or ``sft``).
    stage = "pretrain"

    def __init__(self, cfg: TrainConfig, resume: bool = False, log: Callable[[str], None] = print) -> None:
        self._print = log
        self.cfg, self.model_cfg = self._resolve_configs(cfg)
        cfg = self.cfg
        self.run_dir = cfg.run_dir
        self._stop_requested = False
        self._halt_reason: str | None = None  # set to stop training early (SFT early stopping)

        existing = checkpoint.latest_checkpoint(self.run_dir)
        if resume and existing is None:
            raise FileNotFoundError(f"--resume: no checkpoint in {self.run_dir}")
        if not resume and existing is not None:
            raise FileExistsError(
                f"{self.run_dir} already has checkpoints; pass --resume to continue it, "
                "or choose another run_name"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.device = resolve_device(cfg.device)
        self.precision = resolve_precision(cfg.precision, self.device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True  # TF32 matmuls on Ampere+; no effect on older GPUs
            torch.backends.cudnn.allow_tf32 = True
            if cfg.cuda_memory_limit_gib is not None:
                index = self.device.index if self.device.index is not None else torch.cuda.current_device()
                total = torch.cuda.get_device_properties(index).total_memory
                fraction = min(1.0, cfg.cuda_memory_limit_gib * 1024**3 / total)
                torch.cuda.set_per_process_memory_fraction(fraction, index)

        self._setup_data()
        if self.tokenizer.vocab_size > self.model_cfg.vocab_size:
            raise ValueError(
                f"tokenizer has {self.tokenizer.vocab_size} ids, "
                f"but the model only has {self.model_cfg.vocab_size}"
            )
        self.eos_id = self.tokenizer.special_id(EOS)
        self.tokenizer_sha256 = file_sha256(cfg.tokenizer)

        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.model = Transformer(self.model_cfg).to(self.device)
        if not resume:
            self._init_weights()
        self.model.gradient_checkpointing = cfg.grad_checkpointing
        self.optimizer = make_optimizer(self.model, cfg.optim, self.device)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.precision == "fp16")
        self.loader = self._make_loader()
        self.step = 0
        self.best_val_loss = math.inf
        self.evals_since_best = 0  # consecutive evals without a new best (early stopping)

        if resume:
            self._load(existing)
        save_config(cfg, self.run_dir / "config.yaml")
        save_config(self.model_cfg, self.run_dir / "model.yaml")
        self.run = self._maybe_compile()
        self._tb = None

    # ------------------------------------------------------------ data hooks (SFT overrides these)

    def _resolve_configs(self, cfg: TrainConfig) -> tuple[TrainConfig, ModelConfig]:
        return resolve_model_config(cfg)

    def _setup_data(self) -> None:
        """Load the tokenizer; check the shards were made with it."""
        meta, _ = open_split(self.cfg.data_dir, "train")
        check_tokenizer(meta, self.cfg.tokenizer)
        self.tokenizer = Tokenizer.load(self.cfg.tokenizer)

    def _init_weights(self) -> None:
        """Pretraining starts from the random init in ``Transformer.reset_parameters``."""

    def _make_loader(self) -> PretrainLoader:
        cfg = self.cfg
        return PretrainLoader(cfg.data_dir, "train", cfg.micro_batch, cfg.seq_len, cfg.seed)

    def _sample_prompt(self, prompt: str) -> tuple[list[int], list[int]]:
        """Prompt ids and stop ids for one text sample."""
        # A new document starts right after <|eos|> in the training data, so that's the "start" token.
        return [self.eos_id, *self.tokenizer.encode(prompt)], [self.eos_id]

    def _format_sample(self, prompt: str, generated: str) -> str:
        return prompt + generated

    def _data_metrics(self) -> dict[str, float]:
        """Extra ``data/*`` values for each training log record."""
        return {"data/tokens_seen": self.step * self.cfg.tokens_per_step}

    def _describe_data(self) -> str:
        cfg = self.cfg
        total = cfg.tokens_per_step * cfg.max_steps
        return (
            f"batch {cfg.micro_batch} x {cfg.grad_accum} accum x {cfg.seq_len} = {cfg.tokens_per_step:,} "
            f"tokens/step | {cfg.max_steps:,} steps = {total / 1e6:.1f}M tokens "
            f"({total / (self.loader.num_windows * cfg.seq_len):.2f} epochs "
            f"of {self.loader.num_windows:,} windows)"
        )

    # ------------------------------------------------------------ setup helpers

    def log(self, msg: str) -> None:
        self._print(msg)
        with open(self.run_dir / "train.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _autocast(self) -> contextlib.AbstractContextManager:
        if self.precision == "fp32":
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=DTYPES[self.precision])

    def _maybe_compile(self) -> Callable[..., Any]:
        """``torch.compile`` the model if possible; probe it once so failures fall back to eager.

        The probe runs one forward + backward on a dummy batch before training
        starts, so a compiler error can't happen halfway through a real step
        (which would leave half-accumulated gradients and a loader that has
        already moved on).
        """
        self.compiled = False
        if not self.cfg.compile:
            return self.model
        reason = compile_unsupported_reason(self.device)
        if reason is not None:
            self.log(f"torch.compile unavailable: {reason}; running eager")
            return self.model
        compiled = torch.compile(self.model)
        x = torch.zeros(self.cfg.micro_batch, self.cfg.seq_len, dtype=torch.long, device=self.device)
        try:
            t0 = time.perf_counter()
            with self._autocast():
                loss = compiled(x, x).loss
            loss.backward()
            self.optimizer.zero_grad(set_to_none=True)
            self.log(f"torch.compile ready ({time.perf_counter() - t0:.0f} s)")
        except Exception as e:  # any compiler/toolchain failure: eager still trains correctly
            self.optimizer.zero_grad(set_to_none=True)
            self.log(
                f"torch.compile failed ({type(e).__name__}: {str(e).splitlines()[0][:200]}); running eager"
            )
            return self.model
        self.compiled = True
        return compiled

    # ------------------------------------------------------------ checkpoints

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "loader": self.loader.state_dict(),
            "rng": checkpoint.rng_state(),
            "best_val_loss": self.best_val_loss,
            "evals_since_best": self.evals_since_best,
            "train_config": to_dict(self.cfg),
            "model_config": to_dict(self.model_cfg),
            "stage": self.stage,
            # Lets later steps (SFT, eval, release export) check they use the same tokenizer.
            "tokenizer_sha256": self.tokenizer_sha256,
        }

    def save(self) -> Path:
        path = checkpoint.step_path(self.run_dir, self.step)
        checkpoint.save(self.state_dict(), path)
        checkpoint.prune(self.run_dir, self.cfg.keep_last)
        return path

    def _load(self, path: Path) -> None:
        state = checkpoint.load(path, map_location=self.device)
        saved_model = ModelConfig(**state["model_config"])
        if saved_model != self.model_cfg:
            raise ValueError(
                f"model config changed since {path}:\n  saved:   {saved_model}\n  current: {self.model_cfg}"
            )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.loader.load_state_dict(state["loader"])
        checkpoint.set_rng_state(state["rng"])
        self.step = state["step"]
        self.best_val_loss = state["best_val_loss"]
        self.evals_since_best = state.get("evals_since_best", 0)  # absent before Milestone 10
        # Steps after this checkpoint will be trained again: drop their old metrics
        # (TensorBoard does the same via purge_step).
        metrics = self.run_dir / "metrics.jsonl"
        if metrics.exists():
            lines = metrics.read_text(encoding="utf-8").splitlines()
            kept = [line for line in lines if json.loads(line)["step"] <= self.step]
            metrics.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
        self.log(f"resumed from {path} at step {self.step}")

    # ------------------------------------------------------------ eval + samples

    @torch.no_grad()
    def evaluate(self) -> float:
        """Mean loss over the first ``eval_batches`` micro-batches of the val split."""
        self.model.eval()
        total, count = 0.0, 0
        batches = eval_batches(
            self.cfg.data_dir, "val", self.cfg.micro_batch, self.cfg.seq_len, self.cfg.eval_batches
        )
        for x, y in batches:
            with self._autocast():
                loss = self.model(x.to(self.device), y.to(self.device)).loss
            total += loss.item() * len(x)
            count += len(x)
        self.model.train()
        return total / count

    def samples(self) -> list[str]:
        """One continuation per prompt; same seed every time, so changes reflect the model only."""
        out = []
        gen = torch.Generator(device=self.device)
        for i, prompt in enumerate(self.cfg.sample_prompts):
            gen.manual_seed(self.cfg.seed + i)
            ids, stop_ids = self._sample_prompt(prompt)
            new = list(
                stream(
                    self.model,
                    ids,
                    self.cfg.sample_tokens,
                    temperature=0.8,
                    top_k=50,
                    stop_ids=stop_ids,
                    generator=gen,
                    vocab_limit=self.tokenizer.vocab_size,
                )
            )
            out.append(self._format_sample(prompt, self.tokenizer.decode(new, skip_special=True)))
        return out

    # ------------------------------------------------------------ logging

    def _metrics(self, record: dict[str, Any]) -> None:
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if self._tb is not None:
            for key, value in record.items():
                if key != "step" and isinstance(value, int | float) and math.isfinite(value):
                    self._tb.add_scalar(key, value, record["step"])

    def _open_tensorboard(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:  # tensorboard not installed: metrics.jsonl still has everything
            self.log("tensorboard not installed; logging to metrics.jsonl only")
            return
        self._tb = SummaryWriter(self.run_dir / "tb", purge_step=self.step or None)

    def _do_eval(self) -> None:
        t0 = time.perf_counter()
        val = self.evaluate()
        improved = val < self.best_val_loss
        self.evals_since_best = 0 if improved else self.evals_since_best + 1
        if improved:
            self.best_val_loss = val
            checkpoint.save(self.state_dict(), self.run_dir / checkpoint.BEST)
        self._metrics({"step": self.step, "val/loss": val, "val/ppl": math.exp(min(val, 50))})
        mark = " (best)" if improved else ""
        self.log(f"step {self.step:6d} | val loss {val:.4f} | ppl {math.exp(min(val, 50)):9.2f}{mark}")
        self.log(f"             eval took {time.perf_counter() - t0:.1f} s")

    def _do_samples(self) -> None:
        for i, text in enumerate(self.samples()):
            self.log(f"--- sample {i} @ step {self.step}\n{text}\n---")
            if self._tb is not None:
                self._tb.add_text(f"sample/{i}", text, self.step)

    # ------------------------------------------------------------ main loop

    def _on_sigint(self, signum: int, frame: Any) -> None:
        if self._stop_requested:
            raise KeyboardInterrupt
        self._stop_requested = True
        self._print("\nCtrl-C: stopping after this step and saving a checkpoint (Ctrl-C again to abort)")

    def train(self) -> dict[str, Any]:
        """Run until ``max_steps`` (or Ctrl-C). Returns a summary of the run."""
        old_handler = None
        if threading.current_thread() is threading.main_thread():
            old_handler = signal.signal(signal.SIGINT, self._on_sigint)
        self._open_tensorboard()
        try:
            return self._train_loop()
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGINT, old_handler)
            if self._tb is not None:
                self._tb.close()
                self._tb = None

    def _describe(self) -> None:
        cfg, mc = self.cfg, self.model_cfg
        counts = param_count(mc)
        device = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"
        self.log(
            f"run {cfg.run_name} | model {cfg.model}: {counts['total'] / 1e6:.1f}M params "
            f"({counts['non_embedding'] / 1e6:.1f}M non-embedding), context {mc.context_length}"
        )
        limit = cfg.cuda_memory_limit_gib
        memory = ""
        if limit is not None and self.device.type == "cuda":
            memory = f" | GPU memory limit {limit:g} GiB"
        self.log(
            f"device {device} | precision {self.precision} | compiled {self.compiled} | "
            f"grad checkpointing {cfg.grad_checkpointing}{memory}"
        )
        self.log(self._describe_data())

    def _train_loop(self) -> dict[str, Any]:
        cfg = self.cfg
        self._describe()
        self.model.train()
        flops_per_tok = flops_per_token(self.model_cfg, cfg.seq_len)["train"]
        if self.step == 0:
            if cfg.eval_every:
                self._do_eval()
            if cfg.sample_every and cfg.sample_prompts:
                self._do_samples()

        start_step, t_run = self.step, time.perf_counter()
        window_loss = torch.zeros((), device=self.device)
        window_steps, window_tokens, window_time = 0, 0, 0.0
        last_loss = math.nan
        while self.step < cfg.max_steps:
            t0 = time.perf_counter()
            lr = lr_at(self.step, cfg.optim, cfg.max_steps)
            set_lr(self.optimizer, lr)
            for _ in range(cfg.grad_accum):
                x, y = next(self.loader)
                window_tokens += x.numel()  # SFT batches vary in length; pretraining's never do
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with self._autocast():
                    loss = self.run(x, y).loss / cfg.grad_accum
                self.scaler.scale(loss).backward()
                window_loss += loss.detach()
            self.scaler.unscale_(self.optimizer)
            max_norm = cfg.optim.grad_clip if cfg.optim.grad_clip > 0 else math.inf
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            window_steps += 1

            last = self.step == cfg.max_steps
            if self.step % cfg.log_every == 0 or last or self._stop_requested:
                last_loss = window_loss.item() / window_steps  # .item() syncs the GPU, so timing is honest
                window_time += time.perf_counter() - t0
                if not math.isfinite(last_loss):
                    raise FloatingPointError(f"loss is {last_loss} at step {self.step}; stopping")
                tok_s = window_tokens / window_time
                record = {
                    "step": self.step,
                    "train/loss": last_loss,
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "perf/tokens_per_sec": tok_s,
                    "perf/tflops": tok_s * flops_per_tok / 1e12,
                    "data/epoch": self.loader.epoch,
                    **self._data_metrics(),
                }
                if self.device.type == "cuda":
                    record["perf/peak_mem_gib"] = torch.cuda.max_memory_allocated(self.device) / 1024**3
                self._metrics(record)
                eta = (cfg.max_steps - self.step) * window_time / window_steps
                self.log(
                    f"step {self.step:6d} | loss {last_loss:.4f} | lr {lr:.2e} | "
                    f"gnorm {float(grad_norm):6.3f} | "
                    f"{tok_s:8,.0f} tok/s | {record['perf/tflops']:.2f} TFLOPS | eta {_hms(eta)}"
                )
                window_loss.zero_()
                window_steps, window_tokens, window_time = 0, 0, 0.0
            else:
                window_time += time.perf_counter() - t0

            if cfg.eval_every and (self.step % cfg.eval_every == 0 or last):
                self._do_eval()
            halt = self._stop_requested or self._halt_reason is not None
            final = last or self._halt_reason is not None  # the run is over (not just paused by Ctrl-C)
            if cfg.sample_every and cfg.sample_prompts and (self.step % cfg.sample_every == 0 or final):
                self._do_samples()
            if (cfg.save_every and self.step % cfg.save_every == 0) or last or halt:
                path = self.save()
                self.log(f"saved {path}")
            if self._halt_reason is not None:
                self.log(f"stopping at step {self.step}: {self._halt_reason}")
            if halt:
                break

        elapsed = time.perf_counter() - t_run
        return {
            "step": self.step,
            "finished": self.step >= cfg.max_steps or self._halt_reason is not None,
            "early_stopped": self._halt_reason is not None,
            "train_loss": last_loss,
            "best_val_loss": self.best_val_loss,
            "steps_this_session": self.step - start_step,
            "seconds_this_session": elapsed,
        }


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
