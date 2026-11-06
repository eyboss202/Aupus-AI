"""Measure real training throughput (tokens/sec) for each model preset, on CPU or CUDA.

Why this exists: the token budget for Aupus 0.1 is fixed by Chinchilla-style
scaling (~20 tokens per parameter), so the only unknown in "how long will
training take?" is how many tokens per second this machine can push through a
full training step (forward + backward + grad clipping + AdamW update). We
measure that instead of guessing.

It benchmarks the real ``aupus.model.Transformer`` on random tokens.

Precision (``--dtype``):
  * ``fp32``: plain float32;
  * ``bf16``: bf16 autocast (Ampere+ GPUs; on CPU only fast with AVX-512 BF16);
  * ``fp16``: fp16 autocast + GradScaler, for GPUs with fast fp16 but no bf16
    (e.g. T4). The scaler multiplies the loss so small fp16 gradients don't round
    to 0. Measure first: on GPUs without tensor cores (GTX 16xx) fp16 matmuls
    are ~5x *slower* than fp32 (docs/PLAN.md section 13).

Usage:
    python scripts/benchmark_throughput.py --presets tiny small --micro-batch 1
    python scripts/benchmark_throughput.py --device cuda --presets small --micro-batch 2 --grad-checkpointing
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from dataclasses import asdict

import torch

from aupus.model import ModelConfig, Transformer, flops_per_token, param_count
from aupus.model.config import PRESET_DIR

PRESETS = sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def peak_cpu_rss_gib() -> float | None:
    """Peak resident memory of this process, on Linux/macOS and Windows."""
    try:
        import resource  # Unix only

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1024**3 if sys.platform == "darwin" else rss / 1024**2  # bytes on macOS, KiB on Linux
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # Explicit types: without them ctypes passes the process handle (-1) as a
        # 32-bit int, which is an invalid handle on 64-bit Windows.
        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        c = Counters()
        c.cb = ctypes.sizeof(Counters)
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return c.PeakWorkingSetSize / 1024**3
    return None


def bench(
    name: str,
    cfg: ModelConfig,
    device: str,
    context: int,
    micro_batch: int,
    steps: int,
    warmup: int,
    dtype: str,
    compile_: bool,
    grad_checkpointing: bool,
) -> dict:
    torch.manual_seed(0)
    cfg = ModelConfig(**{**asdict(cfg), "context_length": max(context, cfg.context_length)})
    model = Transformer(cfg).to(device).train()
    model.gradient_checkpointing = grad_checkpointing
    run = torch.compile(model) if compile_ else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=device == "cuda")
    scaler = torch.amp.GradScaler(device, enabled=dtype == "fp16")
    x = torch.randint(0, cfg.vocab_size, (micro_batch, context), device=device)
    y = torch.randint(0, cfg.vocab_size, (micro_batch, context), device=device)
    autocast = torch.autocast(device, dtype=DTYPES[dtype]) if dtype != "fp32" else contextlib.nullcontext()

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    def step() -> float:
        with autocast:
            loss = run(x, y).loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        return loss.item()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(warmup):
        step()
    sync()
    warm_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = step()
    sync()
    elapsed = time.perf_counter() - t0
    tok_s = steps * micro_batch * context / elapsed
    flops = flops_per_token(cfg, context)["train"]
    counts = param_count(cfg)
    result = {
        "preset": name,
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "params_total_M": round(counts["total"] / 1e6, 1),
        "context": context,
        "micro_batch": micro_batch,
        "dtype": dtype,
        "compiled": compile_,
        "grad_checkpointing": grad_checkpointing,
        "tokens_per_sec": round(tok_s, 1),
        # Model FLOPs only: recomputation from checkpointing isn't counted (standard MFU convention).
        "achieved_tflops": round(tok_s * flops / 1e12, 3),
        "sec_per_step": round(elapsed / steps, 3),
        "warmup_sec": round(warm_s, 1),
        "final_loss": round(loss, 3),
        "sanity_initial_loss_expected": round(math.log(cfg.vocab_size), 3),
    }
    if device == "cuda":
        result["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    rss = peak_cpu_rss_gib()
    result["peak_rss_gib"] = round(rss, 2) if rss is not None else None
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--presets", nargs="+", default=PRESETS, choices=PRESETS)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    ap.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    ap.add_argument("--grad-checkpointing", action="store_true", help="recompute activations in backward")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    print(f"# torch {torch.__version__} | device={device} | threads={torch.get_num_threads()}")
    for name in args.presets:
        try:
            r = bench(
                name,
                ModelConfig.preset(name),
                device,
                args.context,
                args.micro_batch,
                args.steps,
                args.warmup,
                args.dtype,
                args.compile,
                args.grad_checkpointing,
            )
        except torch.cuda.OutOfMemoryError:
            r = {"preset": name, "micro_batch": args.micro_batch, "error": "out of memory"}
            torch.cuda.empty_cache()
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
