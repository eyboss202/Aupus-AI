"""Data parallelism over several GPUs (DDP) for the pretraining loop (Vulpes Flos 1.0, K2).

Kaggle gives two T4s. Each holds a full copy of the model; each step's sequences are
split between them, and their gradients are averaged before the optimizer step. That
is ``torch.nn.parallel.DistributedDataParallel`` (DDP), started by ``torchrun``::

    torchrun --standalone --nproc_per_node 2 scripts/train.py --config configs/train/flos-kaggle.yaml

``torchrun`` starts one process per GPU and tells each its place through environment
variables (``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE``). Without them (plain ``python
scripts/train.py``) everything here is a no-op and training is single-process, as before.

**The same run on any number of GPUs.** ``micro_batch x grad_accum`` stays the number
of sequences in one optimizer step, whatever the number of processes. With W
processes, process r does micro-batches r, r + W, r + 2W, ... of the step and skips the
others' windows in the shared, seeded order (``PretrainLoader.skip``). Every step
therefore trains on exactly the sequences a single GPU would, and a run can move
between 2 T4s, one GPU and the desktop with nothing but a ``micro_batch``/``grad_accum``
change (results equal up to float rounding, as with any change of batch layout).

**What each process does.** All of them: forward and backward on their micro-batches
(gradients are summed across processes only on the last one, via ``no_sync``), the
optimizer step (identical everywhere, because the averaged gradients are), and their
share of the validation batches. Only process 0 (``is_main``) writes: logs, metrics,
TensorBoard, samples and checkpoints. The few decisions that must be the same
everywhere (stop now? did compile work?) are agreed with a tiny all-reduce, so one
process can never leave the others waiting forever.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

#: How long a process waits for the others at a collective before failing. Process 0
#: saves 4.2 GB checkpoints and writes samples while the others wait at the next
#: gradient sum, so this is far above the 10-minute default.
TIMEOUT = datetime.timedelta(minutes=30)


@dataclass(frozen=True)
class DistInfo:
    """This process's place in the job. ``world_size == 1`` means plain single-process training."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def from_env() -> DistInfo:
    """Read the variables ``torchrun`` sets; defaults to a single process."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return DistInfo()
    return DistInfo(int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", "0")), world)


def setup(device: torch.device) -> DistInfo:
    """Join the process group when started by ``torchrun`` (NCCL on CUDA, Gloo on CPU)."""
    info = from_env()
    if not info.enabled:
        return info
    if not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend, timeout=TIMEOUT)
    if dist.get_world_size() != info.world_size or dist.get_rank() != info.rank:
        raise RuntimeError("WORLD_SIZE/RANK don't match the initialised process group")
    return info


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def local_device(device: torch.device, info: DistInfo) -> torch.device:
    """Under DDP each process owns GPU ``LOCAL_RANK``; on CPU they share the CPU."""
    if not info.enabled or device.type != "cuda":
        return device
    if torch.cuda.device_count() <= info.local_rank:
        visible = torch.cuda.device_count()
        raise RuntimeError(f"process {info.rank} wants GPU {info.local_rank}, but only {visible} are visible")
    torch.cuda.set_device(info.local_rank)
    return torch.device("cuda", info.local_rank)


def sum_(values: list[float], device: torch.device, info: DistInfo) -> list[float]:
    """Element-wise sum of ``values`` over all processes (just ``values`` without DDP)."""
    if not info.enabled:
        return values
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def any_(flag: bool, device: torch.device, info: DistInfo) -> bool:
    """True on every process if it is True on any one (e.g. one of them was told to stop)."""
    return sum_([float(flag)], device, info)[0] > 0


def all_(flag: bool, device: torch.device, info: DistInfo) -> bool:
    """True on every process only if it is True on all of them."""
    return sum_([float(flag)], device, info)[0] == info.world_size
