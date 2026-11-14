"""Detect the hardware and software environment Vulpes will train on.

Why this exists: every training decision (model size, context length, batch size,
token budget) depends on the machine. Rather than guessing, we measure it and
print a single report that can be pasted into docs/PLAN.md or a bug report.

The script only uses the standard library plus PyTorch *if installed*, so it can
run on a fresh machine before the project environment is set up.

Usage:
    python scripts/check_env.py            # human-readable report
    python scripts/check_env.py --json     # machine-readable report
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GIB = 1024**3


def _read_os_release() -> str:
    """Return the distro's pretty name (Linux), falling back to platform info."""
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip('"')
    except OSError:
        pass
    return f"{platform.system()} {platform.release()}"


def _cpu_info() -> dict[str, Any]:
    """CPU model, logical/physical cores and the SIMD extensions PyTorch's CPU kernels use."""
    model, flags = platform.processor() or "unknown", set()
    physical: set[tuple[str, str]] = set()
    phys_id = "0"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key == "model name":
                model = value
            elif key == "physical id":
                phys_id = value
            elif key == "core id":
                physical.add((phys_id, value))
            elif key == "flags" and not flags:
                flags = set(value.split())
    except OSError:
        pass
    simd = sorted(f for f in flags if f in {"avx", "avx2", "fma", "avx512f", "avx512_bf16", "amx_bf16"})
    return {
        "model": model,
        "logical_cores": os.cpu_count(),
        "physical_cores": len(physical) or None,
        "simd": simd,
    }


def _ram_gib() -> dict[str, float | None]:
    """Total and available system RAM (data prep and CPU training are RAM-bound)."""
    info: dict[str, float | None] = {"total_gib": None, "available_gib": None}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            kib = int(value.split()[0])
            if key == "MemTotal":
                info["total_gib"] = round(kib / 1024**2, 1)
            elif key == "MemAvailable":
                info["available_gib"] = round(kib / 1024**2, 1)
    except (OSError, ValueError):
        pass
    return info


def _disk_gib(path: Path) -> dict[str, float]:
    """Free space on the drive holding the project (data shards + checkpoints live here)."""
    usage = shutil.disk_usage(path)
    return {"total_gib": round(usage.total / GIB, 1), "free_gib": round(usage.free / GIB, 1)}


def _pci_gpus() -> list[str]:
    """GPUs visible on the PCI bus, independent of whether any ML runtime can use them."""
    if not shutil.which("lspci"):
        return []
    try:
        out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [
        line.split(": ", 1)[-1]
        for line in out.splitlines()
        if any(k in line for k in ("VGA compatible", "3D controller", "Display controller"))
    ]


def _amdgpu_memory() -> list[dict[str, Any]]:
    """Dedicated VRAM and GTT (system RAM the GPU may borrow) reported by the amdgpu driver.

    Integrated AMD GPUs have a small dedicated carve-out and borrow the rest from
    system RAM, which matters a lot for deciding whether they are worth using.
    """
    result = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        dev = card / "device"
        vram, gtt = dev / "mem_info_vram_total", dev / "mem_info_gtt_total"
        if vram.exists():
            try:
                result.append(
                    {
                        "card": card.name,
                        "vram_gib": round(int(vram.read_text()) / GIB, 2),
                        "gtt_gib": round(int(gtt.read_text()) / GIB, 2) if gtt.exists() else None,
                    }
                )
            except (OSError, ValueError):
                continue
    return result


def _torch_info() -> dict[str, Any]:
    """PyTorch build, accelerator backend, per-device VRAM and bf16 support."""
    try:
        import torch
    except ImportError:
        return {"installed": False}

    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "rocm_build": getattr(torch.version, "hip", None),
        "accelerator_available": torch.cuda.is_available(),
        "cpu_threads": torch.get_num_threads(),
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            info["devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_vram_gib": round(total / GIB, 2),
                    "free_vram_gib": round(free / GIB, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                }
            )
    else:
        # bf16 on CPU works everywhere but is only *fast* with AVX512-BF16 / AMX.
        info["cpu_bf16_fast"] = False
    return info


def collect() -> dict[str, Any]:
    """Gather the full environment report."""
    return {
        "os": _read_os_release(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpu": _cpu_info(),
        "ram": _ram_gib(),
        "disk_project_drive": _disk_gib(PROJECT_ROOT),
        "pci_gpus": _pci_gpus(),
        "amdgpu_memory": _amdgpu_memory(),
        "rocm_runtime_present": Path("/dev/kfd").exists(),
        "nvidia_smi_present": shutil.which("nvidia-smi") is not None,
        "torch": _torch_info(),
    }


def verdict(report: dict[str, Any]) -> list[str]:
    """Plain-language notes about what this machine can realistically train."""
    notes = []
    torch_info = report["torch"]
    if not torch_info.get("installed"):
        notes.append("PyTorch is not installed in this interpreter.")
    elif not torch_info.get("accelerator_available"):
        notes.append(
            "No CUDA/ROCm device usable by PyTorch: training would run on CPU only. "
            "Fine for tests and the debug preset; far too slow for a ~125M-parameter model."
        )
    else:
        best = max(d["total_vram_gib"] for d in torch_info["devices"])
        if best < 8:
            notes.append(f"Largest GPU has {best} GiB VRAM: expect the 'tiny' preset at most.")
        if not all(d["bf16_supported"] for d in torch_info["devices"]):
            notes.append("bf16 unsupported: training will use fp16 autocast + GradScaler.")
    for mem in report["amdgpu_memory"]:
        if mem["vram_gib"] < 2:
            notes.append(
                f"{mem['card']} is an integrated GPU ({mem['vram_gib']} GiB carve-out, shares system RAM)."
            )
    if (report["disk_project_drive"]["free_gib"] or 0) < 50:
        notes.append("Less than 50 GiB free on the project drive: data shards may not fit.")
    return notes


def _print_human(report: dict[str, Any]) -> None:
    t, cpu = report["torch"], report["cpu"]
    print("=== Vulpes AI environment check ===")
    print(f"OS            : {report['os']} (kernel {report['kernel']})")
    print(f"Python        : {report['python']}")
    print(
        f"CPU           : {cpu['model']} | {cpu['physical_cores']} cores / {cpu['logical_cores']} threads"
        f" | SIMD: {', '.join(cpu['simd']) or 'none'}"
    )
    ram = report["ram"]
    print(f"RAM           : {ram['total_gib']} GiB total, {ram['available_gib']} GiB available")
    d = report["disk_project_drive"]
    print(f"Disk (project): {d['free_gib']} GiB free of {d['total_gib']} GiB")
    print(f"PCI GPUs      : {'; '.join(report['pci_gpus']) or 'none detected'}")
    for m in report["amdgpu_memory"]:
        print(f"  amdgpu {m['card']}: dedicated VRAM {m['vram_gib']} GiB, GTT {m['gtt_gib']} GiB")
    print(f"ROCm /dev/kfd : {'present' if report['rocm_runtime_present'] else 'absent'}")
    print(f"nvidia-smi    : {'present' if report['nvidia_smi_present'] else 'absent'}")
    if t.get("installed"):
        backend = (
            f"CUDA {t['cuda_build']}"
            if t["cuda_build"]
            else f"ROCm {t['rocm_build']}"
            if t["rocm_build"]
            else "CPU-only build"
        )
        print(f"PyTorch       : {t['version']} ({backend}), {t['cpu_threads']} CPU threads")
        print(f"Accelerator   : {'available' if t['accelerator_available'] else 'NOT available'}")
        for dev in t["devices"]:
            print(
                f"  [{dev['index']}] {dev['name']}: {dev['free_vram_gib']}/{dev['total_vram_gib']} GiB free,"
                f" cc {dev['compute_capability']}, bf16={'yes' if dev['bf16_supported'] else 'no'}"
            )
    else:
        print("PyTorch       : not installed")
    notes = verdict(report)
    if notes:
        print("\nNotes:")
        for n in notes:
            print(f"  - {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args()
    report = collect()
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        _print_human(report)


if __name__ == "__main__":
    main()
