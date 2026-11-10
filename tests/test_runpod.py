"""Tests for training on RunPod: the 4090 config, scripts/runpod/train.sh and scripts/runpod/sync.py."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from aupus.config import load_yaml
from aupus.train import TrainConfig, resolve_model_config

ROOT = Path(__file__).resolve().parents[1]


def load_sync():
    spec = importlib.util.spec_from_file_location("runpod_sync", ROOT / "scripts" / "runpod" / "sync.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- the config


def test_runpod_config_is_the_same_run_as_small():
    """Only memory and logging settings may differ, or the run couldn't move between the pod and the PC."""
    pc, _ = resolve_model_config(TrainConfig.from_yaml(ROOT / "configs/train/small.yaml"))
    pod, _ = resolve_model_config(TrainConfig.from_yaml(ROOT / "configs/train/small-runpod.yaml"))
    for key in ("run_name", "model", "data_dir", "tokenizer", "seq_len", "max_steps", "optim", "seed"):
        assert getattr(pod, key) == getattr(pc, key), key
    assert pod.micro_batch * pod.grad_accum == pc.micro_batch * pc.grad_accum == 256
    assert pod.tokens_per_step == pc.tokens_per_step == 262_144
    # The PC resume command in docs/TRAINING.md passes eval_batches=800 for the same 800 val sequences.
    assert pod.micro_batch * pod.eval_batches == 1 * 800
    assert pod.precision == "auto" and pod.compile and not pod.grad_checkpointing
    assert pod.cuda_memory_limit_gib is None


# ---------------------------------------------------------------- sync.py


def test_parse_ssh_command():
    sync = load_sync()
    r = sync.parse_ssh_command("ssh root@203.0.113.7 -p 40123 -i ~/.ssh/id_ed25519")
    assert (r.target, r.port, r.key) == ("root@203.0.113.7", "40123", os.path.expanduser("~/.ssh/id_ed25519"))
    assert r.scp("/w/a.pt", Path("x"))[-2:] == ["root@203.0.113.7:/w/a.pt", "x"]
    assert "-P" in r.scp("/w/a.pt", Path("x")) and "-p" in r.ssh("ls")
    r = sync.parse_ssh_command("ssh -L 6006:localhost:6006 -o Foo=1 -l root 10.0.0.1")
    assert r.target == "root@10.0.0.1" and r.port is None and r.options == ["-o", "Foo=1"]
    with pytest.raises(ValueError, match="exposed TCP"):
        sync.parse_ssh_command("ssh abc123-64410f1b@ssh.runpod.io -i ~/.ssh/id_ed25519")
    with pytest.raises(ValueError, match="user@host"):
        sync.parse_ssh_command("ssh -p 22")


def test_files_to_copy_and_prune(tmp_path):
    sync = load_sync()
    remote = sync.parse_listing(
        "1500 1700000000.25 step_000100.pt\n1600 1700000600.9 step_000200.pt\n10 1700000601.0 config.yaml\n"
        "5 1700000602.0 best.pt.tmp\n3 1700000603.0 PULLED\n7 1700000604.0 best.pt\n"
    )
    assert remote["step_000200.pt"] == (1600, 1700000600)
    # Only the newest step checkpoint, never temp files or the PULLED marker.
    assert sync.files_to_copy(remote, tmp_path) == ["best.pt", "config.yaml", "step_000200.pt"]
    (tmp_path / "config.yaml").write_bytes(b"x" * 10)
    os.utime(tmp_path / "config.yaml", (1700000601, 1700000601))
    assert "config.yaml" not in sync.files_to_copy(remote, tmp_path)  # same size and time: unchanged
    for step in (100, 200, 300):
        (tmp_path / f"step_{step:06d}.pt").write_bytes(b"")
    (tmp_path / "best.pt").write_bytes(b"")
    assert [p.name for p in sync.prune_local(tmp_path, keep=2)] == ["step_000100.pt"]
    assert (tmp_path / "best.pt").exists() and (tmp_path / "step_000300.pt").exists()


class FakePod:
    """Plays the pod for sync.py: runs its ssh commands in a shell, and its scp copies as file copies."""

    def __init__(self, root: Path) -> None:
        self.root = root  # stands in for the pod's filesystem root

    def __call__(self, args: list[str], capture: bool = False) -> str:
        if args[0] == "scp":
            source, dest = args[-2].split(":", 1)[1], Path(args[-1])
            src = self.root / source.lstrip("/")
            if not src.exists():
                raise subprocess.CalledProcessError(1, args)
            if src.is_dir():
                shutil.copytree(src, dest / src.name, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dest)  # like scp -p: keeps the modification time
            return ""
        command = args[-1].replace("/workspace", str(self.root / "workspace"))
        out = subprocess.run(["bash", "-c", command], check=True, text=True, capture_output=True)
        return out.stdout


@pytest.mark.skipif(shutil.which("bash") is None or sys.platform == "win32", reason="needs bash and GNU find")
def test_sync_copies_the_run_and_confirms_the_end(tmp_path, monkeypatch):
    sync = load_sync()
    pod = FakePod(tmp_path / "pod")
    run = pod.root / "workspace/Aupus-AI/runs/aupus-0.1-base"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text("max_steps: 9537\n")
    (run / "metrics.jsonl").write_text('{"step": 100, "train/loss": 4.5}\n')
    for step in (100, 200):
        (run / f"step_{step:06d}.pt").write_bytes(os.urandom(1000))
    (run / "best.pt").write_bytes(os.urandom(1000))
    (run / "tb").mkdir()
    (run / "tb" / "events.out").write_bytes(b"e")
    monkeypatch.setattr(sync, "run", pod)
    remote = sync.parse_ssh_command("ssh root@pod -p 1")
    remote_dir, here = "/workspace/Aupus-AI/runs/aupus-0.1-base", tmp_path / "pc" / "aupus-0.1-base"

    assert not sync.sync_once(remote, remote_dir, here, keep=2)  # training: copies, doesn't confirm
    assert sorted(p.name for p in here.iterdir()) == [
        "best.pt",
        "config.yaml",
        "metrics.jsonl",
        "step_000200.pt",
        "tb",
    ]
    assert (here / "step_000200.pt").read_bytes() == (run / "step_000200.pt").read_bytes()
    assert sync.progress_line(here) == "step 100 / 9,537 (1.0%), train loss 4.500"
    assert not (run / "PULLED").exists()

    (run / "step_000300.pt").write_bytes(os.urandom(1000))
    (run / "step_000100.pt").unlink()  # the pod prunes too
    (run / "POD_DONE").write_text("finished: the run reached max_steps\n")
    assert sync.sync_once(remote, remote_dir, here, keep=2)
    assert sorted(p.name for p in here.glob("step_*.pt")) == ["step_000200.pt", "step_000300.pt"]
    assert "copied to the PC" in (run / "PULLED").read_text()

    # A copy that trained further here (the run continued on the PC) is never overwritten.
    (here / "step_000400.pt").write_bytes(b"")
    with pytest.raises(SystemExit, match="ahead of the pod"):
        sync.sync_once(remote, remote_dir, here, keep=2)


def test_sync_detects_a_corrupted_copy(tmp_path, monkeypatch):
    sync = load_sync()
    here = tmp_path / "pc"
    here.mkdir()
    (here / "best.pt").write_bytes(b"good")
    (here / "POD_DONE").write_text("finished\n")

    def fake_run(args, capture=False):
        if "find" in args[-1]:
            return "4 1 best.pt\n9 1 POD_DONE\n"  # the same sizes, but another time: copied again
        if args[0] == "scp" and "-r" in args:
            raise subprocess.CalledProcessError(1, args)  # no tb/ on this pod
        if args[0] == "scp":
            dest = Path(args[-1])
            dest.write_bytes(b"good" if dest.name.startswith("best") else b"finished\n")
            return ""
        if "sha256sum" in args[-1]:
            return f"{hashlib.sha256(b'other').hexdigest()}  best.pt\n"
        raise AssertionError(args)

    monkeypatch.setattr(sync, "run", fake_run)
    with pytest.raises(OSError, match="SHA-256 differs"):
        sync.sync_once(sync.parse_ssh_command("ssh root@pod"), "/run", here, keep=2)
    assert not (here / "best.pt").exists()  # deleted, so the next round copies it again


# ---------------------------------------------------------------- train.sh


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("bash") is None or sys.platform == "win32", reason="needs bash")
def test_train_sh_trains_resumes_and_marks_the_end(tiny_run, tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    # A stand-in for the pod's venv: put this interpreter first on PATH.
    python_dir = shlex.quote(str(Path(sys.executable).parent))
    (venv / "bin" / "activate").write_text(f"export PATH={python_dir}:$PATH\n")
    cfg = {
        "run_name": "pod",
        "out_dir": str(tmp_path / "runs"),
        "model": str(tiny_run.model_yaml),
        "data_dir": str(tiny_run.data_dir),
        "tokenizer": str(tiny_run.tokenizer),
        "seq_len": 64,
        "micro_batch": 2,
        "grad_accum": 2,
        "max_steps": 4,
        "optim": {"lr": 3e-3, "min_lr": 3e-4, "warmup_steps": 1},
        "device": "cpu",
        "compile": False,
        "eval_every": 2,
        "eval_batches": 2,
        "save_every": 2,
        "log_every": 1,
    }
    config = tmp_path / "pod.yaml"
    config.write_text(yaml.safe_dump(cfg))
    run_dir = tmp_path / "runs" / "pod"
    env = {
        **os.environ,
        "AUPUS_NO_TMUX": "1",
        "VENV": str(venv),
        "CONFIG": str(config),
        "RUN_DIR": str(run_dir),
        "DATA": str(tiny_run.data_dir),
        "EVAL": "0",
        "WAIT_FOR_PC_MIN": "0",
    }
    env.pop("RUNPOD_POD_ID", None)

    def train_sh(*overrides: str) -> subprocess.CompletedProcess:
        script = ROOT / "scripts" / "runpod" / "train.sh"
        return subprocess.run(
            ["bash", str(script), *overrides], env=env, text=True, capture_output=True, timeout=300
        )

    first = train_sh()
    assert first.returncode == 0, first.stdout + first.stderr
    assert "--config" in first.stdout and "finished" in (run_dir / "POD_DONE").read_text()
    assert "stop it yourself" in first.stdout  # not on a pod: nothing to stop
    assert (run_dir / "step_000004.pt").exists() and (tmp_path / "runs" / "pod-console.log").exists()

    (run_dir / "PULLED").write_text("old")
    second = train_sh("max_steps=6")  # the same command resumes, with overrides
    assert second.returncode == 0, second.stdout + second.stderr
    assert "--resume" in second.stdout and "resumed from" in second.stdout
    assert (run_dir / "step_000006.pt").exists()
    assert not (run_dir / "PULLED").exists()  # markers of the previous session are cleared
    assert load_yaml(run_dir / "config.yaml")["max_steps"] == 6
