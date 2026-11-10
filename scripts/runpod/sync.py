"""Keep a copy of a RunPod training run on this PC while it trains, and confirm the final copy.

Run it on the PC, from the Aupus-AI folder, with the "SSH over exposed TCP" command from the
pod's Connect panel in quotes (docs/RUNPOD.md)::

    python scripts/runpod/sync.py "ssh root@203.0.113.7 -p 40123 -i ~/.ssh/id_ed25519"
    python scripts/runpod/sync.py "ssh root@203.0.113.7 -p 40123 -i ~/.ssh/id_ed25519" --every 0   # once

Every ``--every`` minutes it copies ``runs/aupus-0.1-base/`` from the pod into the same folder
here: the newest ``step_NNNNNN.pt`` (1.5 GB), ``best.pt`` when it changed, and the small files
(config, logs, metrics, TensorBoard, evaluation results). It keeps the newest ``--keep`` step
checkpoints here, like the pod does. If the pod is lost, this copy is where the run continues:
on a new pod, or on this PC (docs/TRAINING.md, "Moving a run to another machine").

When ``scripts/runpod/train.sh`` has ended (it writes ``POD_DONE``), this makes a last copy,
checks it with SHA-256 against the pod, and writes ``PULLED`` on the pod. train.sh then stops
the pod. Only after that "checked" line is it safe to terminate the pod.

Only the standard library and the ``ssh``/``scp`` commands are needed (built into Windows 10+,
macOS and Linux), so this runs without the project's virtual environment too.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REMOTE_REPO = "/workspace/Aupus-AI"
RUN = "aupus-0.1-base"
DONE, PULLED = "POD_DONE", "PULLED"
STEP = re.compile(r"^step_(\d{6,})\.pt$")
SKIP = {PULLED}
SSH_OPTIONS = [
    "-o", "BatchMode=yes",  # never wait at a password prompt
    "-o", "StrictHostKeyChecking=accept-new",  # a new pod has a new host key
    "-o", "ConnectTimeout=20",
    "-o", "ServerAliveInterval=30",
]  # fmt: skip


@dataclass(frozen=True)
class Remote:
    """Where the pod is, parsed from the ``ssh ...`` command RunPod shows."""

    target: str  # user@host
    port: str | None = None
    key: str | None = None
    options: list[str] = field(default_factory=list)  # extra -o settings from the command

    def ssh(self, command: str) -> list[str]:
        args = ["ssh", *SSH_OPTIONS, *self.options]
        if self.port:
            args += ["-p", self.port]
        if self.key:
            args += ["-i", self.key]
        return [*args, self.target, command]

    def scp(self, remote_path: str, local: Path, recursive: bool = False) -> list[str]:
        args = ["scp", "-p", *(["-r"] if recursive else []), *SSH_OPTIONS, *self.options]  # -p keeps mtimes
        if self.port:
            args += ["-P", self.port]
        if self.key:
            args += ["-i", self.key]
        return [*args, f"{self.target}:{remote_path}", str(local)]


def parse_ssh_command(text: str) -> Remote:
    """``ssh root@1.2.3.4 -p 40123 -i ~/.ssh/id_ed25519`` -> Remote (other flags are ignored)."""
    if os.name == "nt":
        text = text.replace("\\", "/")  # shlex treats backslashes as escapes; Windows accepts "/"
    tokens = shlex.split(text)
    if tokens and tokens[0] in ("ssh", "ssh.exe"):
        tokens = tokens[1:]
    target = port = key = user = None
    options: list[str] = []
    with_value = {"-p", "-i", "-o", "-l", "-L", "-R", "-D", "-F", "-J"}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in with_value:
            if i + 1 == len(tokens):
                raise ValueError(f"{tok} needs a value")
            value = tokens[i + 1]
            if tok == "-p":
                port = value
            elif tok == "-i":
                key = os.path.expanduser(value)
            elif tok == "-o":
                options += ["-o", value]
            elif tok == "-l":
                user = value
            i += 2
            continue
        if not tok.startswith("-"):
            target = tok
        i += 1
    if target is None:
        raise ValueError("no user@host in the ssh command")
    if "@" not in target and user:
        target = f"{user}@{target}"
    if target.split("@")[-1].endswith("ssh.runpod.io"):
        raise ValueError(
            "that is RunPod's basic SSH proxy, which can't copy files: use the "
            "'SSH over exposed TCP' command (ssh root@<ip> -p <port> ...) from the pod's Connect panel"
        )
    return Remote(target, port, key, options)


def parse_listing(text: str) -> dict[str, tuple[int, int]]:
    """Lines of ``find -printf '%s %T@ %f\\n'`` -> {name: (size, whole-second mtime)}."""
    files = {}
    for line in text.splitlines():
        parts = line.split(" ", 2)
        if len(parts) == 3:
            size, mtime, name = parts
            files[name] = (int(size), int(float(mtime)))
    return files


def newest_step(names: list[str] | set[str]) -> str | None:
    steps = sorted((int(m.group(1)), n) for n in names if (m := STEP.match(n)))
    return steps[-1][1] if steps else None


def step_of(name: str | None) -> int:
    m = STEP.match(name or "")
    return int(m.group(1)) if m else -1


def files_to_copy(remote: dict[str, tuple[int, int]], local_dir: Path) -> list[str]:
    """Remote files that are missing here or differ: the newest step checkpoint only, never temp files."""
    newest = newest_step(list(remote))
    wanted = []
    for name, (size, mtime) in sorted(remote.items()):
        if name in SKIP or name.endswith((".tmp", ".part")) or (STEP.match(name) and name != newest):
            continue
        local = local_dir / name
        if local.is_file() and local.stat().st_size == size and int(local.stat().st_mtime) == mtime:
            continue
        wanted.append(name)
    return wanted


def prune_local(local_dir: Path, keep: int) -> list[Path]:
    """Delete all but the newest ``keep`` step checkpoints here (best.pt is never touched)."""
    steps = sorted((int(m.group(1)), p) for p in local_dir.iterdir() if (m := STEP.match(p.name)))
    removed = [p for _, p in steps[:-keep]] if keep > 0 else []
    for p in removed:
        p.unlink()
    return removed


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(16 << 20):
            h.update(block)
    return h.hexdigest()


def progress_line(local_dir: Path) -> str:
    """'step 1,200 / 9,537 (12.6%), loss 3.812' from the copied metrics, if there are any."""
    config, metrics = local_dir / "config.yaml", local_dir / "metrics.jsonl"
    if not metrics.is_file():
        return ""
    step = loss = None
    for line in metrics.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "train/loss" in record:
            step, loss = record["step"], record["train/loss"]
    if step is None:
        return ""
    text = config.read_text(encoding="utf-8") if config.is_file() else ""
    m = re.search(r"^max_steps: (\d+)", text, re.M)
    total = f" / {int(m.group(1)):,} ({100 * step / int(m.group(1)):.1f}%)" if m else ""
    return f"step {step:,}{total}, train loss {loss:.3f}"


def run(args: list[str], capture: bool = False) -> str:
    out = subprocess.run(args, check=True, text=True, capture_output=capture)
    return out.stdout if capture else ""


def sync_once(remote: Remote, run_dir: str, local_dir: Path, keep: int) -> bool:
    """One round of copying. Returns True when the pod's run has ended and the copy is checked."""
    find = f"find {shlex.quote(run_dir)} -maxdepth 1 -type f -printf '%s %T@ %f\\n' 2>/dev/null || true"
    listing = run(remote.ssh(find), capture=True)
    files = parse_listing(listing)
    if not files:
        print(f"  nothing in {run_dir} on the pod yet")
        return False
    local_dir.mkdir(parents=True, exist_ok=True)
    here = newest_step([p.name for p in local_dir.iterdir()])
    there = newest_step(list(files))
    if step_of(here) > step_of(there):
        raise SystemExit(
            f"{local_dir} is ahead of the pod ({here} > {there}): this copy has trained further "
            "on its own, so nothing is copied over it"
        )
    done = DONE in files  # listed before copying, so everything copied below is final
    for name in files_to_copy(files, local_dir):
        size = files[name][0]
        t0 = time.perf_counter()
        part = local_dir / (name + ".part")
        run(remote.scp(f"{run_dir}/{name}", part))
        copied = part.stat().st_size
        if copied != size:
            part.unlink()
            raise OSError(f"{name}: copied {copied} bytes, expected {size}")
        os.replace(part, local_dir / name)
        if size > 50e6:
            print(f"  copied {name} ({size / 1e9:.2f} GB, {time.perf_counter() - t0:.0f} s)")
    with contextlib.suppress(subprocess.CalledProcessError):  # TensorBoard events: small; may not exist
        run(remote.scp(f"{run_dir}/tb", local_dir, recursive=True), capture=True)
    for p in prune_local(local_dir, keep):
        print(f"  deleted the older {p.name} here")
    if line := progress_line(local_dir):
        print(f"  {line}")
    if not done:
        return False

    print(f"  the pod says: {(local_dir / DONE).read_text(encoding='utf-8').splitlines()[0]}")
    checked = [n for n in ("best.pt", newest_step(list(files))) if n and n in files]
    sums = run(remote.ssh(f"cd {shlex.quote(run_dir)} && sha256sum {' '.join(checked)}"), True)
    theirs = {name: digest for digest, name in (line.split() for line in sums.splitlines())}
    for name in checked:
        if sha256(local_dir / name) != theirs.get(name):
            (local_dir / name).unlink()
            raise OSError(f"{name}: SHA-256 differs from the pod's copy (deleted here; it is copied again)")
        print(f"  {name}: SHA-256 matches the pod")
    note = shlex.quote(f"copied to the PC at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    run(remote.ssh(f"echo {note} > {shlex.quote(f'{run_dir}/{PULLED}')}"))
    return True


def main() -> None:
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=fmt)
    parser.add_argument("ssh_command", help='the pod\'s "SSH over exposed TCP" command, in quotes')
    parser.add_argument("--run", default=RUN, help="run name (folder under runs/)")
    parser.add_argument("--remote-repo", default=REMOTE_REPO, help="the repository folder on the pod")
    parser.add_argument("--dest", type=Path, help="local run folder (default: runs/<run>)")
    parser.add_argument("--every", type=float, default=30, help="minutes between copies (0 = copy once)")
    parser.add_argument("--keep", type=int, default=2, help="step checkpoints to keep here")
    args = parser.parse_args()
    try:
        remote = parse_ssh_command(args.ssh_command)
    except ValueError as e:
        parser.error(str(e))
    run_dir = f"{args.remote_repo}/runs/{args.run}"
    local_dir = args.dest or Path("runs") / args.run
    print(f"copying {remote.target}:{run_dir} -> {local_dir.resolve()}")

    while True:
        print(time.strftime("%H:%M"), "checking the pod")
        try:
            if sync_once(remote, run_dir, local_dir, args.keep):
                print(
                    "\nAll copied and checked. train.sh now stops the pod; once you don't need it any more, "
                    "terminate it (Pods -> ... -> Terminate) so its disk stops costing money."
                )
                return
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"  copy failed ({e}); {'trying again later' if args.every else 'stopping'}")
            if not args.every:
                sys.exit(1)
        if not args.every:
            return
        print(f"  next copy in {args.every:g} min (Ctrl-C quits this; training on the pod goes on)")
        try:
            time.sleep(args.every * 60)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
