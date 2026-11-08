"""``generate``: continue a text prompt with any Aupus model (Milestone 11).

A Base model continues text the way web pages go on. It doesn't answer questions;
for that, use ``chat`` with the Instruct model. Examples::

    python scripts/generate.py --model release/aupus-0.1-base --prompt "The water cycle is"
    python scripts/generate.py --model runs/tiny/best.pt --prompt "In 1492," --temperature 0 -n 40
    echo "A prime number is" | python scripts/generate.py --model release/aupus-0.1-base

The text is printed as it is generated. Sampling settings are explained in
``aupus/model/generate.py``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

import torch

from aupus.cli.streaming import TextStream
from aupus.model import stream
from aupus.model.io import LoadedModel, load_model
from aupus.tokenizer import EOS
from aupus.train import resolve_device


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, required=True, help="release directory or checkpoint .pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])


def add_sampling_args(parser: argparse.ArgumentParser, temperature: float, max_new: int) -> None:
    g = parser.add_argument_group("sampling")
    g.add_argument("-n", "--max-new-tokens", type=int, default=max_new)
    g.add_argument("--temperature", type=float, default=temperature, help="0 = greedy (always the top token)")
    g.add_argument("--top-k", type=int, default=50, help="0 = off")
    g.add_argument("--top-p", type=float, default=0.95, help="1 = off")
    g.add_argument("--repetition-penalty", type=float, default=1.1, help="1 = off")
    g.add_argument("--seed", type=int, default=None, help="fix the dice for reproducible output")


def sampling_kwargs(args: argparse.Namespace, device: torch.device) -> dict:
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    return {
        "temperature": args.temperature,
        "top_k": args.top_k or None,
        "top_p": args.top_p if args.top_p < 1 else None,
        "repetition_penalty": args.repetition_penalty,
        "generator": generator,
    }


def open_model(args: argparse.Namespace) -> LoadedModel:
    return load_model(args.model, device=resolve_device(args.device))


def complete(loaded: LoadedModel, prompt: str, max_new_tokens: int, out: TextIO, **sampling: object) -> str:
    """Stream a continuation of ``prompt`` to ``out``; return the generated text."""
    tok = loaded.tokenizer
    eos = tok.special_id(EOS)
    window = loaded.model.cfg.context_length
    ids = [eos, *tok.encode(prompt)][-(window - 1) :]  # keep the end of an over-long prompt
    text = TextStream(tok)
    for token in stream(loaded.model, ids, max_new_tokens, stop_ids=[eos], **sampling):  # type: ignore[arg-type]
        out.write(text.push(token))
        out.flush()
    out.write(text.flush())
    return text.text


def main(argv: Sequence[str] | None = None, out: TextIO = sys.stdout) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser)
    parser.add_argument("--prompt", help="text to continue (default: read from stdin)")
    add_sampling_args(parser, temperature=0.8, max_new=200)
    args = parser.parse_args(argv)

    prompt = args.prompt if args.prompt is not None else sys.stdin.read().rstrip("\n")
    loaded = open_model(args)
    device = next(loaded.model.parameters()).device
    out.write(prompt)
    complete(loaded, prompt, args.max_new_tokens, out, **sampling_kwargs(args, device))
    out.write("\n")
