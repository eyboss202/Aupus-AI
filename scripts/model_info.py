"""Print exact parameter counts and estimated FLOPs/token for model presets.

Usage:
    python scripts/model_info.py                 # all presets in configs/model/
    python scripts/model_info.py small base      # selected presets
    python scripts/model_info.py --tokens-per-param 20
"""

from __future__ import annotations

import argparse

from aupus.model.config import PRESET_DIR, ModelConfig, flops_per_token, param_count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("presets", nargs="*", help="preset names (default: all)")
    ap.add_argument(
        "--tokens-per-param", type=float, default=20.0, help="Chinchilla-style token budget ratio"
    )
    args = ap.parse_args()
    names = args.presets or sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))

    header = (
        f"{'preset':<8}{'total':>12}{'non-emb':>12}{'embed':>12}{'ctx':>6}"
        f"{'GFLOP/tok train':>17}{'budget (tok)':>15}{'train FLOPs':>13}"
    )
    print(header)
    print("-" * len(header))
    for name in names:
        cfg = ModelConfig.preset(name)
        p = param_count(cfg)
        f = flops_per_token(cfg)
        budget = args.tokens_per_param * p["total"]
        print(
            f"{name:<8}{p['total']:>12,}{p['non_embedding']:>12,}{p['embedding']:>12,}{cfg.context_length:>6}"
            f"{f['train'] / 1e9:>17.3f}{budget / 1e9:>14.2f}B{budget * f['train']:>13.2e}"
        )


if __name__ == "__main__":
    main()
