#!/usr/bin/env python3
"""Train a QLLM variant from a YAML config.

Usage:
    python scripts/train.py --config configs/classical_small.yaml
    python scripts/train.py --config configs/quantum_ffn_4q.yaml --steps 50
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.config import load_yaml  # noqa: E402
from qllm.train.loop import fit, generate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--steps", type=int, default=None, help="Override train.steps")
    parser.add_argument("--run-name", default=None, help="Override tracking.run_name")
    parser.add_argument("--no-track", action="store_true", help="Disable MLflow")
    parser.add_argument(
        "--sample", type=int, default=200, help="Chars to sample after training (0=off)"
    )
    parser.add_argument("--prompt", default="ROMEO:", help="Generation prompt")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.steps is not None:
        cfg = dataclasses.replace(
            cfg, train=dataclasses.replace(cfg.train, steps=args.steps)
        )
    if args.run_name is not None:
        cfg = dataclasses.replace(
            cfg, tracking=dataclasses.replace(cfg.tracking, run_name=args.run_name)
        )
    if args.no_track:
        cfg = dataclasses.replace(
            cfg, tracking=dataclasses.replace(cfg.tracking, enabled=False)
        )

    result = fit(cfg)

    if args.sample > 0:
        text = generate(
            result["model"],
            result["state"].params,
            result["tokenizer"],
            prompt=args.prompt,
            max_new_tokens=args.sample,
        )
        print("\n--- sample ---")
        print(text)
        print("--------------")


if __name__ == "__main__":
    main()
