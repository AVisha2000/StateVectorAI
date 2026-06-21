#!/usr/bin/env python3
"""2x2 ablation: isolate the quantum circuit's contribution.

Grid (planning-doc design; Quixer-style parameter matching):
  quantum-trained   : VQC FFN, circuit weights trained
  quantum-frozen    : VQC FFN, circuit weights FROZEN at random init
  classical-matched : classical FFN, d_ff solved so total params match
  classical-full    : reference classical model from the base config

Decision rules:
  - quantum-trained must beat quantum-frozen, else circuit training adds nothing
  - quantum-trained must beat classical-matched, else no quantum contribution

Runs are tagged by config stem, results accumulate in
results/ablation_<tag>.csv, and the report regenerates from all rows —
so the grid can be filled across multiple invocations via --only.

Usage:
    python benchmarks/ablation.py --base-config configs/quantum_ffn_sharp.yaml \
        --only quantum-trained quantum-frozen
    python benchmarks/ablation.py --base-config configs/quantum_ffn_sharp.yaml \
        --only classical-matched classical-full
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.config import load_yaml  # noqa: E402
from qllm.data.text import CharTokenizer, load_corpus  # noqa: E402
from qllm.models.model import matched_classical_d_ff  # noqa: E402
from qllm.train.loop import fit  # noqa: E402

ALL_VARIANTS = (
    "quantum-trained",
    "quantum-frozen",
    "classical-matched",
    "classical-full",
)
FIELDS = [
    "variant",
    "seed",
    "n_params",
    "val_loss",
    "val_ppl",
    "wall_seconds",
    "grad_norm_ratio",
]


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append(
                {
                    "variant": r["variant"],
                    "seed": int(r["seed"]),
                    "n_params": int(r["n_params"]),
                    "val_loss": float(r["val_loss"]),
                    "val_ppl": float(r["val_ppl"]),
                    "wall_seconds": float(r["wall_seconds"]),
                    "grad_norm_ratio": (
                        float(r["grad_norm_ratio"])
                        if r.get("grad_norm_ratio") not in (None, "", "None")
                        else None
                    ),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/quantum_ffn_4q.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--only", nargs="+", choices=ALL_VARIANTS, default=None)
    parser.add_argument("--tag", default=None, help="default: config file stem")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    base = load_yaml(args.base_config)
    assert base.model.ffn_type == "quantum", "base config must use a quantum FFN"
    tag = args.tag or Path(args.base_config).stem

    vocab = CharTokenizer(load_corpus(base.data.corpus_path)).vocab_size
    mq = base.model
    d_ff_matched = matched_classical_d_ff(mq, vocab)
    print(f"[{tag}] parameter-matched classical twin: d_ff={d_ff_matched}")

    variants = {
        "quantum-trained": mq,
        "quantum-frozen": dataclasses.replace(
            mq, quantum=dataclasses.replace(mq.quantum, trainable=False)
        ),
        "classical-matched": dataclasses.replace(
            mq, ffn_type="classical", d_ff=d_ff_matched
        ),
        "classical-full": dataclasses.replace(mq, ffn_type="classical"),
    }
    selected = args.only or list(ALL_VARIANTS)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"ablation_{tag}.csv"
    rows = [r for r in _load_existing(csv_path) if r["variant"] not in selected]

    t_start = time.time()
    for name in selected:
        model_cfg = variants[name]
        for seed in args.seeds:
            cfg = dataclasses.replace(
                base,
                model=model_cfg,
                train=dataclasses.replace(base.train, seed=seed, steps=args.steps),
                tracking=dataclasses.replace(
                    base.tracking,
                    experiment="qllm-ablation",
                    run_name=f"abl-{tag}-{name}-s{seed}",
                    log_quantum_diagnostics=False,
                ),
            )
            res = fit(cfg, verbose=False, out_dir=args.out)
            s = res["summary"]
            ratio = s["history"][-1].get("grad_norm_ratio")
            rows.append(
                {
                    "variant": name,
                    "seed": seed,
                    "n_params": s["n_params"],
                    "val_loss": s["val_loss"],
                    "val_ppl": s["val_ppl"],
                    "wall_seconds": s["wall_seconds"],
                    "grad_norm_ratio": ratio,
                }
            )
            extra = f"  g_ratio={ratio:.2e}" if ratio is not None else ""
            print(
                f"{name:18s} seed={seed}  params={s['n_params']:7,d}  "
                f"val_ppl={s['val_ppl']:.3f}  ({s['wall_seconds']:.0f}s){extra}"
            )

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    have = {r["variant"] for r in rows}

    def agg(name: str):
        ppls = [r["val_ppl"] for r in rows if r["variant"] == name]
        mean = statistics.mean(ppls)
        std = statistics.stdev(ppls) if len(ppls) > 1 else 0.0
        n_params = next(r["n_params"] for r in rows if r["variant"] == name)
        return mean, std, n_params, ppls

    lines = [
        f"# 2x2 ablation results — {tag}",
        "",
        f"{args.steps} steps, base={args.base_config}, matched d_ff={d_ff_matched}",
        "",
        "| variant | params | val ppl (mean ± std) | per-seed |",
        "|---|---|---|---|",
    ]
    summary = {}
    for name in ALL_VARIANTS:
        if name not in have:
            continue
        mean, std, n_params, ppls = agg(name)
        summary[name] = (mean, std)
        per_seed = ", ".join(f"{p:.2f}" for p in sorted(ppls))
        lines.append(
            f"| {name} | {n_params:,} | {mean:.3f} ± {std:.3f} | {per_seed} |"
        )

    if have >= set(ALL_VARIANTS):
        qt, qf = summary["quantum-trained"], summary["quantum-frozen"]
        cm = summary["classical-matched"]
        lines += ["", "## Decision rules", ""]

        d_train = qf[0] - qt[0]
        sep_train = d_train > (qt[1] + qf[1])
        lines.append(
            f"- trained vs frozen circuit: Δppl = {d_train:+.3f} "
            f"({'SEPARATED' if sep_train else 'within noise'} at ±1σ) — "
            + (
                "training the circuit helps; it is not just a random feature map."
                if sep_train and d_train > 0
                else "circuit training is NOT clearly beneficial; the quantum "
                "layer may be acting as a fixed random feature map."
            )
        )

        d_cls = cm[0] - qt[0]
        sep_cls = abs(d_cls) > (qt[1] + cm[1])
        lines.append(
            f"- trained quantum vs parameter-matched classical: "
            f"Δppl = {d_cls:+.3f} "
            f"({'SEPARATED' if sep_cls else 'within noise'} at ±1σ) — "
            + (
                "quantum FFN beats its equal-parameter classical twin here."
                if sep_cls and d_cls > 0
                else "classical twin beats the quantum FFN at equal parameters."
                if sep_cls
                else "no separable difference; no evidence of quantum "
                "contribution."
            )
        )
    else:
        missing = sorted(set(ALL_VARIANTS) - have)
        lines += ["", f"(partial grid — missing: {', '.join(missing)})"]

    report = "\n".join(lines) + "\n"
    (out / f"ablation_{tag}.md").write_text(report)
    print("\n" + report)
    print(f"wall: {time.time() - t_start:.0f}s -> {out / f'ablation_{tag}.md'}")


if __name__ == "__main__":
    main()
