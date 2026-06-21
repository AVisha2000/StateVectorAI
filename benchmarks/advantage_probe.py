#!/usr/bin/env python3
"""Quantum-advantage potential probe (Huang et al. geometric-difference test).

For each qubit count:
  1. compute g(K_C || K_Q) against a classical kernel family (label-free:
     answers "is advantage even POSSIBLE on this data distribution?"),
  2. run the positive control — kernel regression on labels engineered to
     maximally favor the quantum kernel (quantum must win here or the
     detector is broken),
  3. run the negative control — labels engineered for the best classical
     kernel (classical must win/tie),
  4. log off-diagonal kernel concentration (the exponential-concentration
     failure mode that eventually destroys g's usefulness at scale).

Usage:
    python benchmarks/advantage_probe.py --qubits 4 6 8 --samples 240
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.quantum.advantage import (  # noqa: E402
    advantage_experiment,
    best_classical_r2,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ansatz", default="reuploading")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    rows = []
    for n in args.qubits:
        t0 = time.time()
        rep = advantage_experiment(
            n_qubits=n,
            n_samples=args.samples,
            n_layers=args.layers,
            ansatz=args.ansatz,
            seed=args.seed,
        )
        row = {
            "n_qubits": n,
            "g_min": rep.g_min,
            "sqrt_N": float(np.sqrt(rep.n_samples)),
            "r2_quantum_on_engineered": rep.r2_engineered["quantum"],
            "r2_best_classical_on_engineered": best_classical_r2(rep.r2_engineered),
            "r2_quantum_on_classical_labels": rep.r2_classical_natural["quantum"],
            "r2_best_classical_on_classical_labels": best_classical_r2(
                rep.r2_classical_natural
            ),
            "kq_offdiag_mean": rep.kq_offdiag_mean,
            "kq_offdiag_std": rep.kq_offdiag_std,
            "wall_seconds": round(time.time() - t0, 2),
        }
        rows.append(row)
        print(
            f"n={n:2d}  g_min={row['g_min']:6.2f} (sqrt(N)={row['sqrt_N']:.1f})  "
            f"[engineered] Q={row['r2_quantum_on_engineered']:.3f} "
            f"C={row['r2_best_classical_on_engineered']:.3f}   "
            f"[classical-natural] Q={row['r2_quantum_on_classical_labels']:.3f} "
            f"C={row['r2_best_classical_on_classical_labels']:.3f}   "
            f"offdiag={row['kq_offdiag_mean']:.3f}±{row['kq_offdiag_std']:.3f}"
            f"  ({row['wall_seconds']}s)"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"advantage_{args.ansatz}_L{args.layers}"
    with open(out / f"{stem}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # verdicts
    pos_ok = all(
        r["r2_quantum_on_engineered"]
        > r["r2_best_classical_on_engineered"] + 0.05
        for r in rows
    )
    neg_ok = all(
        r["r2_best_classical_on_classical_labels"]
        >= r["r2_quantum_on_classical_labels"] - 0.05
        for r in rows
    )
    print(
        f"\npositive control (quantum wins on engineered labels): "
        f"{'PASS' if pos_ok else 'FAIL'}"
    )
    print(
        f"negative control (classical wins/ties on classical labels): "
        f"{'PASS' if neg_ok else 'FAIL'}"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ns = [r["n_qubits"] for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

        ax1.plot(ns, [r["g_min"] for r in rows], "o-", label="g vs best classical")
        ax1.axhline(rows[0]["sqrt_N"], ls="--", c="gray", label="√N (max possible)")
        ax1.axhline(1.0, ls=":", c="gray", label="g≈1 (no advantage possible)")
        ax1.set_xlabel("qubits")
        ax1.set_ylabel("geometric difference g")
        ax1.set_title(f"Advantage potential ({args.ansatz}, L={args.layers}, "
                      f"N={args.samples})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        width = 0.35
        xs = np.arange(len(ns))
        ax2.bar(
            xs - width / 2,
            [r["r2_quantum_on_engineered"] for r in rows],
            width,
            label="quantum kernel",
            color="tab:purple",
        )
        ax2.bar(
            xs + width / 2,
            [r["r2_best_classical_on_engineered"] for r in rows],
            width,
            label="best classical",
            color="tab:orange",
        )
        ax2.set_xticks(xs, [str(n) for n in ns])
        ax2.set_xlabel("qubits")
        ax2.set_ylabel("test R² on ENGINEERED labels")
        ax2.set_title("Positive control: separation where g is large")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(out / f"{stem}.png", dpi=150)
        print(f"plot -> {out / (stem + '.png')}")
    except ImportError:
        print("matplotlib not installed; skipped plot")

    if args.mlflow:
        import mlflow

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("qllm-advantage")
        with mlflow.start_run(run_name=stem):
            mlflow.log_params(
                {
                    "ansatz": args.ansatz,
                    "layers": args.layers,
                    "samples": args.samples,
                }
            )
            for row in rows:
                mlflow.log_metrics(
                    {k: v for k, v in row.items() if isinstance(v, (int, float))},
                    step=row["n_qubits"],
                )
            mlflow.log_artifact(str(out / f"{stem}.csv"))
            if (out / f"{stem}.png").exists():
                mlflow.log_artifact(str(out / f"{stem}.png"))
        print("logged to MLflow experiment 'qllm-advantage'")


if __name__ == "__main__":
    main()
