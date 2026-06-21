#!/usr/bin/env python3
"""Barren-plateau scaling probe: circuit diagnostics vs qubit count.

This is the Phase-3 harness. For each qubit count it computes:
  - Var[dC/d theta] over random inits (the barren-plateau signature),
  - Meyer-Wallach entangling capability Q,
  - expressibility KL vs the Haar distribution,
then fits the exponential gradient-variance decay rate and renders the
scaling plots.

Usage:
    python benchmarks/scaling_probe.py --qubits 2 4 6 8 10 12 --layers 2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.quantum import metrics as qmetrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ansatz", default="reuploading")
    parser.add_argument("--grad-samples", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--mw-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    parser.add_argument("--mlflow", action="store_true", help="Log rows to MLflow")
    args = parser.parse_args()

    rows = []
    for n in args.qubits:
        t0 = time.time()
        row = {"n_qubits": n, "n_layers": args.layers, "ansatz": args.ansatz}
        row.update(
            qmetrics.gradient_variance(
                n, args.layers, args.ansatz,
                n_samples=args.grad_samples, seed=args.seed,
            )
        )
        row["meyer_wallach_q"] = qmetrics.average_meyer_wallach(
            n, args.layers, args.ansatz, n_samples=args.mw_samples, seed=args.seed
        )
        row["expressibility_kl"] = qmetrics.expressibility_kl(
            n, args.layers, args.ansatz, n_pairs=args.pairs, seed=args.seed
        )
        row["wall_seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        print(
            f"n={n:3d}  grad_var(first)={row['grad_var_first_param']:.3e}  "
            f"grad_var(mean)={row['grad_var_mean']:.3e}  "
            f"Q={row['meyer_wallach_q']:.3f}  "
            f"KL={row['expressibility_kl']:.3f}  ({row['wall_seconds']}s)"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"scaling_{args.ansatz}_L{args.layers}"

    with open(out / f"{stem}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Exponential decay fit: log(var) ~ a*n + b  ->  var shrinks by e^a per qubit.
    # Use the MEAN variance over parameters: individual params can be
    # structurally zero (e.g. the first RZ acts on |0> and only phases it),
    # which would corrupt a single-parameter fit.
    ns = np.array([r["n_qubits"] for r in rows], dtype=float)
    log_var = np.log(np.array([r["grad_var_mean"] for r in rows]))
    slope, intercept = np.polyfit(ns, log_var, 1)
    decay_per_qubit = float(np.exp(slope))
    fit_info = {
        "log_var_slope": float(slope),
        "variance_decay_factor_per_qubit": decay_per_qubit,
        "interpretation": (
            f"Var[grad] shrinks by ~{(1 - decay_per_qubit) * 100:.0f}% per added "
            "qubit (exponential decay => barren-plateau signature)"
            if decay_per_qubit < 1
            else "No exponential decay detected in this range"
        ),
    }
    (out / f"{stem}_fit.json").write_text(json.dumps(fit_info, indent=2))
    print(f"\nfit: {fit_info['interpretation']}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

        ax1.semilogy(ns, np.exp(log_var), "o-", label="Var[∂C/∂θ] (mean over params)")
        ax1.semilogy(
            ns, np.exp(intercept + slope * ns), "--",
            label=f"exp fit: ×{decay_per_qubit:.2f}/qubit",
        )
        ax1.set_xlabel("qubits")
        ax1.set_ylabel("gradient variance (log scale)")
        ax1.set_title(f"Barren-plateau probe ({args.ansatz}, L={args.layers})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            ns, [r["meyer_wallach_q"] for r in rows], "s-", color="tab:green",
            label="Meyer-Wallach Q",
        )
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel("qubits")
        ax2.set_ylabel("entangling capability Q", color="tab:green")
        ax2b = ax2.twinx()
        ax2b.plot(
            ns, [r["expressibility_kl"] for r in rows], "^-", color="tab:purple",
            label="expressibility KL",
        )
        ax2b.set_ylabel("expressibility KL (lower = more expressive)",
                        color="tab:purple")
        ax2.set_title("Circuit characterization vs scale")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out / f"{stem}.png", dpi=150)
        print(f"plot -> {out / (stem + '.png')}")
    except ImportError:
        print("matplotlib not installed; skipped plot")

    if args.mlflow:
        import mlflow

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("qllm-scaling")
        with mlflow.start_run(run_name=stem):
            mlflow.log_params(
                {"ansatz": args.ansatz, "layers": args.layers,
                 "grad_samples": args.grad_samples}
            )
            for row in rows:
                mlflow.log_metrics(
                    {k: v for k, v in row.items() if isinstance(v, (int, float))},
                    step=row["n_qubits"],
                )
            mlflow.log_metrics(
                {k: v for k, v in fit_info.items() if isinstance(v, float)}
            )
            mlflow.log_artifact(str(out / f"{stem}.csv"))
            if (out / f"{stem}.png").exists():
                mlflow.log_artifact(str(out / f"{stem}.png"))
        print("logged to MLflow experiment 'qllm-scaling'")


if __name__ == "__main__":
    main()
