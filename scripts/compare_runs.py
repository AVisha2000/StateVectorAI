#!/usr/bin/env python3
"""Print a comparison table of MLflow runs (the 'compare runs' workflow)."""
from __future__ import annotations

import argparse

import mlflow
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="qllm")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    df = mlflow.search_runs(experiment_names=[args.experiment])
    if df.empty:
        print(f"No runs found in experiment '{args.experiment}'.")
        return

    cols = {
        "tags.mlflow.runName": "run",
        "params.model.attn_type": "attn",
        "params.model.ffn_type": "ffn",
        "tags.qubits": "qubits",
        "params.n_params": "params",
        "metrics.val_ppl": "val_ppl",
        "metrics.val_bpc": "val_bpc",
        "metrics.q_grad_var_mean": "q_grad_var",
        "metrics.q_meyer_wallach_q": "q_MW",
        "metrics.wall_seconds": "wall_s",
    }
    present = {k: v for k, v in cols.items() if k in df.columns}
    table = df[list(present)].rename(columns=present)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
