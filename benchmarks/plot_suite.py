#!/usr/bin/env python3
"""Plot every QNLP method against the classical baseline from the results DB.

Left: val perplexity per variant (mean ± std over seeds), classical
highlighted with a dashed reference line. Right: parameters vs perplexity
(the efficiency view). Reads ONLY the SQLite DB, so it can be re-rendered
any time, including after GPU runs add more rows.

Usage:
    python benchmarks/plot_suite.py --dataset text
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from qllm.resultsdb import ResultsDB  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-name", default="qnlp-v1")
    parser.add_argument("--dataset", default="text")
    parser.add_argument("--db", default="results/qllm_results.db")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    rows = ResultsDB(args.db).fetch(args.suite_name, args.dataset)
    if not rows:
        print("no rows found for this suite/dataset")
        return

    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["variant"], []).append(r)

    stats = []
    for name, rs in by.items():
        ppls = [r["val_ppl"] for r in rs]
        stats.append(
            {
                "variant": name,
                "mean": statistics.mean(ppls),
                "std": statistics.stdev(ppls) if len(ppls) > 1 else 0.0,
                "params": rs[0]["n_params"],
                "n": len(ppls),
            }
        )
    stats.sort(key=lambda s: s["mean"])
    classical = next((s for s in stats if s["variant"] == "classical"), None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    names = [s["variant"] for s in stats]
    means = [s["mean"] for s in stats]
    stds = [s["std"] for s in stats]
    colors = [
        "tab:orange" if n == "classical"
        else "tab:gray" if n.startswith("gru")
        else "tab:purple"
        for n in names
    ]
    y = range(len(names))
    ax1.barh(y, means, xerr=stds, color=colors, alpha=0.85, capsize=3)
    ax1.set_yticks(y, names)
    ax1.invert_yaxis()
    ax1.set_xlabel("validation perplexity (lower = better)")
    ax1.set_title(
        f"QNLP component swaps vs classical — {args.dataset} "
        f"({stats[0]['n']} seeds)"
    )
    if classical:
        ax1.axvline(classical["mean"], ls="--", c="tab:orange", lw=1.2,
                    label="classical baseline")
        ax1.legend(loc="lower right")
    lo = min(means) - 3 * max(max(stds), 0.02)
    hi = max(means) + 3 * max(max(stds), 0.02)
    ax1.set_xlim(max(lo, 1.0), hi)
    ax1.grid(True, alpha=0.3, axis="x")

    for s in stats:
        marker = "*" if s["variant"] == "classical" else "o"
        color = ("tab:orange" if s["variant"] == "classical"
                 else "tab:gray" if s["variant"].startswith("gru")
                 else "tab:purple")
        ax2.errorbar(s["params"], s["mean"], yerr=s["std"], fmt=marker,
                     ms=11 if marker == "*" else 7, color=color, capsize=3)
        ax2.annotate(s["variant"], (s["params"], s["mean"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("trainable parameters (log)")
    ax2.set_ylabel("validation perplexity")
    ax2.set_title("Efficiency view: params vs perplexity")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = Path(args.out) / f"qnlp_suite_{args.dataset}.png"
    fig.savefig(path, dpi=150)
    print(f"plot -> {path}")


if __name__ == "__main__":
    main()
