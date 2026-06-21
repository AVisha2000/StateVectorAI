#!/usr/bin/env python3
"""Plot the contextuality memory wall from contextual-v1.

Constrained-position accuracy (parity bits predictable only from context
memory) vs GRU capacity, one curve per contextuality depth n_live. Theory
(arXiv:2209.14353): deeper contextuality needs more classical memory to
reach ceiling; a contextual quantum recurrence would need O(n). The upward
shift of each curve with capacity, and the rightward push with n_live, is
the classical memory wall.
"""
from __future__ import annotations

import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from qllm.resultsdb import ResultsDB  # noqa: E402


def main() -> None:
    db = ResultsDB()
    acc = {(x["variant"], x["dataset"]): x["value"]
           for x in db.fetch_metrics("contextual-v1")
           if x["name"] == "constrained_acc"}
    by_live: dict[int, list[tuple[int, float]]] = {}
    for r in db.fetch("contextual-v1"):
        live = int(re.search(r"live(\d+)", r["dataset"]).group(1))
        a = acc.get((r["variant"], r["dataset"]))
        if a is not None:
            by_live.setdefault(live, []).append((r["n_params"], a))

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = plt.cm.plasma(
        [i / max(len(by_live) - 1, 1) for i in range(len(by_live))])
    for (live, pts), c in zip(sorted(by_live.items()), colors):
        pts = sorted(set(pts))
        xs = [p for p, _ in pts]
        ys = [a for _, a in pts]
        ax.plot(xs, ys, "o-", color=c, lw=2, ms=7,
                label=f"n_live={live}")
    ax.axhline(0.5, ls=":", color="gray", label="chance (parity bit)")
    ax.axhline(1.0, ls="--", color="green", alpha=0.4, label="ceiling")
    ax.set_xscale("log")
    ax.set_xlabel("GRU parameters (log)")
    ax.set_ylabel("constrained-token accuracy (parity bits)")
    ax.set_title("Classical memory wall on a contextual task\n"
                 "(higher n_live = deeper contextuality)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = Path("results") / "contextual_wall.png"
    fig.savefig(out, dpi=150)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
