#!/usr/bin/env python3
"""Separation-curve plot from memory-sweep-v2 in the results DB.

Left: classical GRU capacity (params) needed to reach the planted
quantum-filter gain, vs memory qubits m (the quantum model uses m+1
qubits). Right: per-m memory-gain vs GRU size, with the planted-filter
line — the gap at small GRU sizes is the separation.
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
    suite = "memory-sweep-v2"
    ms, gains_by_model, planted = [], {}, {}
    rows = [r for r in db.fetch(suite) if "-smoke" not in r["dataset"]]
    metrics = {(x["variant"], x["dataset"]): x["value"]
               for x in db.fetch_metrics(suite) if x["name"] == "memory_gain"}

    for r in rows:
        mm = int(re.search(r"-m(\d+)-", r["dataset"]).group(1))
        ms.append(mm)
        g = metrics.get((r["variant"], r["dataset"]))
        if g is None:
            continue
        if r["variant"] == "planted":
            planted[mm] = g
        else:
            gains_by_model.setdefault(r["variant"], {})[mm] = (
                r["n_params"], g)
    ms = sorted(set(ms))

    # capacity to match planted: smallest GRU whose gain >= 0.97*planted
    match_m, match_params = [], []
    for mm in ms:
        if mm not in planted:
            continue
        cands = []
        for model, d in gains_by_model.items():
            if mm in d:
                params, g = d[mm]
                if g >= 0.97 * planted[mm]:
                    cands.append(params)
        if cands:
            match_m.append(mm)
            match_params.append(min(cands))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Left: gain retention at FIXED small capacity as m grows. This is the
    # defensible separation signal — no threshold-crossing artifact. A
    # fixed-size recurrence loses gain as the belief filter (~4^m) outgrows
    # its hidden state; we plot each small GRU and the planted filter.
    small_models = ["gru8", "gru16", "gru32"]
    for model in small_models:
        d = gains_by_model.get(model, {})
        xs = [mm for mm in ms if mm in d]
        ys = [d[mm][1] for mm in xs]
        if xs:
            ax1.plot(xs, ys, "o-", lw=2, ms=7, label=model)
    pxs = [mm for mm in ms if mm in planted]
    ax1.plot(pxs, [planted[mm] for mm in pxs], "s--", color="black",
             lw=2, ms=8, label="planted quantum filter")
    ax1.set_xlabel("memory qubits m  (quantum filter uses m+1 qubits)")
    ax1.set_ylabel("memory gain (ppl below markov-3)")
    ax1.set_title("Gain retention vs memory size")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(True, alpha=0.3)

    colors = plt.cm.viridis(
        [i / max(len(ms) - 1, 1) for i in range(len(ms))])
    for mm, c in zip(ms, colors):
        pts = sorted((p, g) for m_, (p, g) in
                     [(model, gains_by_model[model][mm])
                      for model in gains_by_model if mm in gains_by_model[model]])
        if not pts:
            continue
        xs = [p for p, _ in pts]
        ys = [g for _, g in pts]
        ax2.plot(xs, ys, "o-", color=c, label=f"m={mm}", ms=5)
        if mm in planted:
            ax2.axhline(planted[mm], ls="--", color=c, alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("GRU parameters (log)")
    ax2.set_ylabel("memory gain (ppl below markov-3)")
    ax2.set_title("Gain vs capacity; dashed = planted quantum filter")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path("results") / "separation_curve.png"
    fig.savefig(out, dpi=150)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
