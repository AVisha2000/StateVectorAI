#!/usr/bin/env python3
"""Screen candidate datasets for quantum-favoredness BEFORE training.

For each dataset: g_min (advantage room of the feature map on this input
distribution) and s_ratio = s_classical / s_quantum on the actual
next-token proxy targets (does the task structure live in the
quantum-favored subspace?). Train only where both are large — this is
the compute-allocation policy that v0.3 made possible.

Usage:
    python benchmarks/data_screen.py --samples 240
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.data.quantum_seq import (  # noqa: E402
    markov_control_sequences,
    monitored_ising_sequences,
)
from qllm.data.text import CharTokenizer, load_corpus  # noqa: E402
from qllm.quantum.advantage import screen_sequence_dataset  # noqa: E402


def build_candidates(seed: int) -> dict[str, tuple[np.ndarray, int]]:
    out: dict[str, tuple[np.ndarray, int]] = {}

    chaotic, vocab = monitored_ising_sequences(
        n_qubits=6, n_measured=2, n_sequences=32, seq_len=1024,
        theta_zz=np.pi / 4, theta_x=np.pi / 4, seed=seed,
    )
    out["ising-chaotic"] = (chaotic, vocab)

    weak, vocab_w = monitored_ising_sequences(
        n_qubits=6, n_measured=2, n_sequences=32, seq_len=1024,
        theta_zz=np.pi / 4, theta_x=0.3, seed=seed,
    )
    out["ising-weakfield"] = (weak, vocab_w)

    out["markov3-of-chaotic"] = (
        markov_control_sequences(chaotic, vocab, order=3, seed=seed + 1),
        vocab,
    )

    text = load_corpus("data/input.txt")
    tok = CharTokenizer(text)
    out["shakespeare"] = (tok.encode(text), tok.vocab_size)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--window", type=int, default=6, help="screen qubits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    rows = []
    for name, (ids, vocab) in build_candidates(args.seed).items():
        t0 = time.time()
        rep = screen_sequence_dataset(
            ids, vocab, n_qubits=args.window, n_samples=args.samples,
            seed=args.seed,
        )
        # simple predictability context: empirical token entropy (bits)
        counts = np.bincount(np.asarray(ids), minlength=vocab).astype(float)
        p = counts / counts.sum()
        entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        rows.append(
            {
                "dataset": name,
                "vocab": vocab,
                "token_entropy_bits": round(entropy, 3),
                "g_min": round(rep.g_min, 2),
                "s_quantum": round(rep.s_quantum, 2),
                "s_classical_best": round(rep.s_classical_best, 2),
                "s_ratio": round(rep.s_ratio, 3),
                "kq_offdiag_mean": round(rep.kq_offdiag_mean, 4),
                "wall_seconds": round(time.time() - t0, 1),
            }
        )
        r = rows[-1]
        print(
            f"{name:20s} vocab={r['vocab']:3d} H={r['token_entropy_bits']:.2f}b  "
            f"g={r['g_min']:6.2f}  s_Q={r['s_quantum']:8.2f}  "
            f"s_C={r['s_classical_best']:8.2f}  s_ratio={r['s_ratio']:6.3f}  "
            f"({r['wall_seconds']}s)"
        )

    rows.sort(key=lambda r: r["s_ratio"], reverse=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "data_screen.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nranked by s_ratio (quantum-favoredness of the actual task):")
    for r in rows:
        print(f"  {r['s_ratio']:6.3f}  {r['dataset']}")
    print(f"\n-> {out / 'data_screen.csv'}")


if __name__ == "__main__":
    main()
