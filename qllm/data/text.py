"""Text data utilities: corpus loading, char-level tokenization, batching.

Char-level LM on tiny-shakespeare is the Phase-1 testing ground: small enough
to iterate fast on 1 CPU, real enough that perplexity differences are
meaningful. Word-level PTB/WikiText-2 slots in later behind the same API.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

_SYNTHETIC = (
    "the quick brown fox jumps over the lazy dog. "
    "pack my box with five dozen liquor jugs. "
    "how vexingly quick daft zebras jump! "
) * 2000


def load_corpus(path: str | Path, synthetic_fallback: bool = True) -> str:
    """Load a text corpus from disk; optionally fall back to synthetic text.

    The synthetic fallback keeps tests and CI hermetic (no network).
    """
    path = Path(path)
    if path.exists():
        return path.read_text(encoding="utf-8")
    if synthetic_fallback:
        return _SYNTHETIC
    raise FileNotFoundError(
        f"Corpus not found at {path}. Download it, e.g.:\n"
        f"  curl -o {path} {TINY_SHAKESPEARE_URL}"
    )


class CharTokenizer:
    """Reversible character-level tokenizer built from a corpus."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[c] for c in text], dtype=np.int32)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def train_val_split(ids: np.ndarray, val_fraction: float = 0.1):
    if ids.ndim == 2:
        n_val = max(1, int(ids.shape[0] * val_fraction))
        if n_val >= ids.shape[0]:
            raise ValueError("trajectory split requires at least one train trajectory")
        return ids[:-n_val], ids[-n_val:]
    n_val = max(1, int(len(ids) * val_fraction))
    return ids[:-n_val], ids[-n_val:]


def sample_batch(
    rng: np.random.Generator, ids: np.ndarray, batch_size: int, seq_len: int
) -> np.ndarray:
    """Random contiguous crops, shape (batch, seq_len + 1).

    Slice [:, :-1] as inputs and [:, 1:] as next-token targets.
    """
    if ids.ndim == 2:
        max_start = ids.shape[1] - seq_len
        if max_start <= 0:
            raise ValueError("seq_len must be shorter than each trajectory")
        rows = rng.integers(0, ids.shape[0], size=batch_size)
        starts = rng.integers(0, max_start, size=batch_size)
        return np.stack([
            ids[row, start : start + seq_len + 1]
            for row, start in zip(rows, starts, strict=True)
        ]).astype(np.int32)
    max_start = len(ids) - seq_len - 1
    starts = rng.integers(0, max_start, size=batch_size)
    return np.stack([ids[s : s + seq_len + 1] for s in starts]).astype(np.int32)
