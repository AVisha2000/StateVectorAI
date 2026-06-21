"""Dataset dispatch: every DataConfig kind -> (ids, tokenizer-like).

This is the data-side plugin point, mirroring the model-side registries:
the training loop is dataset-agnostic, and new generators slot in with a
config string.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import DataConfig
from .quantum_seq import (
    IdentityTokenizer,
    markov_control_sequences,
    monitored_ising_sequences,
)
from .text import CharTokenizer, load_corpus

DATASET_KINDS = ("text", "monitored_ising", "markov_control")


def load_dataset(cfg: DataConfig):
    """Return (ids: np.int32 array, tokenizer with vocab_size/encode/decode)."""
    if cfg.kind == "text":
        text = load_corpus(cfg.corpus_path)
        tokenizer = CharTokenizer(text)
        return tokenizer.encode(text), tokenizer

    if cfg.kind == "seq_cancellation":
        from .seq_cancellation import seq_cancellation

        cache_dir = Path("results/.data_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = (f"seqcancel_v{cfg.ctx_observables}_w{cfg.ctx_context_size}"
               f"_d{cfg.seq_cancel_density:.3f}_s{cfg.gen_sequences}"
               f"_l{cfg.gen_len}_seed{cfg.gen_seed}")
        cache = cache_dir / f"{key}.npz"
        if cache.exists():
            payload = np.load(cache)
            return payload["ids"].astype(np.int32), IdentityTokenizer(int(payload["vocab"]))
        ids, vocab = seq_cancellation(
            n_sequences=cfg.gen_sequences, seq_len=cfg.gen_len,
            vocab_size=cfg.ctx_observables, parity_window=cfg.ctx_context_size,
            density=cfg.seq_cancel_density, seed=cfg.gen_seed)
        ids = np.asarray(ids, dtype=np.int32)
        np.savez(cache, ids=ids, vocab=vocab)
        return ids, IdentityTokenizer(vocab)

    if cfg.kind == "interference":
        from .interference_task import interference_sequences

        cache_dir = Path("results/.data_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = f"interference_v{cfg.ctx_observables}_s{cfg.gen_sequences}_l{cfg.gen_len}_seed{cfg.gen_seed}"
        cache = cache_dir / f"{key}.npz"
        if cache.exists():
            payload = np.load(cache)
            return payload["ids"].astype(np.int32), IdentityTokenizer(int(payload["vocab"]))
        ids, vocab = interference_sequences(
            n_sequences=cfg.gen_sequences, seq_len=cfg.gen_len,
            vocab_size=cfg.ctx_observables, seed=cfg.gen_seed)
        ids = np.asarray(ids, dtype=np.int32)
        np.savez(cache, ids=ids, vocab=vocab)
        return ids, IdentityTokenizer(vocab)

    if cfg.kind == "contextual":
        from .contextual import contextual_parity_sequences

        cache_dir = Path("results/.data_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = (f"contextual_o{cfg.ctx_observables}_c{cfg.ctx_context_size}"
               f"_live{cfg.ctx_n_live}_s{cfg.gen_sequences}_l{cfg.gen_len}"
               f"_seed{cfg.gen_seed}")
        cache = cache_dir / f"{key}.npz"
        if cache.exists():
            payload = np.load(cache)
            globals()["_LAST_CTX_MASK"] = payload["mask"]
            return payload["ids"].astype(np.int32), IdentityTokenizer(
                int(payload["vocab"]))
        ids, vocab, mask = contextual_parity_sequences(
            n_sequences=cfg.gen_sequences, seq_len=cfg.gen_len,
            n_observables=cfg.ctx_observables,
            context_size=cfg.ctx_context_size, n_live=cfg.ctx_n_live,
            seed=cfg.gen_seed)
        ids = np.asarray(ids, dtype=np.int32)
        np.savez(cache, ids=ids, vocab=vocab, mask=mask)
        globals()["_LAST_CTX_MASK"] = mask
        return ids, IdentityTokenizer(vocab)

    if cfg.kind in ("monitored_ising", "markov_control"):
        cache_dir = Path("results/.data_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = (
            f"{cfg.kind}_q{cfg.gen_qubits}_m{cfg.gen_measured}"
            f"_s{cfg.gen_sequences}_l{cfg.gen_len}"
            f"_zz{cfg.gen_theta_zz:.4f}_x{cfg.gen_theta_x:.4f}"
            f"_spt{cfg.gen_steps_per_token}_seed{cfg.gen_seed}"
            f"_k{cfg.markov_order}"
        )
        cache = cache_dir / f"{key}.npz"
        if cache.exists():
            payload = np.load(cache)
            return payload["ids"].astype(np.int32), IdentityTokenizer(
                int(payload["vocab"])
            )

        ids, vocab = monitored_ising_sequences(
            n_qubits=cfg.gen_qubits,
            n_measured=cfg.gen_measured,
            n_sequences=cfg.gen_sequences,
            seq_len=cfg.gen_len,
            theta_zz=cfg.gen_theta_zz,
            theta_x=cfg.gen_theta_x,
            steps_per_token=cfg.gen_steps_per_token,
            seed=cfg.gen_seed,
        )
        if cfg.kind == "markov_control":
            ids = markov_control_sequences(
                ids, vocab, order=cfg.markov_order, seed=cfg.gen_seed + 1
            )
        ids = np.asarray(ids, dtype=np.int32)
        np.savez(cache, ids=ids, vocab=vocab)
        return ids, IdentityTokenizer(vocab)

    raise ValueError(f"Unknown data kind '{cfg.kind}'. Options: {DATASET_KINDS}")
