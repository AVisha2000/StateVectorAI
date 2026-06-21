"""Quantum-generated sequence datasets (and matched classical controls).

The v0.3 synthesis said: on classical text, quantum layers degenerate to
random features because the labels don't live in the quantum-favored
subspace. The natural place to look for quantum-favored sequence structure
is data that IS quantum:

``monitored_ising_sequences`` emits measurement records from kicked-Ising
dynamics with mid-circuit measurements. Each step: apply the Floquet
unitary U = U_x(theta_x) . U_zz(theta_zz), then projectively measure the
first k qubits -> one token in {0..2^k-1}; the n-k unmeasured qubits
carry *quantum memory* between tokens. This is a quantum hidden Markov
process — the class for which quantum stochastic-process models provably
require less memory than any classical generator. theta_zz = theta_x =
pi/4 is the maximally chaotic self-dual point; small theta_x is
near-classical.

``markov_control_sequences`` is the mandatory twin: an order-k Markov
chain ESTIMATED FROM the quantum corpus, then sampled fresh — identical
k-gram statistics, all longer-range/quantum correlations destroyed. Any
quantum-model advantage must survive this control to mean anything.

Pure NumPy statevector simulation (tiny systems, exact, deterministic).
"""
from __future__ import annotations

import numpy as np


class IdentityTokenizer:
    """Tokenizer-shaped adapter for already-tokenized integer sequences."""

    def __init__(self, vocab_size: int):
        self._vocab = vocab_size
        self.stoi: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return self._vocab

    def encode(self, ids) -> np.ndarray:
        return np.asarray(ids, dtype=np.int32)

    def decode(self, ids) -> str:
        return " ".join(str(int(i)) for i in ids)


def _kicked_ising_unitary(
    n_qubits: int, theta_zz: float, theta_x: float
) -> np.ndarray:
    """Dense Floquet unitary U = U_x . U_zz (ring ZZ coupling).

    Bit convention: qubit q is bit (n-1-q) of the basis index (qubit 0 =
    most significant), matching the kron order of U_x.
    """
    dim = 2**n_qubits
    basis = np.arange(dim)
    bits = (basis[:, None] >> (n_qubits - 1 - np.arange(n_qubits))[None, :]) & 1
    z = 1 - 2 * bits  # (+1 for |0>, -1 for |1>)
    zz = (z * np.roll(z, -1, axis=1)).sum(axis=1)
    u_zz_diag = np.exp(-1j * theta_zz * zz)

    rx = np.array(
        [
            [np.cos(theta_x), -1j * np.sin(theta_x)],
            [-1j * np.sin(theta_x), np.cos(theta_x)],
        ]
    )
    u_x = np.array([[1.0 + 0.0j]])
    for _ in range(n_qubits):
        u_x = np.kron(u_x, rx)

    return u_x * u_zz_diag[None, :]


def _floquet_apply(
    psi: np.ndarray, n_qubits: int, theta_zz: float, theta_x: float,
    n_periods: int,
) -> np.ndarray:
    """Apply U_F**n_periods = (U_x . U_zz)**n_periods gate-wise.

    Memory/compute O(2**n) per op instead of the dense O(4**n) unitary —
    required beyond ~10 qubits (13 qubits dense = 1 GB and ~1e13 FLOPs).
    Identical math: diagonal ZZ phase, then RX(2*theta_x convention as in
    `_kicked_ising_unitary`) per qubit via tensor reshape.
    """
    n_seq = psi.shape[0]
    basis = np.arange(2**n_qubits)
    bits = (basis[:, None] >> (n_qubits - 1 - np.arange(n_qubits))[None, :]) & 1
    z = 1 - 2 * bits
    zz_diag = np.exp(
        -1j * theta_zz * (z * np.roll(z, -1, axis=1)).sum(axis=1)
    )
    rx = np.array(
        [
            [np.cos(theta_x), -1j * np.sin(theta_x)],
            [-1j * np.sin(theta_x), np.cos(theta_x)],
        ]
    )
    for _ in range(n_periods):
        psi = psi * zz_diag[None, :]
        for q in range(n_qubits):
            d_pre, d_post = 2**q, 2 ** (n_qubits - 1 - q)
            t = psi.reshape(n_seq, d_pre, 2, d_post)
            psi = np.einsum("ji,saib->sajb", rx, t).reshape(n_seq, -1)
    return psi


def monitored_ising_sequences(
    n_qubits: int = 6,
    n_measured: int = 2,
    n_sequences: int = 64,
    seq_len: int = 2048,
    theta_zz: float = np.pi / 4,
    theta_x: float = np.pi / 4,
    steps_per_token: int = 1,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Measurement-record token sequences from monitored kicked-Ising dynamics.

    Returns (ids, vocab_size): ids is the concatenation of ``n_sequences``
    independent trajectories of ``seq_len`` tokens; vocab = 2**n_measured.
    Each token is the joint projective-measurement outcome of qubits
    0..n_measured-1 after ``steps_per_token`` Floquet periods; those qubits
    collapse, the remaining n_qubits - n_measured qubits persist as quantum
    memory. Larger ``steps_per_token`` / smaller ``n_measured`` = weaker
    monitoring = more coherent memory between emissions.
    """
    assert 0 < n_measured < n_qubits
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    n_out = 2**n_measured
    n_rest = dim // n_out

    psi = np.zeros((n_sequences, dim), dtype=np.complex128)
    psi[:, 0] = 1.0  # |0...0>

    tokens = np.empty((n_sequences, seq_len), dtype=np.int32)
    for t in range(seq_len):
        psi = _floquet_apply(psi, n_qubits, theta_zz, theta_x, steps_per_token)
        blocks = psi.reshape(n_sequences, n_out, n_rest)
        probs = np.abs(blocks) ** 2
        p_out = probs.sum(axis=2)
        p_out /= p_out.sum(axis=1, keepdims=True)

        cdf = np.cumsum(p_out, axis=1)
        u = rng.random((n_sequences, 1))
        outcome = (u > cdf).sum(axis=1).astype(np.int32)
        outcome = np.clip(outcome, 0, n_out - 1)
        tokens[:, t] = outcome

        # collapse: keep only the measured block, renormalize
        mask = np.zeros((n_sequences, n_out, 1))
        mask[np.arange(n_sequences), outcome, 0] = 1.0
        blocks = blocks * mask
        norms = np.sqrt((np.abs(blocks) ** 2).sum(axis=(1, 2), keepdims=True))
        psi = (blocks / np.maximum(norms, 1e-300)).reshape(n_sequences, dim)

    return tokens.reshape(-1), n_out


def markov_control_sequences(
    quantum_ids: np.ndarray,
    vocab_size: int,
    order: int = 3,
    seed: int = 0,
    smoothing: float = 0.5,
) -> np.ndarray:
    """Order-k Markov twin of a token corpus: same k-gram statistics,
    longer-range (incl. quantum-memory) correlations destroyed."""
    ids = np.asarray(quantum_ids, dtype=np.int64)
    n = len(ids)
    assert n > order + 1

    # context -> next-token counts
    counts: dict[tuple, np.ndarray] = {}
    for i in range(n - order):
        ctx = tuple(ids[i : i + order])
        if ctx not in counts:
            counts[ctx] = np.zeros(vocab_size)
        counts[ctx][ids[i + order]] += 1

    rng = np.random.default_rng(seed)
    uniform = np.full(vocab_size, 1.0 / vocab_size)
    out = np.empty(n, dtype=np.int32)
    out[:order] = ids[:order]
    for i in range(order, n):
        ctx = tuple(out[i - order : i])
        c = counts.get(ctx)
        if c is None:
            p = uniform
        else:
            p = c + smoothing
            p = p / p.sum()
        out[i] = rng.choice(vocab_size, p=p)
    return out
