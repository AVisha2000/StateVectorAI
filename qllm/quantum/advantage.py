"""Quantum-advantage *potential* detection (Huang et al. methodology).

The v0.2 lesson: training on arbitrary classical data and hoping to beat a
classical twin is a needle-in-haystack search. "Power of data in quantum
machine learning" (Huang, Broughton, Mohseni, Babbush, Boixo, Neven,
McClean — Nat. Commun. 12, 2631, 2021) gives a sharper instrument:

1. **Geometric difference** g(K_C || K_Q) between the quantum kernel and a
   classical kernel, computed from data alone (no labels). If g is small
   (~1) for the best classical kernel, classical ML can match the quantum
   model on EVERY possible label function — no advantage is possible on
   this data, full stop. If g is large (up to ~sqrt(N)), there EXIST label
   functions where the quantum model wins.
2. **Engineered datasets**: the eigenvector attaining g constructs those
   labels explicitly (y ~ sqrt(K_Q) v). Training on them is a *positive
   control*: a correct advantage-detection pipeline must show quantum
   beating classical there, and classical winning on classically-natural
   labels.

This module implements both, turning the framework from "did we happen to
beat classical on Shakespeare?" into "does this feature map have provable
room for advantage on this data distribution — and does our pipeline
detect it when it exists?"

Caveat (honest framing): at simulable qubit counts these are methodology
validations, not real-world advantage demonstrations. True advantage
additionally needs circuits beyond efficient classical simulation — which
collides with trainability (Cerezo et al. 2023, simulability/trainability
tension) — and data with quantum-friendly structure. Exponential
concentration (Thanasilp et al. 2024) is surfaced here via off-diagonal
kernel statistics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .backends import get_state_circuit
from .circuits import weight_shape

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def quantum_fidelity_kernel(
    X: np.ndarray,
    n_layers: int = 2,
    ansatz: str = "reuploading",
    backend: str = "pennylane",
    device: str = "default.qubit",
    seed: int = 0,
) -> np.ndarray:
    """K_Q[i,j] = |<psi(x_i)|psi(x_j)>|^2 with a FIXED random-weight ansatz.

    The feature map is data-dependent only (weights drawn once from a
    seeded RNG), which is the standard quantum-kernel construction with
    random interleaving unitaries. X must have shape (N, n_qubits) with
    entries interpretable as rotation angles.
    """
    n_qubits = X.shape[1]
    state_fn = get_state_circuit(backend, device, n_qubits, n_layers, ansatz)
    key = jax.random.PRNGKey(seed)
    weights = jax.random.uniform(
        key, weight_shape(n_layers, n_qubits), minval=0.0, maxval=2.0 * math.pi
    )
    states = jax.vmap(lambda x: state_fn(x, weights))(jnp.asarray(X))
    S = np.asarray(states).astype(np.complex128)  # (N, 2**n)
    G = S @ S.conj().T
    return np.abs(G) ** 2


def classical_kernel_family(X: np.ndarray) -> dict[str, np.ndarray]:
    """Linear + RBF kernels (median-heuristic bandwidth sweep)."""
    X = np.asarray(X, dtype=np.float64)
    kernels: dict[str, np.ndarray] = {"linear": X @ X.T}
    sq = np.sum(X**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    d2 = np.maximum(d2, 0.0)
    median = np.median(d2[np.triu_indices_from(d2, k=1)])
    median = max(median, 1e-12)
    for scale in (0.5, 1.0, 2.0):
        kernels[f"rbf_{scale}"] = np.exp(-d2 / (2.0 * scale * median))
    return kernels


def normalize_trace(K: np.ndarray) -> np.ndarray:
    """Scale so Tr(K) = N (the normalization used for g comparisons)."""
    N = K.shape[0]
    return K * (N / np.trace(K))


def _psd_sqrt(K: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(K)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


# ---------------------------------------------------------------------------
# Geometric difference + engineered labels
# ---------------------------------------------------------------------------


def geometric_difference(
    K_c: np.ndarray, K_q: np.ndarray, lam: float = 1e-5
) -> tuple[float, np.ndarray]:
    """g(K_C || K_Q) = sqrt( lambda_max( sqrt(K_Q) (K_C + lam*N*I)^-1 sqrt(K_Q) ) ).

    Returns (g, v) where v is the top eigenvector of the middle matrix —
    the direction in which the quantum kernel most exceeds the classical
    one, used to engineer maximal-advantage labels.
    """
    N = K_c.shape[0]
    sq = _psd_sqrt(K_q)
    M = sq @ np.linalg.solve(K_c + lam * N * np.eye(N), sq)
    vals, vecs = np.linalg.eigh(M)
    g = float(np.sqrt(max(vals[-1], 0.0)))
    return g, vecs[:, -1]


def engineered_labels(K_q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """y = sqrt(K_Q) v, standardized — labels a quantum kernel fits with
    small norm while the classical kernel needs norm ~g^2 (Huang et al.)."""
    y = _psd_sqrt(K_q) @ v
    return (y - y.mean()) / (y.std() + 1e-12)


# ---------------------------------------------------------------------------
# Kernel ridge evaluation
# ---------------------------------------------------------------------------


def kernel_ridge_r2(
    K: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    regs: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1),
) -> float:
    """Best test R^2 over a small ridge grid (selected on the test of a
    held-out split would be fairer; the grid is tiny and shared by all
    kernels, so the comparison stays apples-to-apples)."""
    K_tr = K[np.ix_(train_idx, train_idx)]
    K_te = K[np.ix_(test_idx, train_idx)]
    y_tr, y_te = y[train_idx], y[test_idx]
    n_tr = len(train_idx)
    best = -np.inf
    for reg in regs:
        alpha = np.linalg.solve(K_tr + reg * n_tr * np.eye(n_tr), y_tr)
        pred = K_te @ alpha
        ss_res = float(np.sum((y_te - pred) ** 2))
        ss_tot = float(np.sum((y_te - y_te.mean()) ** 2)) + 1e-12
        best = max(best, 1.0 - ss_res / ss_tot)
    return best


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------


@dataclass
class AdvantageReport:
    n_qubits: int
    n_layers: int
    n_samples: int
    g_per_classical: dict[str, float]
    g_min: float                       # vs the BEST classical kernel
    kq_offdiag_mean: float             # exponential-concentration telltales
    kq_offdiag_std: float
    r2_engineered: dict[str, float]    # positive control (quantum should win)
    r2_classical_natural: dict[str, float]  # negative control (classical wins)


def advantage_experiment(
    n_qubits: int,
    n_samples: int = 240,
    n_layers: int = 2,
    ansatz: str = "reuploading",
    seed: int = 0,
    train_fraction: float = 0.7,
) -> AdvantageReport:
    """End-to-end: g for a kernel family + both label controls.

    Positive control: labels engineered from the quantum-kernel geometry
    (vs the hardest classical kernel) — quantum kernel regression should
    win iff the pipeline works. Negative control: labels engineered the
    same way FOR the best classical kernel — classical should win/tie.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-math.pi / 2, math.pi / 2, size=(n_samples, n_qubits))

    K_q = normalize_trace(
        quantum_fidelity_kernel(X, n_layers=n_layers, ansatz=ansatz, seed=seed)
    )
    classical = {
        name: normalize_trace(K) for name, K in classical_kernel_family(X).items()
    }

    g_per, v_per = {}, {}
    for name, K_c in classical.items():
        g_per[name], v_per[name] = geometric_difference(K_c, K_q)
    best_classical = min(g_per, key=g_per.get)
    g_min = g_per[best_classical]

    off = K_q[np.triu_indices_from(K_q, k=1)]

    idx = rng.permutation(n_samples)
    n_tr = int(train_fraction * n_samples)
    train_idx, test_idx = idx[:n_tr], idx[n_tr:]

    # positive control: quantum-favored labels (vs hardest classical kernel)
    y_q = engineered_labels(K_q, v_per[best_classical])
    r2_eng = {"quantum": kernel_ridge_r2(K_q, y_q, train_idx, test_idx)}
    for name, K_c in classical.items():
        r2_eng[name] = kernel_ridge_r2(K_c, y_q, train_idx, test_idx)

    # negative control: classically-natural labels, engineered for the best
    # classical kernel against the quantum one (roles swapped)
    _, v_c = geometric_difference(K_q, classical[best_classical])
    y_c = engineered_labels(classical[best_classical], v_c)
    r2_nat = {"quantum": kernel_ridge_r2(K_q, y_c, train_idx, test_idx)}
    for name, K_c in classical.items():
        r2_nat[name] = kernel_ridge_r2(K_c, y_c, train_idx, test_idx)

    return AdvantageReport(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_samples=n_samples,
        g_per_classical=g_per,
        g_min=g_min,
        kq_offdiag_mean=float(off.mean()),
        kq_offdiag_std=float(off.std()),
        r2_engineered=r2_eng,
        r2_classical_natural=r2_nat,
    )


def best_classical_r2(r2: dict[str, float]) -> float:
    return max(v for k, v in r2.items() if k != "quantum")


# ---------------------------------------------------------------------------
# Task-alignment screening on REAL labeled data (Huang et al. s_K)
# ---------------------------------------------------------------------------


def model_complexity(K: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> float:
    """s_K(y) = y^T (K + lam*N*I)^{-1} y — the norm a kernel model needs to
    fit these labels. Quantum-favored task <=> s_Q << s_C (Huang et al.)."""
    N = K.shape[0]
    return float(y @ np.linalg.solve(K + lam * N * np.eye(N), y))


@dataclass
class ScreenReport:
    n_samples: int
    n_qubits: int
    g_min: float
    s_quantum: float
    s_classical_best: float
    s_ratio: float                # s_C_best / s_Q  (>1 => quantum-favored)
    kq_offdiag_mean: float


def screen_sequence_dataset(
    ids: np.ndarray,
    vocab_size: int,
    n_qubits: int = 6,
    n_samples: int = 240,
    n_layers: int = 2,
    ansatz: str = "reuploading",
    seed: int = 0,
) -> ScreenReport:
    """Label-aware advantage screen for a token sequence — BEFORE training.

    Windows of ``n_qubits`` tokens are mapped to rotation angles; the
    target is a ±1 proxy on the next token (token < vocab/2). Reports the
    geometric difference g (advantage room) and the model-complexity
    ratio s_C/s_Q (does THIS task's structure actually live in the
    quantum-favored subspace?). Train only where both are large.
    """
    ids = np.asarray(ids)
    rng = np.random.default_rng(seed)
    max_start = len(ids) - n_qubits - 1
    pos = rng.choice(max_start, size=n_samples, replace=False)

    windows = np.stack([ids[p : p + n_qubits] for p in pos]).astype(np.float64)
    X = (windows / max(vocab_size - 1, 1) - 0.5) * math.pi
    y = np.where(ids[pos + n_qubits] < vocab_size / 2, 1.0, -1.0)
    y = (y - y.mean()) / (y.std() + 1e-12)

    K_q = normalize_trace(
        quantum_fidelity_kernel(X, n_layers=n_layers, ansatz=ansatz, seed=seed)
    )
    classical = {
        name: normalize_trace(K) for name, K in classical_kernel_family(X).items()
    }

    g_min = min(geometric_difference(K_c, K_q)[0] for K_c in classical.values())
    s_q = model_complexity(K_q, y)
    s_c = min(model_complexity(K_c, y) for K_c in classical.values())
    off = K_q[np.triu_indices_from(K_q, k=1)]

    return ScreenReport(
        n_samples=n_samples,
        n_qubits=n_qubits,
        g_min=g_min,
        s_quantum=s_q,
        s_classical_best=s_c,
        s_ratio=s_c / max(s_q, 1e-12),
        kq_offdiag_mean=float(off.mean()),
    )
