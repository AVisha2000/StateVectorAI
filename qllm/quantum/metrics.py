"""Quantum diagnostics for hybrid-model evaluation and scaling studies.

Three metrics from the planning doc, all backend-agnostic via the state /
expval circuit protocol:

- Meyer-Wallach entangling capability Q (Sim et al. 2019): cheap, scalable.
- Expressibility KL vs the Haar fidelity distribution (Sim et al. 2019).
  NB: high expressibility correlates with barren plateaus (Holmes 2022) —
  this is a characterization metric, not a score to maximize.
- Gradient variance Var[dC/d theta] vs qubit count: THE barren-plateau
  signature. Exponential decay (straight line on log-y vs n) means the
  component is untrainable at scale.
"""
from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .backends import get_expval_circuit, get_state_circuit
from .capabilities import Capability, resolve_backend_capabilities
from .circuits import weight_shape


GRADIENT_METRIC_KEYS = (
    "grad_var_first_param",
    "grad_var_mean",
    "grad_var_max",
)


# ---------------------------------------------------------------------------
# Meyer-Wallach entangling capability
# ---------------------------------------------------------------------------

def meyer_wallach_q(state: jnp.ndarray) -> jnp.ndarray:
    """Meyer-Wallach measure Q in [0, 1] for a pure statevector.

    Q = 2 * (1 - (1/n) * sum_k Tr(rho_k^2)), rho_k the single-qubit reduced
    density matrix. Q = 0 for product states, Q = 1 for e.g. GHZ states.
    """
    dim = state.shape[-1]
    n_qubits = int(math.log2(dim))
    psi = state.reshape((2,) * n_qubits)

    purities = []
    for k in range(n_qubits):
        a = jnp.moveaxis(psi, k, 0).reshape(2, -1)
        rho = a @ a.conj().T
        purities.append(jnp.real(jnp.trace(rho @ rho)))
    avg_purity = jnp.mean(jnp.stack(purities))
    return 2.0 * (1.0 - avg_purity)


def average_meyer_wallach(
    n_qubits: int,
    n_layers: int,
    ansatz: str = "reuploading",
    backend: str = "pennylane",
    device: str = "default.qubit",
    n_samples: int = 64,
    seed: int = 0,
    *,
    diff_method: str = "backprop",
    shots: int | None = None,
    mps_max_bond_dimension: int | None = None,
    mps_max_truncation_error: float | None = None,
    mps_relative_truncation: bool = False,
) -> float:
    """Mean Q over circuits with uniform-random parameters (zero data input)."""
    state_fn = get_state_circuit(
        backend,
        device,
        n_qubits,
        n_layers,
        ansatz,
        diff_method,
        shots,
        mps_max_bond_dimension,
        mps_max_truncation_error,
        mps_relative_truncation,
    )
    key = jax.random.PRNGKey(seed)
    shape = weight_shape(n_layers, n_qubits)
    weights = jax.random.uniform(
        key, (n_samples, *shape), minval=0.0, maxval=2.0 * math.pi
    )
    inputs = jnp.zeros(n_qubits)
    states = jax.vmap(lambda w: state_fn(inputs, w))(weights)
    qs = jax.vmap(meyer_wallach_q)(states)
    return float(jnp.mean(qs))


# ---------------------------------------------------------------------------
# Expressibility (KL divergence vs Haar fidelity distribution)
# ---------------------------------------------------------------------------

def expressibility_kl(
    n_qubits: int,
    n_layers: int,
    ansatz: str = "reuploading",
    backend: str = "pennylane",
    device: str = "default.qubit",
    n_pairs: int = 300,
    n_bins: int = 75,
    seed: int = 0,
    *,
    diff_method: str = "backprop",
    shots: int | None = None,
    mps_max_bond_dimension: int | None = None,
    mps_max_truncation_error: float | None = None,
    mps_relative_truncation: bool = False,
) -> float:
    """KL( empirical fidelity histogram || Haar ). Lower = more expressive.

    Haar bin mass uses the analytic CDF P(F <= f) = 1 - (1 - f)^(N - 1),
    N = 2**n_qubits.
    """
    state_fn = get_state_circuit(
        backend,
        device,
        n_qubits,
        n_layers,
        ansatz,
        diff_method,
        shots,
        mps_max_bond_dimension,
        mps_max_truncation_error,
        mps_relative_truncation,
    )
    key = jax.random.PRNGKey(seed)
    shape = weight_shape(n_layers, n_qubits)
    w1, w2 = jax.random.uniform(
        key, (2, n_pairs, *shape), minval=0.0, maxval=2.0 * math.pi
    )
    inputs = jnp.zeros(n_qubits)

    def fidelity(wa, wb):
        sa = state_fn(inputs, wa)
        sb = state_fn(inputs, wb)
        return jnp.abs(jnp.vdot(sa, sb)) ** 2

    fids = np.asarray(jax.vmap(fidelity)(w1, w2))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    emp, _ = np.histogram(fids, bins=edges)
    p_emp = emp / emp.sum()

    big_n = 2**n_qubits
    cdf = 1.0 - (1.0 - edges) ** (big_n - 1)
    p_haar = np.diff(cdf)
    p_haar = np.clip(p_haar, 1e-12, None)

    mask = p_emp > 0
    return float(np.sum(p_emp[mask] * np.log(p_emp[mask] / p_haar[mask])))


# ---------------------------------------------------------------------------
# Gradient variance (barren-plateau probe)
# ---------------------------------------------------------------------------

def gradient_variance(
    n_qubits: int,
    n_layers: int,
    ansatz: str = "reuploading",
    backend: str = "pennylane",
    device: str = "default.qubit",
    n_samples: int = 100,
    seed: int = 0,
    *,
    diff_method: str = "backprop",
    shots: int | None = None,
    mps_max_bond_dimension: int | None = None,
    mps_max_truncation_error: float | None = None,
    mps_relative_truncation: bool = False,
) -> dict[str, float]:
    """Variance of the cost gradient over random parameter initializations.

    Cost is the local observable <Z_0> (local costs are the trainable
    regime; global costs plateau even faster). Returns variance of the
    first partial derivative (the McClean et al. convention) and the mean
    variance over all parameters.
    """
    circuit = get_expval_circuit(
        backend,
        device,
        diff_method,
        shots,
        n_qubits,
        n_layers,
        ansatz,
        "z",
        mps_max_bond_dimension,
        mps_max_truncation_error,
        mps_relative_truncation,
    )
    inputs = jnp.zeros(n_qubits)

    def cost(w):
        return circuit(inputs, w)[0]  # <Z_0>

    key = jax.random.PRNGKey(seed)
    shape = weight_shape(n_layers, n_qubits)
    weights = jax.random.uniform(
        key, (n_samples, *shape), minval=0.0, maxval=2.0 * math.pi
    )
    grads = jax.vmap(jax.grad(cost))(weights)
    grads = grads.reshape(n_samples, -1)

    var_per_param = jnp.var(grads, axis=0)
    return {
        "grad_var_first_param": float(var_per_param[0]),
        "grad_var_mean": float(jnp.mean(var_per_param)),
        "grad_var_max": float(jnp.max(var_per_param)),
    }


def _diagnostic_status(
    capability: Capability,
    *,
    measured: bool,
) -> dict[str, Any]:
    """Describe whether one diagnostic was measured and why."""
    return {
        "status": "measured" if measured else capability.status,
        "available": measured,
        "semantics": capability.semantics,
        "reason": None if measured else capability.limitation,
    }


def quantum_diagnostics(
    n_qubits: int,
    n_layers: int,
    ansatz: str = "reuploading",
    backend: str = "pennylane",
    device: str = "default.qubit",
    n_grad_samples: int = 64,
    n_pairs: int = 200,
    n_mw_samples: int = 32,
    seed: int = 0,
    *,
    diff_method: str = "backprop",
    shots: int | None = None,
    mps_max_bond_dimension: int | None = None,
    mps_max_truncation_error: float | None = None,
    mps_relative_truncation: bool = False,
) -> dict[str, Any]:
    """Collect only diagnostics supported by the configured backend mode.

    Flat metric keys are retained for existing consumers. Unsupported metrics
    are explicit ``None`` values, while ``availability`` records the backend
    contract and reason instead of silently falling back to a dense simulator.
    """
    capabilities = resolve_backend_capabilities(
        backend,
        device,
        diff_method,
        shots,
        mps_max_bond_dimension,
        mps_max_truncation_error,
        mps_relative_truncation,
    )
    diagnostics: dict[str, Any] = {
        key: None for key in GRADIENT_METRIC_KEYS
    }
    gradient_measured = capabilities.gradients.supported
    if gradient_measured:
        diagnostics.update(
            gradient_variance(
                n_qubits,
                n_layers,
                ansatz=ansatz,
                backend=backend,
                device=device,
                n_samples=n_grad_samples,
                seed=seed,
                diff_method=diff_method,
                shots=shots,
                mps_max_bond_dimension=mps_max_bond_dimension,
                mps_max_truncation_error=mps_max_truncation_error,
                mps_relative_truncation=mps_relative_truncation,
            )
        )

    state_measured = capabilities.state_access.supported
    if state_measured:
        diagnostics["meyer_wallach_q"] = average_meyer_wallach(
            n_qubits,
            n_layers,
            ansatz=ansatz,
            backend=backend,
            device=device,
            n_samples=n_mw_samples,
            seed=seed,
            diff_method=diff_method,
            shots=shots,
            mps_max_bond_dimension=mps_max_bond_dimension,
            mps_max_truncation_error=mps_max_truncation_error,
            mps_relative_truncation=mps_relative_truncation,
        )
        diagnostics["expressibility_kl"] = expressibility_kl(
            n_qubits,
            n_layers,
            ansatz=ansatz,
            backend=backend,
            device=device,
            n_pairs=n_pairs,
            seed=seed,
            diff_method=diff_method,
            shots=shots,
            mps_max_bond_dimension=mps_max_bond_dimension,
            mps_max_truncation_error=mps_max_truncation_error,
            mps_relative_truncation=mps_relative_truncation,
        )
    else:
        diagnostics["meyer_wallach_q"] = None
        diagnostics["expressibility_kl"] = None

    diagnostics["availability"] = {
        "gradient_variance": _diagnostic_status(
            capabilities.gradients,
            measured=gradient_measured,
        ),
        "meyer_wallach_q": _diagnostic_status(
            capabilities.state_access,
            measured=state_measured,
        ),
        "expressibility_kl": _diagnostic_status(
            capabilities.state_access,
            measured=state_measured,
        ),
    }
    return diagnostics


def parameter_shift_gradient_snr(
    n_qubits: int,
    n_layers: int,
    ansatz: str = "reuploading",
    backend: str = "pennylane",
    device: str = "default.qubit",
    shots: int = 1024,
    seed: int = 0,
) -> dict[str, float]:
    """Gradient signal-to-shot-noise estimate at one random initialization.

    Uses analytic expvals for the shifted circuits, then estimates the
    standard error that finite-shot Pauli-Z measurements would have had. This
    gives a cheap hardware-realism gate without making the test stochastic.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    circuit = get_expval_circuit(
        backend, device, "backprop", None, n_qubits, n_layers, ansatz
    )
    key = jax.random.PRNGKey(seed)
    kx, kw = jax.random.split(key)
    inputs = jax.random.uniform(kx, (n_qubits,), minval=-1.0, maxval=1.0)
    weights = jax.random.uniform(
        kw, weight_shape(n_layers, n_qubits), minval=0.0, maxval=2.0 * math.pi
    )

    flat = np.asarray(weights).reshape(-1)
    grads, ses = [], []
    for idx in range(len(flat)):
        bump = np.zeros_like(flat)
        bump[idx] = math.pi / 2.0
        wp = jnp.asarray((flat + bump).reshape(weights.shape))
        wm = jnp.asarray((flat - bump).reshape(weights.shape))
        fp = float(circuit(inputs, wp)[0])
        fm = float(circuit(inputs, wm)[0])
        grad = 0.5 * (fp - fm)
        se_p = math.sqrt(max(1.0 - fp * fp, 0.0) / shots)
        se_m = math.sqrt(max(1.0 - fm * fm, 0.0) / shots)
        se = 0.5 * math.sqrt(se_p * se_p + se_m * se_m)
        grads.append(grad)
        ses.append(se)

    grads_arr = np.asarray(grads)
    ses_arr = np.asarray(ses)
    snr = np.abs(grads_arr) / np.maximum(ses_arr, 1e-12)
    return {
        "median_snr": float(np.median(snr)),
        "mean_snr": float(np.mean(snr)),
        "fraction_above_2se": float(np.mean(snr > 2.0)),
        "mean_abs_grad": float(np.mean(np.abs(grads_arr))),
        "median_se": float(np.median(ses_arr)),
        "n_parameters": int(len(flat)),
    }


def gradient_variance_scaling_fit(rows: list[dict[str, float]]) -> dict[str, float]:
    """Fit log Var[grad] against qubit count for plateau diagnostics."""
    if len(rows) < 2:
        raise ValueError("at least two rows are required for a scaling fit")
    ns = np.asarray([r["n_qubits"] for r in rows], dtype=np.float64)
    var = np.asarray([r["grad_var_mean"] for r in rows], dtype=np.float64)
    if np.any(var <= 0) or not np.isfinite(var).all():
        raise ValueError("gradient variances must be positive and finite")
    slope, intercept = np.polyfit(ns, np.log(var), 1)
    decay = float(np.exp(slope))
    return {
        "log_var_slope": float(slope),
        "log_var_intercept": float(intercept),
        "variance_decay_factor_per_qubit": decay,
        "exponential_decay_detected": bool(decay < 1.0),
    }


def scaling_sweep(
    qubit_counts: list[int],
    n_layers: int = 2,
    ansatz: str = "reuploading",
    backend: str = "pennylane",
    device: str = "default.qubit",
    n_grad_samples: int = 64,
    n_pairs: int = 200,
    n_mw_samples: int = 32,
    seed: int = 0,
    *,
    diff_method: str = "backprop",
    shots: int | None = None,
    mps_max_bond_dimension: int | None = None,
    mps_max_truncation_error: float | None = None,
    mps_relative_truncation: bool = False,
) -> list[dict[str, Any]]:
    """The Phase-3 harness: all diagnostics across a qubit-count sweep."""
    rows = []
    for n in qubit_counts:
        row: dict[str, Any] = {"n_qubits": n, "n_layers": n_layers}
        row.update(
            quantum_diagnostics(
                n,
                n_layers,
                ansatz=ansatz,
                backend=backend,
                device=device,
                n_grad_samples=n_grad_samples,
                n_pairs=n_pairs,
                n_mw_samples=n_mw_samples,
                seed=seed,
                diff_method=diff_method,
                shots=shots,
                mps_max_bond_dimension=mps_max_bond_dimension,
                mps_max_truncation_error=mps_max_truncation_error,
                mps_relative_truncation=mps_relative_truncation,
            )
        )
        rows.append(row)
    return rows
