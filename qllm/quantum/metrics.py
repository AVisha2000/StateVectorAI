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

import jax
import jax.numpy as jnp
import numpy as np

from .backends import get_expval_circuit, get_state_circuit
from .circuits import weight_shape


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
) -> float:
    """Mean Q over circuits with uniform-random parameters (zero data input)."""
    state_fn = get_state_circuit(backend, device, n_qubits, n_layers, ansatz)
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
) -> float:
    """KL( empirical fidelity histogram || Haar ). Lower = more expressive.

    Haar bin mass uses the analytic CDF P(F <= f) = 1 - (1 - f)^(N - 1),
    N = 2**n_qubits.
    """
    state_fn = get_state_circuit(backend, device, n_qubits, n_layers, ansatz)
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
) -> dict[str, float]:
    """Variance of the cost gradient over random parameter initializations.

    Cost is the local observable <Z_0> (local costs are the trainable
    regime; global costs plateau even faster). Returns variance of the
    first partial derivative (the McClean et al. convention) and the mean
    variance over all parameters.
    """
    circuit = get_expval_circuit(
        backend, device, "backprop", None, n_qubits, n_layers, ansatz
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
) -> list[dict[str, float]]:
    """The Phase-3 harness: all diagnostics across a qubit-count sweep."""
    rows = []
    for n in qubit_counts:
        row: dict[str, float] = {"n_qubits": n, "n_layers": n_layers}
        row.update(
            gradient_variance(
                n, n_layers, ansatz, backend, device, n_grad_samples, seed
            )
        )
        row["meyer_wallach_q"] = average_meyer_wallach(
            n, n_layers, ansatz, backend, device, n_mw_samples, seed
        )
        row["expressibility_kl"] = expressibility_kl(
            n, n_layers, ansatz, backend, device, n_pairs, seed=seed
        )
        rows.append(row)
    return rows
