"""Classical-weight -> quantum-circuit transplant ("load weights into circuits").

The idea: take a TRAINED classical model's FFN and carry its structure
into a quantum layer instead of starting the circuit from random. The
honest cost is contraction — a d_model x d_ff weight block cannot fit in
an n-qubit unitary, so it must be compressed to a 2**n-dimensional core
(this is why the approach is SLM-scale only). The pipeline:

1. **Linearize** the FFN around 0: W_lin ~= gelu'(0) * W_down @ W_up
   (d x d). The GELU is dropped — measured, not hidden, as part of the
   contraction loss.
2. **Contract**: SVD of W_lin; project onto the top-d_q (=2**n) left-
   singular subspace B: core C = B^T W_lin B, with retained Frobenius
   energy reported.
3. **Polar-decompose** C = V @ P: V orthogonal (the rotation), P
   symmetric positive (the stretch). The circuit gets V; P folds into
   the classical post-projection.
4. **Compile** V into the parameterized gate set by gradient descent on
   ||U(theta) - V||_F (depth-vs-fidelity curve = how expensive the
   classical rotation is in circuit depth).
5. **Surgery**: build a hybrid model whose FFN is ``QuantumLinearFFN``
   (amplitude-space unitary between Dense maps), initialize pre <- B,
   circuit <- compiled theta, post <- (B @ P), copy every shared
   parameter from the donor, then fine-tune. Compare against the same
   architecture cold-started from random.

This realizes (a small, exact-simulation version of) the
weights-as-tensor-networks/unitaries line of QLLM compression.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .recurrent import _apply_1q, _apply_cnot, _rot

# ---------------------------------------------------------------------------
# Dense unitary from our gate set (Rot layer + ZZ-ring phase + CNOT ring)
# ---------------------------------------------------------------------------


def _zz_vec(n_qubits: int) -> jnp.ndarray:
    basis = np.arange(2**n_qubits)
    bits = (basis[:, None] >> (n_qubits - 1 - np.arange(n_qubits))[None, :]) & 1
    z = 1 - 2 * bits
    return jnp.asarray(
        (z * np.roll(z, -1, axis=1)).sum(axis=1).astype(np.float32)
    )


def apply_circuit(
    psi: jnp.ndarray,
    circuit_weights: jnp.ndarray,  # (L, n, 3)
    zz_phase: jnp.ndarray,         # (L,)
    n_qubits: int,
    global_phase: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """Apply the parameterized unitary to a batch of states (B, 2**n).

    ``global_phase`` matters because downstream layers read Re(U x): a
    target V is only reproducible as e^{i g} U(theta), so g must be an
    explicit (cheap, trainable) parameter rather than an unpunishable
    gauge.
    """
    psi = psi * jnp.exp(1j * jnp.asarray(global_phase, dtype=jnp.complex64))
    zz = _zz_vec(n_qubits)

    def layer_fn(carry, wz):
        w_l, z_l = wz
        carry = carry * jnp.exp(-1j * z_l * zz)[None, :]
        for q in range(n_qubits):
            carry = _apply_1q(carry, _rot(*w_l[q]), q, n_qubits)
        if n_qubits > 1:
            for q in range(n_qubits):
                carry = _apply_cnot(carry, q, (q + 1) % n_qubits, n_qubits)
        return carry, None

    psi, _ = jax.lax.scan(layer_fn, psi, (circuit_weights, zz_phase))
    return psi


def dense_unitary(
    circuit_weights: jnp.ndarray,
    zz_phase: jnp.ndarray,
    n_qubits: int,
    global_phase: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """Materialize U(theta) as a (2**n, 2**n) matrix (columns = basis images)."""
    dim = 2**n_qubits
    eye = jnp.eye(dim, dtype=jnp.complex64)
    return apply_circuit(eye, circuit_weights, zz_phase, n_qubits, global_phase).T


# ---------------------------------------------------------------------------
# Contraction + polar decomposition
# ---------------------------------------------------------------------------


@dataclass
class Transplant:
    basis: np.ndarray            # (d_model, 2**n) orthonormal columns
    target_unitary: np.ndarray   # (2**n, 2**n) orthogonal V
    positive_part: np.ndarray    # (2**n, 2**n) symmetric PSD P
    retained_energy: float       # ||B B^T W B B^T||_F / ||W||_F
    compiled_weights: np.ndarray | None = None
    compiled_zz: np.ndarray | None = None
    compiled_phase: float | None = None
    compile_fidelity: float | None = None


def polar_compress(W_lin: np.ndarray, n_qubits: int) -> Transplant:
    """Contract a (d, d) linear map to a 2**n core and polar-decompose it."""
    d_q = 2**n_qubits
    assert W_lin.shape[0] == W_lin.shape[1] >= d_q
    U, _, _ = np.linalg.svd(W_lin)
    B = U[:, :d_q]                       # top left-singular subspace
    C = B.T @ W_lin @ B                  # contracted core (general d_q x d_q)
    # polar: C = V P with V orthogonal, P = sqrt(C^T C)
    Uc, Sc, Vct = np.linalg.svd(C)
    V = Uc @ Vct
    P = Vct.T @ np.diag(Sc) @ Vct
    retained = float(
        np.linalg.norm(B @ C @ B.T) / (np.linalg.norm(W_lin) + 1e-12)
    )
    return Transplant(
        basis=B, target_unitary=V, positive_part=P, retained_energy=retained
    )


def linearized_ffn(up_kernel: np.ndarray, down_kernel: np.ndarray) -> np.ndarray:
    """First-order FFN map around 0: gelu'(0)=0.5."""
    return 0.5 * (up_kernel @ down_kernel)  # (d_model, d_model)


# ---------------------------------------------------------------------------
# Unitary compilation
# ---------------------------------------------------------------------------


def compile_unitary(
    target: np.ndarray,
    n_qubits: int,
    n_layers: int,
    steps: int = 600,
    lr: float = 0.05,
    seed: int = 0,
    restarts: int = 3,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit e^{ig} U(theta) to the target by Frobenius loss, multi-restart.

    Returns (circuit_weights, zz_phase, global_phase, fidelity) with
    fidelity = 1 - ||e^{ig}U - V||_F / ||V||_F.
    """
    if (
        isinstance(steps, bool)
        or not isinstance(steps, (int, np.integer))
        or steps <= 0
    ):
        raise ValueError("steps must be a positive integer")
    if (
        isinstance(restarts, bool)
        or not isinstance(restarts, (int, np.integer))
        or restarts <= 0
    ):
        raise ValueError("restarts must be a positive integer")

    tgt = jnp.asarray(target.astype(np.complex64))

    def loss_fn(params):
        U = dense_unitary(params["w"], params["z"], n_qubits, params["g"])
        return jnp.sum(jnp.abs(U - tgt) ** 2)

    tx = optax.adam(lr)
    def optimize(params, opt):
        def scan_step(carry, _):
            current_params, current_opt = carry
            loss, grads = jax.value_and_grad(loss_fn)(current_params)
            next_params, next_opt = _apply(
                tx, current_params, current_opt, grads
            )
            return (next_params, next_opt), loss

        (params, opt), losses = jax.lax.scan(
            scan_step, (params, opt), xs=None, length=steps
        )
        return params, opt, losses[-1]

    optimize = jax.jit(optimize)

    best = None
    for r in range(restarts):
        key = jax.random.PRNGKey(seed + 1000 * r)
        k1, k2, k3 = jax.random.split(key, 3)
        params = {
            "w": jax.random.uniform(
                k1, (n_layers, n_qubits, 3), maxval=2 * math.pi
            ),
            "z": jax.random.uniform(k2, (n_layers,), maxval=2 * math.pi),
            "g": jax.random.uniform(k3, (), maxval=2 * math.pi),
        }
        opt = tx.init(params)
        params, opt, loss = optimize(params, opt)
        if best is None or float(loss) < best[0]:
            best = (float(loss), params)

    loss, params = best
    err = math.sqrt(loss) / float(np.linalg.norm(target) + 1e-12)
    return (
        np.asarray(params["w"]),
        np.asarray(params["z"]),
        float(params["g"]),
        1.0 - err,
    )


def _apply(tx, params, opt, grads):
    updates, opt = tx.update(grads, opt, params)
    return optax.apply_updates(params, updates), opt


def transplant_from_donor(
    donor_block_ffn: dict, n_qubits: int, n_layers: int, compile_steps: int = 400
) -> Transplant:
    """Full pipeline for one classical FFN parameter subtree."""
    up = np.asarray(donor_block_ffn["up_proj"]["kernel"])
    down = np.asarray(donor_block_ffn["down_proj"]["kernel"])
    t = polar_compress(linearized_ffn(up, down), n_qubits)
    # Row convention: activations are ROW vectors, so apply_circuit maps
    # row p -> row p @ U^T. The warm layer computes x B [U^T] P B^T and the
    # target is x B (V P) B^T, hence compile U = V^T (not V).
    w, z, g, fid = compile_unitary(
        t.target_unitary.T, n_qubits, n_layers, steps=compile_steps
    )
    return dataclasses.replace(
        t, compiled_weights=w, compiled_zz=z, compiled_phase=g,
        compile_fidelity=fid,
    )
