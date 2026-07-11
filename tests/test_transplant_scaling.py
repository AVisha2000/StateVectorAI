"""Static-shape scaling contracts for unitary transplant compilation."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from qllm.quantum.transplant import (
    _apply,
    compile_unitary,
    dense_unitary,
)


def _reference_compile(
    target: np.ndarray,
    n_qubits: int,
    n_layers: int,
    *,
    steps: int,
    lr: float,
    seed: int,
    restarts: int,
):
    """Previous Python-step implementation, retained only as a parity oracle."""
    tgt = jnp.asarray(target.astype(np.complex64))

    def loss_fn(params):
        unitary = dense_unitary(
            params["w"], params["z"], n_qubits, params["g"]
        )
        return jnp.sum(jnp.abs(unitary - tgt) ** 2)

    tx = optax.adam(lr)

    @jax.jit
    def step(params, opt):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        params, opt = _apply(tx, params, opt, grads)
        return loss, params, opt

    best = None
    for restart in range(restarts):
        key = jax.random.PRNGKey(seed + 1000 * restart)
        k1, k2, k3 = jax.random.split(key, 3)
        params = {
            "w": jax.random.uniform(
                k1, (n_layers, n_qubits, 3), maxval=2 * math.pi
            ),
            "z": jax.random.uniform(k2, (n_layers,), maxval=2 * math.pi),
            "g": jax.random.uniform(k3, (), maxval=2 * math.pi),
        }
        opt = tx.init(params)
        for _ in range(steps):
            loss, params, opt = step(params, opt)
        candidate = (float(loss), params)
        if best is None or candidate[0] < best[0]:
            best = candidate

    loss, params = best
    error = math.sqrt(loss) / float(np.linalg.norm(target) + 1e-12)
    return (
        np.asarray(params["w"]),
        np.asarray(params["z"]),
        float(params["g"]),
        1.0 - error,
    )


def test_compile_unitary_scan_matches_previous_step_loop(monkeypatch):
    target = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    kwargs = dict(
        n_qubits=1, n_layers=2, steps=5, lr=0.03, seed=7, restarts=2
    )
    expected = _reference_compile(target, **kwargs)

    scan_lengths: list[int | None] = []
    real_scan = jax.lax.scan

    def recording_scan(function, initial, xs=None, length=None, **scan_kwargs):
        scan_lengths.append(length)
        return real_scan(function, initial, xs, length=length, **scan_kwargs)

    monkeypatch.setattr(jax.lax, "scan", recording_scan)
    actual = compile_unitary(target, **kwargs)

    assert kwargs["steps"] in scan_lengths
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-6, atol=1e-6)
    assert actual[2] == pytest.approx(expected[2], rel=1e-6, abs=1e-6)
    assert actual[3] == pytest.approx(expected[3], rel=1e-6, abs=1e-6)


def test_compile_unitary_is_deterministic():
    target = np.eye(2, dtype=np.float32)
    kwargs = dict(n_qubits=1, n_layers=1, steps=3, seed=11, restarts=2)
    first = compile_unitary(target, **kwargs)
    second = compile_unitary(target, **kwargs)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2:] == second[2:]


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("steps", 0),
        ("steps", -1),
        ("steps", 1.5),
        ("restarts", 0),
        ("restarts", -1),
        ("restarts", True),
    ],
)
def test_compile_unitary_rejects_invalid_loop_counts(argument, value):
    kwargs = dict(n_qubits=1, n_layers=1, steps=1, restarts=1)
    kwargs[argument] = value
    with pytest.raises(ValueError, match=rf"{argument} must be a positive integer"):
        compile_unitary(np.eye(2, dtype=np.float32), **kwargs)
