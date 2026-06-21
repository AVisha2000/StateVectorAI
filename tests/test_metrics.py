"""Diagnostics tests: known-value checks + sanity of the scaling harness."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from qllm.quantum.metrics import (
    average_meyer_wallach,
    expressibility_kl,
    gradient_variance_scaling_fit,
    gradient_variance,
    meyer_wallach_q,
    parameter_shift_gradient_snr,
)


def test_meyer_wallach_product_state_is_zero():
    # |00> is a product state -> Q = 0
    state = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.complex64)
    np.testing.assert_allclose(float(meyer_wallach_q(state)), 0.0, atol=1e-6)


def test_meyer_wallach_bell_state_is_one():
    bell = jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.complex64) / jnp.sqrt(2.0)
    np.testing.assert_allclose(float(meyer_wallach_q(bell)), 1.0, atol=1e-5)


def test_meyer_wallach_ghz3_is_one():
    ghz = jnp.zeros(8, dtype=jnp.complex64).at[0].set(1.0).at[7].set(1.0)
    ghz = ghz / jnp.sqrt(2.0)
    np.testing.assert_allclose(float(meyer_wallach_q(ghz)), 1.0, atol=1e-5)


def test_average_meyer_wallach_in_range():
    q = average_meyer_wallach(3, 2, n_samples=16, seed=0)
    assert 0.0 <= q <= 1.0
    assert q > 0.1  # entangling ansatz should generate entanglement


def test_expressibility_deeper_is_more_expressive():
    """More layers -> closer to Haar -> lower KL (fixed seed, 2 qubits)."""
    kl_shallow = expressibility_kl(2, 1, n_pairs=150, seed=0)
    kl_deep = expressibility_kl(2, 4, n_pairs=150, seed=0)
    assert np.isfinite(kl_shallow) and np.isfinite(kl_deep)
    assert kl_shallow > 0 and kl_deep > 0
    assert kl_deep < kl_shallow


def test_gradient_variance_keys_and_positivity():
    out = gradient_variance(3, 2, n_samples=32, seed=0)
    for key in ("grad_var_first_param", "grad_var_mean", "grad_var_max"):
        assert key in out
        assert np.isfinite(out[key])
        assert out[key] > 0


def test_parameter_shift_gradient_snr_reports_trainability_fields():
    out = parameter_shift_gradient_snr(2, 1, shots=512, seed=0)
    assert out["n_parameters"] == 2 * 1 * 3
    assert np.isfinite(out["median_snr"])
    assert np.isfinite(out["mean_snr"])
    assert 0.0 <= out["fraction_above_2se"] <= 1.0
    assert out["mean_abs_grad"] > 0.0
    assert out["median_se"] > 0.0


def test_gradient_variance_scaling_fit_detects_exponential_decay():
    rows = [
        {"n_qubits": 2, "grad_var_mean": 0.25},
        {"n_qubits": 3, "grad_var_mean": 0.125},
        {"n_qubits": 4, "grad_var_mean": 0.0625},
    ]
    fit = gradient_variance_scaling_fit(rows)
    assert fit["log_var_slope"] < 0.0
    assert 0.0 < fit["variance_decay_factor_per_qubit"] < 1.0
    assert fit["exponential_decay_detected"] is True
