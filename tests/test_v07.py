"""v0.7: evaluation battery + classical->quantum weight transplant."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from qllm.config import ModelConfig, QuantumConfig
from qllm.evaluation import (
    calibration,
    conditional_entropy,
    kgram_tv_distance,
    markov_baseline_ppl,
    sample_ids,
)
from qllm.models.model import build_model
from qllm.quantum.transplant import (
    compile_unitary,
    dense_unitary,
    linearized_ffn,
    polar_compress,
)

RNG = np.random.default_rng(0)


def _tiny_model():
    cfg = ModelConfig(d_model=16, n_heads=2, n_blocks=1, d_ff=32, max_seq_len=16)
    model, _ = build_model(cfg, vocab_size=4)
    tokens = jnp.array(RNG.integers(0, 4, (2, 8)))
    params = model.init(jax.random.PRNGKey(0), tokens)["params"]
    return model, params


# ---------------------------------------------------------------------------
# Evaluation battery
# ---------------------------------------------------------------------------


def test_conditional_entropy_uniform_iid():
    ids = RNG.integers(0, 4, 8000)
    h = conditional_entropy(ids, vocab=4, k=2)
    assert abs(h - 2.0) < 0.05  # log2(4)


def test_markov_baseline_finite_and_ordered():
    ids = RNG.integers(0, 4, 6000)
    train, val = ids[:5000], ids[5000:]
    p1 = markov_baseline_ppl(train, val, vocab=4, order=1)
    assert np.isfinite(p1) and 1.0 <= p1 <= 5.0  # ~4 for iid uniform


def test_kgram_tv_identity_and_separation():
    a = RNG.integers(0, 2, 4000)
    assert kgram_tv_distance(a, a, vocab=2, k=3) < 1e-12
    b = np.zeros(4000, dtype=np.int64)
    assert kgram_tv_distance(a, b, vocab=2, k=3) > 0.5


def test_sample_ids_shapes_and_vocab():
    model, params = _tiny_model()
    out = sample_ids(
        model, params, vocab=4, prompt_ids=np.array([0, 1]),
        n_tokens=12, context_len=16, seed=0,
    )
    assert out.shape == (12,)
    assert out.min() >= 0 and out.max() < 4


def test_calibration_keys_and_ranges():
    model, params = _tiny_model()
    ids = RNG.integers(0, 4, 2000).astype(np.int32)
    rep = calibration(model, params, ids, vocab=4, batch_size=4,
                      seq_len=8, n_batches=3)
    assert set(rep) == {"ece", "nll", "accuracy"}
    assert 0.0 <= rep["ece"] <= 1.0
    assert rep["nll"] > 0


# ---------------------------------------------------------------------------
# Transplant pipeline
# ---------------------------------------------------------------------------


def test_polar_compress_properties():
    W = RNG.normal(size=(16, 16))
    t = polar_compress(W, n_qubits=3)  # core dim 8
    B, V, P = t.basis, t.target_unitary, t.positive_part
    np.testing.assert_allclose(B.T @ B, np.eye(8), atol=1e-8)
    np.testing.assert_allclose(V.T @ V, np.eye(8), atol=1e-8)
    np.testing.assert_allclose(V @ P, B.T @ W @ B, atol=1e-8)
    assert 0.0 < t.retained_energy <= 1.0 + 1e-9
    # P symmetric PSD
    np.testing.assert_allclose(P, P.T, atol=1e-8)
    assert np.linalg.eigvalsh(P).min() > -1e-8


def test_dense_unitary_is_unitary():
    w = jnp.asarray(RNG.uniform(0, 2 * np.pi, (2, 3, 3)), dtype=jnp.float32)
    z = jnp.asarray(RNG.uniform(0, 2 * np.pi, (2,)), dtype=jnp.float32)
    U = np.asarray(dense_unitary(w, z, n_qubits=3))
    np.testing.assert_allclose(U @ U.conj().T, np.eye(8), atol=1e-5)


def test_compile_unitary_fits_random_orthogonal():
    M = RNG.normal(size=(8, 8))
    Q, _ = np.linalg.qr(M)
    _, _, _, fid = compile_unitary(Q, n_qubits=3, n_layers=6, steps=600, seed=0, restarts=2)
    assert fid > 0.7, f"compile fidelity too low: {fid:.3f}"


def test_linearized_ffn_shape():
    up = RNG.normal(size=(16, 32))
    down = RNG.normal(size=(32, 16))
    W = linearized_ffn(up, down)
    assert W.shape == (16, 16)


def test_quantum_linear_ffn_in_model():
    cfg = ModelConfig(
        d_model=16, n_heads=2, n_blocks=1, d_ff=32, max_seq_len=16,
        ffn_type="quantum_linear",
        quantum=QuantumConfig(n_qubits=3, n_circuit_layers=2),
    )
    model, _ = build_model(cfg, vocab_size=5)
    tokens = jnp.array(RNG.integers(0, 5, (2, 6)))
    params = model.init(jax.random.PRNGKey(0), tokens)["params"]
    logits = model.apply({"params": params}, tokens)
    assert logits.shape == (2, 6, 5)
    assert jnp.isfinite(logits).all()

    grads = jax.grad(
        lambda p: (model.apply({"params": p}, tokens) ** 2).sum()
    )(params)
    ffn = grads["block_0"]["ffn"]
    for key in ("circuit_weights", "zz_phase"):
        assert float(jnp.abs(ffn[key]).sum()) > 0.0


def test_transplant_round_trip_realizes_target_map():
    """With near-perfect compilation, the warm-started QuantumLinearFFN must
    reproduce x @ (B C B^T) — pins the row/transpose conventions forever."""
    from qllm.quantum.transplant import compile_unitary, polar_compress

    rng = np.random.default_rng(3)
    d, n = 8, 2  # core dim 4: tiny, compiles to high fidelity fast
    W = rng.normal(size=(d, d))
    t = polar_compress(W, n_qubits=n)
    w, z, g, fid = compile_unitary(
        t.target_unitary.T, n, n_layers=6, steps=800, restarts=2, seed=0
    )
    assert fid > 0.95, f"compile too weak for round-trip test: {fid:.3f}"

    cfg = ModelConfig(
        d_model=d, n_heads=2, n_blocks=1, d_ff=16, max_seq_len=8,
        ffn_type="quantum_linear",
        quantum=QuantumConfig(n_qubits=n, n_circuit_layers=6),
    )
    model, _ = build_model(cfg, vocab_size=5)
    tokens = jnp.array(rng.integers(0, 5, (1, 4)))
    params = jax.device_get(
        model.init(jax.random.PRNGKey(0), tokens)["params"]
    )
    ffn = params["block_0"]["ffn"]
    ffn["pre_proj"]["kernel"] = np.asarray(t.basis, np.float32)
    ffn["circuit_weights"] = np.asarray(w, np.float32)
    ffn["zz_phase"] = np.asarray(z, np.float32)
    ffn["global_phase"] = np.asarray(g, np.float32)
    ffn["post_proj"]["kernel"] = np.asarray(
        (t.basis @ t.positive_part).T, np.float32
    )

    from qllm.classical.layers import LowRankFFN  # noqa: F401  (registry sanity)
    from qllm.quantum.layers import QuantumLinearFFN

    layer = QuantumLinearFFN(d_model=d, quantum=cfg.quantum)
    x = rng.normal(size=(3, 2, d)).astype(np.float32)
    out = layer.apply({"params": ffn}, jnp.asarray(x))
    C = t.target_unitary @ t.positive_part
    expected = x @ (t.basis @ C @ t.basis.T).astype(np.float32)
    err = float(np.abs(np.asarray(out) - expected).max())
    scale = float(np.abs(expected).max()) + 1e-9
    assert err / scale < 0.15, f"round-trip mismatch: rel {err/scale:.3f}"


def test_lowrank_ffn_in_model():
    cfg = ModelConfig(d_model=16, n_heads=2, n_blocks=1, d_ff=32,
                      max_seq_len=16, ffn_type="lowrank", ffn_rank=4)
    model, _ = build_model(cfg, vocab_size=5)
    tokens = jnp.array(RNG.integers(0, 5, (2, 6)))
    params = model.init(jax.random.PRNGKey(0), tokens)["params"]
    logits = model.apply({"params": params}, tokens)
    assert logits.shape == (2, 6, 5)
    assert jnp.isfinite(logits).all()
