"""Backend capability metadata and exact-overlap contracts."""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qllm.config import ExperimentConfig, ModelConfig, QuantumConfig, validate_config
from qllm.quantum.backends import (
    PennyLaneBackend,
    TensorCircuitMPSBackend,
    get_expval_circuit,
    get_state_circuit,
    make_backend,
)
from qllm.quantum.capabilities import (
    CAPABILITY_NAMES,
    backend_capabilities_payload,
    resolve_backend_capabilities,
)
from qllm.quantum.circuits import weight_shape
from qllm.tracking import log_quantum_diagnostics


def _mps_quantum_config(**changes) -> QuantumConfig:
    config = QuantumConfig(
        n_qubits=4,
        n_circuit_layers=2,
        backend="tensorcircuit_mps",
        device="mps",
        diff_method="backprop",
        shots=None,
        mps_max_bond_dimension=4,
    )
    return dataclasses.replace(config, **changes)


@pytest.mark.parametrize(
    ("backend", "device"),
    [
        ("pennylane", "default.qubit"),
        ("tensorcircuit", "statevector"),
        ("jax_native_statevector", "jax:cpu"),
    ],
)
def test_capability_payload_is_complete_and_json_serializable(backend, device):
    payload = backend_capabilities_payload(backend, device)

    assert payload["schema_version"] == 1
    assert tuple(payload["capabilities"]) == CAPABILITY_NAMES
    assert payload["execution_regime"] == "exact_analytic"
    assert payload["exactness"] == "exact"
    assert payload["representation"] == "dense_statevector"
    assert payload["approximation"] is None
    assert json.loads(json.dumps(payload)) == payload


def test_finite_shot_metadata_is_distinct_and_has_no_state_access():
    capabilities = resolve_backend_capabilities(
        "pennylane", "default.qubit", "parameter-shift", 128
    )

    assert capabilities.execution_regime == "finite_shot_estimate"
    assert capabilities.representation == "dense_statevector"
    assert capabilities.exactness == "sampled"
    assert capabilities.approximation == {
        "method": "finite_shot_sampling",
        "shots": 128,
        "error_metric": "sampling_error",
        "convergence": "not_measured",
    }
    assert capabilities.expectations.semantics == "finite_shot_estimate"
    assert not capabilities.state_access.supported


def test_unverified_modes_are_not_reported_as_supported_or_approximate():
    unknown_device = resolve_backend_capabilities(
        "pennylane", "plugin.device", "parameter-shift", None
    )
    tensorcircuit = resolve_backend_capabilities("tensorcircuit", "statevector")

    assert unknown_device.exactness == "unverified"
    assert unknown_device.approximation is None
    assert unknown_device.gradients.status == "conditional"
    assert tensorcircuit.gradients.status == "unverified"
    assert not tensorcircuit.gradients.supported


def test_mps_capabilities_are_explicitly_approximate_and_json_safe():
    payload = backend_capabilities_payload(
        "tensorcircuit_mps",
        "mps",
        "backprop",
        None,
        8,
        None,
        False,
    )

    assert payload["execution_regime"] == "approximate_analytic"
    assert payload["exactness"] == "approximate"
    assert payload["representation"] == "matrix_product_state"
    assert payload["approximation"] == {
        "method": "mps_svd_truncation",
        "configured_truncation_mode": "fixed_bond_dimension_only",
        "configured_max_bond_dimension": 8,
        "configured_max_truncation_error": None,
        "configured_relative_truncation": False,
        "threshold_support": "unsupported_for_jit_vmap_training",
        "threshold_support_reason": (
            "TensorCircuit-NG 1.7 threshold-selected ranks are not "
            "JAX-transform safe"
        ),
        "realized_max_bond_dimension": None,
        "realized_max_bond_dimension_status": "unmeasured",
        "discarded_weight": None,
        "discarded_weight_status": "unmeasured",
        "convergence": None,
        "convergence_status": "unmeasured",
    }
    assert payload["capabilities"]["expectations"]["supported"] is True
    assert payload["capabilities"]["gradients"]["supported"] is True
    assert payload["capabilities"]["state_access"]["status"] == "unsupported"
    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.parametrize(
    "args, match",
    [
        (("pennylane", "default.qubit", "backprop", 100), "finite-shot"),
        (("pennylane", "default.qubit", "definitely-invalid", None), "Unsupported"),
        (("tensorcircuit", "statevector", "backprop", 100), "shots must be None"),
        (("tensorcircuit", "statevector", "parameter-shift", None), "only diff_method"),
        (("tensorcircuit", "default.qubit", "backprop", None), "device='statevector'"),
        (
            ("tensorcircuit_mps", "mps", "backprop", None, None, None, False),
            "positive integer",
        ),
        (
            ("tensorcircuit_mps", "statevector", "backprop", None, 4, None, False),
            "device='mps'",
        ),
        (
            ("tensorcircuit_mps", "mps", "parameter-shift", None, 4, None, False),
            "only diff_method",
        ),
        (
            ("tensorcircuit_mps", "mps", "backprop", 100, 4, None, False),
            "shots must be None",
        ),
        (
            ("tensorcircuit_mps", "mps", "backprop", None, 4, float("nan"), False),
            "finite non-negative",
        ),
        (
            ("tensorcircuit_mps", "mps", "backprop", None, 4, 1e-6, False),
            "must be None",
        ),
        (
            ("tensorcircuit_mps", "mps", "backprop", None, 4, None, True),
            "must be false",
        ),
        (
            ("tensorcircuit_mps", "mps", "backprop", None, 4, None, 1),
            "true or false",
        ),
        (("jax_native_statevector", "jax:cpu", "backprop", 10), "finite shots"),
    ],
)
def test_invalid_execution_modes_fail_without_backend_import(args, match):
    with pytest.raises(ValueError, match=match):
        resolve_backend_capabilities(*args)


def test_tensorcircuit_mode_is_rejected_before_optional_import():
    with pytest.raises(ValueError, match="only diff_method"):
        make_backend(
            "tensorcircuit",
            device="statevector",
            diff_method="parameter-shift",
        )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"mps_max_bond_dimension": None}, "must be provided"),
        ({"mps_max_bond_dimension": 0}, "positive integer"),
        ({"device": "statevector"}, "device must be 'mps'"),
        ({"diff_method": "parameter-shift"}, "diff_method must be 'backprop'"),
        ({"shots": 32}, "shots must be null"),
        ({"mps_max_truncation_error": -1e-5}, "must be at least 0"),
        ({"mps_max_truncation_error": 1e-5}, "must be null"),
        ({"mps_relative_truncation": True}, "must be false"),
        ({"mps_relative_truncation": 1}, "must be true or false"),
    ],
)
def test_mps_config_validation_fails_early(changes, match):
    cfg = ExperimentConfig(
        model=ModelConfig(quantum=_mps_quantum_config(**changes))
    )

    assert any(match in error for error in validate_config(cfg))


def test_valid_mps_config_is_dependency_free_and_accepts_static_limits():
    cfg = ExperimentConfig(
        model=ModelConfig(
            quantum=_mps_quantum_config(
                mps_max_truncation_error=None,
                mps_relative_truncation=False,
            )
        )
    )

    assert validate_config(cfg) == []


@pytest.mark.parametrize(
    "changes",
    [
        {"mps_max_bond_dimension": 4},
        {"mps_max_truncation_error": 1e-6},
        {"mps_relative_truncation": True},
    ],
)
def test_non_mps_backend_rejects_inert_mps_settings(changes):
    cfg = ExperimentConfig(
        model=ModelConfig(quantum=dataclasses.replace(QuantumConfig(), **changes))
    )

    errors = validate_config(cfg)
    assert any("supported only" in error for error in errors)


def test_mps_state_rejection_precedes_optional_import(monkeypatch):
    import qllm.quantum.backends as backend_module

    get_state_circuit.cache_clear()

    def fail_import(_name):
        pytest.fail("state rejection must not import TensorCircuit")

    monkeypatch.setattr(backend_module, "import_module", fail_import)
    with pytest.raises(ValueError, match="dense state materialization"):
        get_state_circuit(
            "tensorcircuit_mps",
            "mps",
            4,
            2,
            "reuploading",
            "backprop",
            None,
            4,
        )


def test_mps_missing_dependency_error_points_to_optional_extra(monkeypatch):
    import qllm.quantum.backends as backend_module

    def missing_dependency(name):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(backend_module, "import_module", missing_dependency)
    with pytest.raises(ImportError, match=r"qllm\[mps\]"):
        make_backend(
            "tensorcircuit_mps",
            "mps",
            "backprop",
            None,
            4,
        )


@pytest.mark.parametrize(
    "options",
    [
        {"mps_max_truncation_error": 1e-6},
        {"mps_relative_truncation": True},
    ],
)
def test_unsupported_mps_threshold_modes_fail_before_optional_import(
    monkeypatch,
    options,
):
    import qllm.quantum.backends as backend_module

    def fail_import(_name):
        pytest.fail("unsupported threshold modes must fail before import")

    monkeypatch.setattr(backend_module, "import_module", fail_import)
    with pytest.raises(ValueError, match="must be (None|false)"):
        make_backend(
            "tensorcircuit_mps",
            "mps",
            "backprop",
            None,
            4,
            **options,
        )


def test_mps_split_rules_and_ring_edge_are_passed_to_runtime(monkeypatch):
    import qllm.quantum.backends as backend_module

    observed: dict[str, object] = {"cnots": [], "measurements": []}

    class FakeMPSCircuit:
        def __init__(self, n_qubits, *, split):
            observed["n_qubits"] = n_qubits
            observed["split"] = split

        def ry(self, _wire, *, theta):
            del theta

        def rz(self, _wire, *, theta):
            del theta

        def cnot(self, control, target):
            observed["cnots"].append((control, target))

        def expectation_ps(self, *, z, normalize):
            observed["measurements"].append((tuple(z), normalize))
            return jnp.asarray(len(z), dtype=jnp.float32)

    fake_runtime = SimpleNamespace(
        MPSCircuit=FakeMPSCircuit,
        backend=SimpleNamespace(real=lambda value: value),
        set_backend=lambda name: observed.update(runtime_backend=name),
    )
    monkeypatch.setattr(
        backend_module,
        "import_module",
        lambda _name: fake_runtime,
    )
    backend = TensorCircuitMPSBackend(
        mps_max_bond_dimension=7,
    )
    circuit = backend.expval_circuit(4, 1, "hardware_efficient", "zz")
    values = circuit(jnp.zeros(4), jnp.zeros((1, 4, 3)))

    assert observed["runtime_backend"] == "jax"
    assert observed["split"] == {
        "max_singular_values": 7,
        "relative": False,
    }
    assert (3, 0) in observed["cnots"]
    assert len(observed["measurements"]) == 10
    assert all(normalize is True for _, normalize in observed["measurements"])
    np.testing.assert_array_equal(np.asarray(values[:4]), np.ones(4))
    np.testing.assert_array_equal(np.asarray(values[4:]), np.full(6, 2.0))


def test_mps_diagnostics_measure_gradients_and_expose_state_unavailability(
    monkeypatch,
):
    from qllm.quantum import metrics as qmetrics

    expected_gradients = {
        "grad_var_first_param": 0.25,
        "grad_var_mean": 0.5,
        "grad_var_max": 0.75,
    }
    monkeypatch.setattr(
        qmetrics,
        "gradient_variance",
        lambda *args, **kwargs: expected_gradients,
    )
    monkeypatch.setattr(
        qmetrics,
        "average_meyer_wallach",
        lambda *args, **kwargs: pytest.fail("MPS must not request a statevector"),
    )
    monkeypatch.setattr(
        qmetrics,
        "expressibility_kl",
        lambda *args, **kwargs: pytest.fail("MPS must not request a statevector"),
    )

    class RecordingTracker:
        def __init__(self):
            self.metrics = None

        def log_metrics(self, metrics):
            self.metrics = metrics

    tracker = RecordingTracker()
    diagnostics = log_quantum_diagnostics(
        tracker,
        _mps_quantum_config(),
        n_grad_samples=2,
        n_pairs=2,
        n_mw_samples=2,
    )

    assert diagnostics["meyer_wallach_q"] is None
    assert diagnostics["expressibility_kl"] is None
    assert diagnostics["availability"]["gradient_variance"]["status"] == "measured"
    for key in ("meyer_wallach_q", "expressibility_kl"):
        status = diagnostics["availability"][key]
        assert status["status"] == "unsupported"
        assert "dense state materialization" in status["reason"]
    assert tracker.metrics == {
        f"q_{key}": value for key, value in expected_gradients.items()
    }


def test_quantum_core_and_embedding_thread_mps_static_settings(monkeypatch):
    import qllm.quantum.layers as layer_module
    from qllm.quantum.layers import QuantumCore, QuantumEmbedding

    config = _mps_quantum_config()
    core = QuantumCore.from_config(config, out_features=3)
    assert core.mps_max_bond_dimension == 4
    assert core.mps_max_truncation_error is None
    assert core.mps_relative_truncation is False

    observed = {}

    def fake_factory(*args):
        observed["factory_args"] = args
        n_qubits = args[4]

        def circuit(_inputs, _weights):
            return jnp.ones(n_qubits, dtype=jnp.float32)

        return circuit

    monkeypatch.setattr(layer_module, "get_expval_circuit", fake_factory)
    embedding = QuantumEmbedding(vocab_size=5, d_model=3, quantum=config)
    tokens = jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32)
    variables = embedding.init(jax.random.PRNGKey(4), tokens)
    output = embedding.apply(variables, tokens)

    assert output.shape == (2, 2, 3)
    assert observed["factory_args"][-3:] == (4, None, False)


def test_nonfinite_measured_diagnostic_is_rejected_before_tracking(monkeypatch):
    from qllm.quantum import metrics as qmetrics

    monkeypatch.setattr(
        qmetrics,
        "quantum_diagnostics",
        lambda *args, **kwargs: {
            "grad_var_first_param": float("nan"),
            "grad_var_mean": None,
            "grad_var_max": None,
            "meyer_wallach_q": None,
            "expressibility_kl": None,
            "availability": {
                "gradient_variance": {
                    "status": "measured",
                    "available": True,
                    "semantics": "test",
                    "reason": None,
                },
                "meyer_wallach_q": {
                    "status": "unsupported",
                    "available": False,
                    "semantics": "unsupported",
                    "reason": "test",
                },
                "expressibility_kl": {
                    "status": "unsupported",
                    "available": False,
                    "semantics": "unsupported",
                    "reason": "test",
                },
            },
        },
    )

    class RejectingTracker:
        def log_metrics(self, _metrics):
            pytest.fail("non-finite metrics must never reach the tracker")

    with pytest.raises(ValueError, match="must be finite"):
        log_quantum_diagnostics(RejectingTracker(), _mps_quantum_config())


def test_invalid_pennylane_method_cannot_enter_resource_plan(
    tiny_quantum_cfg,
):
    import dataclasses

    import jax

    from qllm.resources import static_resource_plan

    invalid = dataclasses.replace(
        tiny_quantum_cfg,
        model=dataclasses.replace(
            tiny_quantum_cfg.model,
            quantum=dataclasses.replace(
                tiny_quantum_cfg.model.quantum,
                diff_method="definitely-invalid",
            ),
        ),
    )
    with pytest.raises(ValueError, match="Unsupported PennyLane diff_method"):
        static_resource_plan(
            invalid,
            n_params=1,
            requested_device="cpu",
            resolved_device=jax.devices("cpu")[0],
        )


def test_finite_shot_state_factory_is_rejected_explicitly():
    backend = PennyLaneBackend(diff_method="parameter-shift", shots=32)
    with pytest.raises(ValueError, match="State access is unavailable"):
        backend.state_circuit(2, 1, "hardware_efficient")

    with pytest.raises(ValueError, match="State access is unavailable"):
        get_state_circuit(
            "pennylane",
            "default.qubit",
            2,
            1,
            "hardware_efficient",
            "parameter-shift",
            32,
        )


@pytest.mark.parametrize("readout", ["z", "zz"])
def test_pennylane_exact_expvals_match_state_derived_values(readout):
    n_qubits, n_layers = 3, 2
    ansatz = "reuploading"
    key_inputs, key_weights = jax.random.split(jax.random.PRNGKey(19))
    inputs = jax.random.uniform(key_inputs, (n_qubits,), minval=-1.0, maxval=1.0)
    weights = jax.random.uniform(
        key_weights, weight_shape(n_layers, n_qubits), minval=-0.8, maxval=0.8
    )
    state = np.asarray(
        get_state_circuit(
            "pennylane", "default.qubit", n_qubits, n_layers, ansatz
        )(inputs, weights)
    )
    probabilities = np.abs(state) ** 2
    basis = np.arange(2**n_qubits)
    z_values = [
        1.0 - 2.0 * ((basis >> (n_qubits - 1 - wire)) & 1)
        for wire in range(n_qubits)
    ]
    expected = [float(probabilities @ z) for z in z_values]
    if readout == "zz":
        expected.extend(
            float(probabilities @ (z_values[i] * z_values[j]))
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        )

    circuit = get_expval_circuit(
        "pennylane",
        "default.qubit",
        "backprop",
        None,
        n_qubits,
        n_layers,
        ansatz,
        readout,
    )
    np.testing.assert_allclose(
        np.asarray(circuit(inputs, weights)), np.asarray(expected), rtol=1e-5, atol=1e-6
    )
