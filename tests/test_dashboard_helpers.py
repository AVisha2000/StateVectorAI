"""Regression contracts for shared dashboard payload helpers."""
from __future__ import annotations

import pytest

from qllm.dashboard import diagnostics, explore, gpu_reservation, lab, model_tests, queries, workspace
from qllm.resultsdb import ResultsDB


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"config": {"decoded": True}, "config_json": '{"ignored": true}'}, {"decoded": True}),
        ({"config_json": "not json"}, {}),
        ({"config_json": "[]"}, {}),
        (None, {}),
    ],
)
def test_config_decoding_is_identical_across_dashboard_views(row, expected):
    decoders = (
        model_tests._decode_config,
        gpu_reservation._decode_config,
        explore._decode_config,
        workspace._decode_config,
        lab._decode_config,
    )
    for decode in decoders:
        assert decode(row) == expected
        assert isinstance(decode(row), dict)


def test_enrich_job_uses_shared_non_object_config_fallback(tmp_path):
    db = ResultsDB(tmp_path / "helpers.db")
    job_id = db.create_lab_job(
        {
            "preset_id": "classical-small",
            "dataset_name": "default-text",
            "run_name": "invalid-config",
            "seed": 0,
            "steps": 1,
            "eval_every": 1,
            "config": {},
        }
    )
    db.update_lab_job(job_id, config_json="[]")
    enriched = lab.enrich_job(db.get_lab_job(job_id), db)
    assert enriched["config"] == {}


class _StepsDB:
    def __init__(self):
        self.calls = []

    def fetch_steps(self, run_key, *, run_uuid=None):
        self.calls.append((run_key, run_uuid))
        return [
            {"name": "loss", "step": 1, "value": 0.9},
            {"name": "accuracy", "step": 1, "value": 0.1},
            {"name": "loss", "step": 2, "value": 0.8},
        ]


def test_curve_is_identical_across_views_and_forwards_uuid():
    expected = {
        "loss": [{"step": 1, "value": 0.9}, {"step": 2, "value": 0.8}],
        "accuracy": [{"step": 1, "value": 0.1}],
    }
    for build_curve in (queries._curve, workspace._curve):
        db = _StepsDB()
        curve = build_curve(db, "suite/variant/data/0/2", "run-uuid")
        assert curve == expected
        assert type(curve) is dict
        assert db.calls == [("suite/variant/data/0/2", "run-uuid")]
        assert build_curve(db, None) == {}


@pytest.mark.parametrize(
    "job, expected",
    [
        (
            {"preset_id": "base", "run_key": "lab/from-key/data/0/2", "config": {"lab.quantum_override.n_qubits": 9, "lab.quantum_override.n_circuit_layers": 9}},
            "from-key",
        ),
        (
            {"preset_id": "base", "config_json": '{"lab.quantum_override.n_qubits": "3", "lab.quantum_override.n_circuit_layers": "2"}'},
            "base-q3-d2",
        ),
        (
            {"preset_id": "base", "config_json": '{"lab.quantum_override.n_qubits": "invalid", "lab.quantum_override.n_circuit_layers": 2}'},
            "base",
        ),
        (
            {"preset_id": "base", "config": {"lab.study_cell.n_qubits": "4", "lab.study_cell.n_circuit_layers": "3"}},
            "base-q4-d3",
        ),
        (
            {
                "preset_id": "base",
                "config": {
                    "lab.quantum_override.n_qubits": "5",
                    "lab.study_cell.n_qubits": "4",
                    "lab.study_cell.n_circuit_layers": "3",
                },
            },
            "base-q5-d3",
        ),
    ],
)
def test_variant_is_identical_across_views(job, expected):
    assert lab._job_variant(job) == expected
    assert explore._job_variant(job) == expected


@pytest.mark.parametrize(
    "job, expected",
    [
        ({"artifact_dir": "persisted/run", "checkpoint_path": "ignored/checkpoints/model"}, "persisted/run"),
        ({"checkpoint_path": "checkpoint/run/checkpoints/model"}, "checkpoint/run"),
        ({"run_name": "legacy-run"}, "legacy-run"),
    ],
)
def test_artifact_dir_is_identical_across_views(tmp_path, job, expected):
    root = tmp_path / "results"
    root.mkdir()
    expected_path = (root / expected).resolve()
    assert model_tests._artifact_dir(root, job) == expected_path
    assert diagnostics._artifact_dir(root, job) == expected_path


def test_artifact_dir_remains_confined_across_views(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    job = {"artifact_dir": "../outside", "run_name": "legacy-run"}
    for artifact_dir in (model_tests._artifact_dir, diagnostics._artifact_dir):
        with pytest.raises(ValueError):
            artifact_dir(root, job)
