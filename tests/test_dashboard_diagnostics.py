from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from qllm.dashboard.diagnostics import DIMENSION_NAMES, diagnostics_payload
from qllm.resultsdb import ResultsDB


def _job(db: ResultsDB, artifacts: Path, *, group_id: str | None = None, qubits: int = 2) -> dict:
    job_id = db.create_lab_job({
        "status": "done", "preset_id": "quantum-ffn-4q", "dataset_name": "test",
        "run_name": f"run-{qubits}-{group_id or 'single'}", "seed": qubits,
        "steps": 2, "eval_every": 1, "group_id": group_id,
        "artifact_dir": str(artifacts / f"run-{qubits}-{group_id or 'single'}"),
        "config": {"lab.study_cell.n_qubits": qubits},
    })
    return db.get_lab_job(job_id)


def _write_summary(job: dict, payload: dict) -> None:
    directory = Path(job["artifact_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _diagnostics(variance: float = 0.25) -> dict:
    return {
        "grad_var_first_param": variance / 2, "grad_var_mean": variance,
        "grad_var_max": variance * 2, "expressibility_kl": 0.12,
        "meyer_wallach_q": 0.75,
        "parameter_shift_gradient_snr": {"median_snr": 3.0, "mean_snr": 4.0},
        "availability": {"expressibility_kl": {"reason": "saved backend reason"}},
    }


def test_diagnostics_payload_extracts_measured_saved_dimensions(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    job = _job(db, tmp_path / "artifacts")
    _write_summary(job, {"quantum_diagnostics": _diagnostics()})

    payload = diagnostics_payload(db, job["id"], tmp_path / "artifacts")

    assert set(payload["diagnostics"]) == set(DIMENSION_NAMES)
    assert payload["diagnostics"]["gradient_variance"]["value"]["grad_var_mean"] == 0.25
    assert payload["diagnostics"]["parameter_shift_gradient_snr"]["status"] == "measured"
    assert payload["diagnostics"]["expressibility_kl"]["value"] == 0.12
    assert payload["diagnostics"]["meyer_wallach_q"]["value"] == 0.75
    assert payload["diagnostics"]["scaling_fit"]["status"] == "unavailable"
    assert all("source" in row and "provenance" in row for row in payload["diagnostics"].values())
    assert "advantage" in payload["interpretation_warnings"][0]["message"].lower()


@pytest.mark.parametrize("summary", [None, "{broken", []])
def test_missing_or_invalid_summary_returns_explicit_unavailable(tmp_path, summary):
    db = ResultsDB(tmp_path / "results.db")
    job = _job(db, tmp_path / "artifacts")
    if summary is not None:
        directory = Path(job["artifact_dir"])
        directory.mkdir(parents=True)
        content = summary if isinstance(summary, str) else json.dumps(summary)
        (directory / "summary.json").write_text(content, encoding="utf-8")

    payload = diagnostics_payload(db, job["id"], tmp_path / "artifacts")

    assert {row["status"] for row in payload["diagnostics"].values()} == {"unavailable"}
    assert all(row["reason"] for row in payload["diagnostics"].values())


def test_unconfined_artifact_path_is_unavailable(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    job = _job(db, tmp_path / "artifacts")
    db.update_lab_job(job["id"], artifact_dir=str(tmp_path / "outside"))

    payload = diagnostics_payload(db, job["id"], tmp_path / "artifacts")

    assert payload["diagnostics"]["gradient_variance"]["status"] == "unavailable"
    assert "must stay within" in payload["diagnostics"]["gradient_variance"]["reason"]


def test_scaling_fit_uses_only_same_group_persisted_rows(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    first = _job(db, tmp_path / "artifacts", group_id="sweep", qubits=2)
    second = _job(db, tmp_path / "artifacts", group_id="sweep", qubits=4)
    other = _job(db, tmp_path / "artifacts", group_id="other", qubits=8)
    _write_summary(first, {"quantum_diagnostics": _diagnostics(0.25)})
    _write_summary(second, {"quantum_diagnostics": _diagnostics(0.0625)})
    _write_summary(other, {"quantum_diagnostics": _diagnostics(1000.0)})

    payload = diagnostics_payload(db, first["id"], tmp_path / "artifacts")

    scaling = payload["diagnostics"]["scaling_fit"]
    assert scaling["status"] == "measured"
    assert scaling["provenance"]["persisted_rows"] == 2
    assert scaling["provenance"]["distinct_qubit_counts"] == 2
    assert scaling["value"]["variance_decay_factor_per_qubit"] == pytest.approx(0.5)


def test_scaling_fit_requires_distinct_qubit_counts(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    first = _job(db, tmp_path / "artifacts", group_id="seeds", qubits=4)
    second = _job(db, tmp_path / "artifacts", group_id="seeds", qubits=4)
    _write_summary(first, {"quantum_diagnostics": _diagnostics(0.25)})
    _write_summary(second, {"quantum_diagnostics": _diagnostics(0.125)})

    payload = diagnostics_payload(db, first["id"], tmp_path / "artifacts")

    scaling = payload["diagnostics"]["scaling_fit"]
    assert scaling["status"] == "unavailable"
    assert scaling["provenance"]["persisted_rows"] == 2
    assert scaling["provenance"]["distinct_qubit_counts"] == 1


def test_unknown_job_raises_key_error(tmp_path):
    with pytest.raises(KeyError, match="Unknown job 999"):
        diagnostics_payload(ResultsDB(tmp_path / "results.db"), 999, tmp_path)


def test_nonfinite_or_malformed_values_are_rejected_and_saved_reason_is_preserved(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    job = _job(db, tmp_path / "artifacts")
    diagnostics = _diagnostics()
    diagnostics["expressibility_kl"] = math.nan
    diagnostics["grad_var_mean"] = "not-a-number"
    diagnostics["meyer_wallach_q"] = None
    diagnostics["availability"]["meyer_wallach_q"] = {"reason": "state access unavailable"}
    _write_summary(job, {"quantum_diagnostics": diagnostics})

    payload = diagnostics_payload(db, job["id"], tmp_path / "artifacts")

    assert payload["diagnostics"]["gradient_variance"]["status"] == "unavailable"
    assert payload["diagnostics"]["expressibility_kl"]["status"] == "unavailable"
    assert payload["diagnostics"]["meyer_wallach_q"]["reason"] == "state access unavailable"


def test_payload_has_no_composite_or_advantage_score_field(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    job = _job(db, tmp_path / "artifacts")
    _write_summary(job, {"quantum_diagnostics": _diagnostics()})
    payload = diagnostics_payload(db, job["id"], tmp_path / "artifacts")

    def keys(value):
        if isinstance(value, dict):
            yield from value
            for item in value.values():
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    all_keys = set(keys(payload))
    assert "advantage_score" not in all_keys
    assert "composite_score" not in all_keys
