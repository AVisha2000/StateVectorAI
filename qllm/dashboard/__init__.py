"""Self-hosted dashboard for the QLLM testbed (own MLflow replacement)."""
from __future__ import annotations

import dataclasses


def with_dashboard(cfg, suite, variant, dataset, seed,
                   db="results/qllm_results.db"):
    """Return a copy of an ExperimentConfig with per-step dashboard logging
    enabled. Drop-in for benchmarks: wrap the cfg before fit()."""
    tr = dataclasses.replace(
        cfg.tracking, dashboard_db=db, dashboard_suite=suite,
        dashboard_variant=variant, dashboard_dataset=dataset,
        dashboard_seed=seed)
    return dataclasses.replace(cfg, tracking=tr)
