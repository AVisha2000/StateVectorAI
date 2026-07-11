# Local Platform Completion Audit

Date: 2026-07-11  
Scope: milestones M01-M09 and the local CPU-capable engineering/dashboard
backlog. This audit classifies work; it does not promote a research claim.

## Status vocabulary

- **Completed**: implemented and covered by repository tests/verification.
- **Superseded**: the original design was replaced by a safer or more general
  contract that satisfies the underlying objective.
- **Deferred**: useful follow-on work outside local-platform completion.
- **Human-gated**: requires explicit approval because it uses hardware/spend,
  changes claim strength, exposes services, or mutates research artifacts.

## Engineering enhancement items

The 2026-07-11 backlog re-audit found that the original item-1 implementation
had not migrated several benchmark/report callers from the flat compatibility
loader and that shared information-theoretic measurements still assumed a
single stream. The current item-1 evidence includes that follow-up closure;
historical experiment artifacts were not rewritten or rerun.

| # | Item | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Boundary-safe synthetic sampling | Completed | `qllm/data/datasets.py:DatasetBundle`, boundary-aware split/sampling, trajectory-aware `qllm/evaluation.py`, bundle-preserving memory/interference/report benchmark callers and dashboard model tests, plus boundary/caller regressions in `tests/test_config_data.py`, `tests/test_v07.py`, `tests/test_quantum_data.py`, `tests/test_contextual.py`, and `tests/test_seq_cancellation.py` |
| 2 | Comprehensive config validation | Completed | `qllm/registry.py`, `qllm/config.py:validate_config`, CLI/model-spec/queue callers, `tests/test_config_data.py` |
| 3 | Two-stream causality and metric labeling | Partial; historical correction human-gated | Causal prefix summaries and leakage tests are complete in `qllm/models/two_stream.py` and `tests/test_two_stream.py`; dashboard/query contracts mark `two-stream-v1` as teacher-forced side-information and `rerun_required`, but `RESULTS.md` section 20 still needs an explicitly approved conservative correction before this item is complete. No post-repair `two-stream-causal-v2` study has completed. |
| 4 | Kernel evaluation split hygiene | Completed | deterministic train/validation/test selection in `benchmarks/advantage_probe.py`, `tests/test_advantage.py` |
| 5 | Study-level paired statistics | Completed | paired bootstrap, sign-flip, equivalence, and power planning in `qllm/research_protocol.py`, study payloads, `tests/test_research_protocol.py` |
| 6 | Full fairness comparison | Completed | claim-specific schemas and complete allowed/disallowed mismatch reporting in `research/claims.yaml` and `qllm/research_protocol.py` |
| 7 | Parameter-matched analogue generation | Superseded | `qllm/dashboard/analogues.py` and claim-specific analogue ladders combine architecture-aware controls, parameter tolerance, resources, frozen/random controls, and strong challengers; exact parameter equality is not treated as universal fairness |
| 8 | Result claim ledger | Completed | canonical `research/claims.yaml`, schema/loaders in `qllm/claims.py`, statistics/classification in `qllm/research_protocol.py`, and dashboard claim IDs/status/limitations |
| 9 | Durable dashboard queue | Completed | transactional claims, worker leases/heartbeats/recovery in `qllm/resultsdb.py` and `qllm/dashboard/runner.py`; recovery tests in `tests/test_dashboard_lab.py` |
| 10 | Checkpoint/resume | Completed | atomic latest/best checkpoints and CLI/dashboard resume metadata in `qllm/train/artifacts.py`, `qllm/train/loop.py`, runner/training tests |
| 11 | Idempotent per-step logging | Completed | additive repeatable SQLite migrations and unique step writes in `qllm/resultsdb.py` |
| 12 | Generation across architectures | Completed | explicit supported/unsupported generation outcomes in `qllm/train/loop.py:generation_capability`/`generate_outcome` and architecture tests |
| 13 | Localhost safety guardrails | Completed | loopback defaults, explicit remote flag/CORS, containment checks in `qllm/dashboard/run.py`, `server.py`, `security.py` |
| 14 | Dashboard comparison warnings | Completed | structured server warnings in `qllm/dashboard/evidence.py`; warning-first React evidence routes and regression tests |
| 15 | Dataset import boundaries | Completed | revision/hash/size/provenance/truncation handling and path containment in `qllm/dashboard/datasets.py`, `security.py` |
| 16 | Compile/runtime telemetry | Completed | measured runtime ledger in `qllm/resources.py`, collection/summaries in `qllm/train/loop.py` and `qllm/train/artifacts.py`, queue estimates in `qllm/dashboard/resources.py`, and dashboard ledgers |
| 17 | Scan/vectorization cleanup | Completed | static-shape scan/vectorized recurrent, contextual, transplant, and causal-prefix paths with parity/gradient tests |
| 18 | Scalable backend roadmap | Completed for local CPU scope; hardware scaling remains human-gated | distinct fixed-bond `tensorcircuit_mps` execution in `qllm/quantum/backends.py`; approximate capability/resource metadata with dense-state access disabled; deterministic value/gradient/JIT/nested-`vmap`/low-bond overlap coverage in `tests/test_tensorcircuit_mps.py`; realized error, convergence, discarded weight, and peak memory remain explicitly unmeasured |
| 19 | Component registries | Completed | centralized options in `qllm/registry.py` consumed by config, model, data, dashboard, circuit, backend, readout, and conditioning paths |
| 20 | Dataset object contract | Completed | canonical `DatasetBundle` plus compatible flat adapter in `qllm/data/datasets.py`; contextual masks no longer depend on a module global |
| 21 | Tested dependency matrix | Completed | validated top-level CPU/WSL/MPS pinned profiles, Windows/Linux Python 3.11/3.12 clean-install CI in `.github/workflows/dependency-matrix.yml`, `scripts/check_dependency_profiles.py`, native-Windows clean CPU/MPS install evidence, GPU visibility tooling, and expanded run environment manifests |
| 22 | Researcher onboarding | Completed | [`RESEARCHER_GUIDE.md`](RESEARCHER_GUIDE.md) and dashboard evidence vocabulary |
| 23 | Engineer onboarding | Completed | [`DEVELOPMENT.md`](DEVELOPMENT.md), scoped `AGENTS.md`, project skills, and component map |

The repeated GitHub issue backlog in `ENHANCEMENT_PLAN.md` maps to items 1-6,
9-10, 13, and 21 above. It is completed or superseded under the same evidence
except for the explicitly human-gated historical wording correction in item 3;
it is not a second independent backlog.

## UI upgrade phases

| Phase | Status | Evidence |
| --- | --- | --- |
| Research cockpit refresh | Completed | overview, filtered experiments, actions, queue state, responsive empty/error paths |
| Read-only model diagram | Completed | `qllm/dashboard/model_graph.py`, `ModelDiagram.jsx`, preset/job/comparison routes |
| Dedicated comparison | Completed | fairness/protocol/delta/curve/evidence views in `workspace.py` and `Comparison.jsx` |
| Studies and multi-seed evidence | Completed | study tables/payloads, paired/equivalence/power analysis, study/report pages |
| Per-layer model specs | Completed | `BlockConfig`, model-spec persistence/validation, global compatibility tests |
| Visual model builder | Completed | constrained registry-driven editor, save/validate/queue workflow |
| Quantum-advantage reporting | Superseded by cautious evidence reporting | claim IDs, warnings, fairness mismatches, analogue limitations, resource ledgers, paired statistics; no automatic claim promotion |
| Dataset/task hierarchy and data additions | Completed | explore/research result payloads, model specs, studies/study jobs, linked comparison groups, additive SQLite migrations |

## Explicit follow-on programs

These are not hidden incompleteness in the local platform:

- **Human-gated:** GPU/cluster sweeps, CUDA/JAX environment changes, cloud or
  paid services, QPU execution, remote dashboard exposure, destructive
  artifact/database migrations, and any stronger `RESULTS.md` or claim status.
- **Deferred:** additional tensor-network/Lightning/backend integrations,
  hardware or large-scale validation of the local MPS path, theorem-faithful
  contextual constructions, stronger predictive-state/classical attacks, hard
  routing, and the experiments in `GPU_QUEUE.md`.
- **Research-dependent:** reruns of historical two-stream side-information
  studies and any confirmation study whose claim contract calls for more
  independent pairs or controls.

`GPU_QUEUE.md` remains a proposed run program, not permission to execute it.
The authoritative next decisive test for each research area remains in
`research/claims.yaml` and `docs/RESEARCH_MAP.yaml`.

## Completion conclusion

All local CPU-capable enhancement and UI items are implemented, superseded by
an evidence-backed contract, or explicitly classified above. The repository is
complete as a local research platform under the nine-milestone scope. This
conclusion says nothing stronger about quantum advantage; scientific claims
remain at their recorded levels and statuses.
