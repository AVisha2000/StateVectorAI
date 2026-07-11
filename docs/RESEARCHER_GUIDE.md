# Researcher Guide

QLLM is a verification-first system for testing quantum mechanisms in machine
learning. A negative or null result is useful. A lower loss, larger Hilbert
space, entanglement signal, parameter-count reduction, or simulator result is
not by itself quantum advantage.

## Start with the claim contract

Before queuing a study, select the relevant entry in
[`research/claims.yaml`](../research/claims.yaml). It defines the claim ID,
metric type, present status, required fairness fields, allowed differences,
analogue ladder, practical-equivalence margin, limitations, and next decisive
test. [`RESULTS.md`](../RESULTS.md) is the historical record;
[`RESEARCH_MAP.yaml`](RESEARCH_MAP.yaml) is the area index; and
[`RESEARCH_PROGRAM.md`](RESEARCH_PROGRAM.md) is the program roadmap.

Do not edit claim level/status or strengthen `RESULTS.md` wording as an
ordinary engineering step. Claim promotion is a human gate.

## Evidence flow

```text
claim contract -> dataset bundle -> validated candidate and controls
               -> matched study -> immutable run manifests
               -> paired/equivalence/power analysis -> cautious report
```

The canonical path is:

1. Choose a claim ID and its declared metric.
2. Build every candidate and control from the same canonical config/data path.
3. Validate configuration with `qllm.config.validate_config` before queueing.
4. Record generator, split, initialization, minibatch, circuit, and hardware
   calibration axes separately. Use independent overrides only where the
   execution path supports them; current legacy runs couple initialization and
   minibatch (and applicable circuit randomness) to `train.seed`, use a
   deterministic split, and treat hardware calibration as not applicable.
5. Run the smallest CPU smoke first. Hardware and paid compute require explicit
   approval.
6. Interpret the study report together with fairness mismatches, analogue
   limitations, resource ledgers, and protocol warnings.
7. Record negative, equivalent, failed, cancelled, and contradicted outcomes.

## Evidence levels and sample size

- A standalone single-seed run is descriptive only.
- One matched pair is smoke evidence.
- Two or three pairs are a variance pilot, not an empirical edge.
- Six pairs is the repository-wide minimum for an edge assessment, but the
  claim-specific pilot-variance power plan may require more.
- A paired result still cannot support an edge when the confidence interval,
  sign-flip test, practical-equivalence assessment, power plan, fairness
  schema, or required analogue ladder is incomplete.

The deterministic statistics live in `qllm/research_protocol.py`. Dashboard
views present those results; the React UI does not recompute or promote them.

## Required controls

Always use the exact ladder declared by the claim. Typical families are:

| Study family | Minimum control set |
| --- | --- |
| Quantum component swap | Linked architecture-aware classical component, parameter/resource accounting, frozen or random quantum control, and a strong classical challenger |
| Kernel geometry/application | Prespecified train/validation/test selection, strong classical kernel families, and equal model-selection/resource budgets |
| Quantum memory | Matched classical memory frontier, independent generator instances, split/init/minibatch seed accounting, and oracle/resource accounting |
| Two-stream conditioning | Causal-prefix model, parameter-matched classical conditioner, no-conditioning ablation, and current metric contract |
| Diagnostic scaling | Prespecified sweep axis, repeated initializations, precision/backend identity, and resource ledger; diagnostics are not quality advantages |

An intentional architecture difference is allowed only when the claim schema
names it. Every other mismatch remains visible. Missing required controls or a
disallowed mismatch makes comparative interpretation invalid.

## Dashboard warnings

Evidence views show these warnings before verdicts and metrics:

| Code | Meaning |
| --- | --- |
| `single_seed` | Descriptive run or smoke pair; more independent matched evidence is required |
| `unmatched_comparison` | Candidate and baseline are not linked |
| `missing_control` | A required analogue or control rung is absent |
| `negligible_gain_high_cost` | A positive gain is below the claim's predeclared margin while wall cost is higher |
| `invalid_protocol` | Rerun requirement, metric/claim mixing, duplicate seeds, malformed manifest, or disallowed fairness mismatch blocks interpretation |

Missing values mean unavailable, not zero. Exact, sampled, noisy, and
approximate backend results remain distinct. Simulator wall time measures the
cost of classical simulation and must not be presented as QPU runtime.

## Resource and reproducibility checklist

For every serious run, inspect:

- experiment/run UUIDs and config, code, data, environment, and seed hashes;
- latest/best checkpoint, completed step, resume lineage, and recovery count;
- generator/split/initialization/minibatch/circuit/calibration seed axes;
- compile/first-step, steady-state, and total wall timing;
- parameters, state dimension, logical circuit calls, backend/device,
  precision, and available peak-memory evidence;
- backend capabilities and exactness/approximation metadata.

Historical rows remain readable, but unavailable fields must stay explicit.

## Research question to entry point

| Question | Entry point |
| --- | --- |
| Quantum/classical component ablation | `python benchmarks/ablation.py --help` |
| Kernel geometry and held-out controls | `python benchmarks/advantage_probe.py --help` |
| Gradient/trainability scaling | `python benchmarks/scaling_probe.py --help` |
| Predictive quantum-memory frontier | `python benchmarks/memory_sweep.py --help` |
| Contextual parity-memory diagnostics | `python benchmarks/contextual_sweep.py --help` |
| Causal two-stream conditioning | `python benchmarks/two_stream_probe.py --help` |
| Dataset predictability/long-memory screen | `python benchmarks/data_screen.py --help` |

Start the local cockpit with `python -m qllm.dashboard.run --port 8000`. Create
a study for multi-seed work; use the run workspace only for smoke/debugging.
The dashboard is loopback-only unless the explicit trusted-network remote flags
are supplied.

## Safe interpretation template

Report the studied regime, comparator, data-access model, metric, paired count,
interval/test/equivalence/power results, fairness and control status, resources,
limitations, and next decisive test. Prefer "diagnostic," "negative,"
"equivalent," "inconclusive," or "candidate under this protocol" when that is
what the evidence supports.
