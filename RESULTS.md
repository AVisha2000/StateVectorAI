# RESULTS — v0.1 verified runs (June 10, 2026, 1 CPU core)

## 1. Test suite

`pytest tests/ -q` → **32 passed, 1 skipped** (TensorCircuit parity test;
optional dep not installed), ~2 min. Includes:

- parameter-shift ≡ backprop gradients (max diff 1.2e-7) ≡ finite differences
- causality (future tokens cannot affect past logits)
- gradient flow to every parameter incl. `circuit_weights`
- jit ≡ eager for classical and quantum layers
- Meyer-Wallach known values (|00⟩→0, Bell→1, GHZ₃→1)
- end-to-end `fit()` for classical AND hybrid configs (same pipeline)

## 2. Training comparison (300 steps, identical pipeline/data/seed)

| run | attn | ffn | params | val loss | val ppl | val bpc | wall |
|---|---|---|---|---|---|---|---|
| classical-small | classical | classical | 116,673 | 2.4400 | **11.47** | 3.520 | 16s |
| quantum-attn-4q | quantum_proj | classical | 109,561 | 2.4469 | 11.55 | 3.530 | 29s |
| quantum-ffn-4q | classical | quantum 4q | 51,705 | 2.4850 | 12.00 | 3.585 | 27s |

Reading (with appropriate caution — single seed, 300 steps, far from
convergence):

- The pipeline is **fair**: same data order, same optimizer, same eval
  batches; only the configured block differs.
- The 4-qubit VQC FFN reaches within ~5% of classical perplexity with
  **44% of the parameters** — but this is NOT evidence of quantum
  advantage. The honest next test is a parameter-matched classical model
  (~52k params) and a frozen/random-circuit control (the 2×2 ablation in
  the plan).
- Quantum-attn ≈ classical here mostly because only the output projection
  (a small fraction of compute) was replaced.

Quantum step overhead at this scale: ~25 ms/step vs ~3 ms classical
(~8×) on CPU — entirely tractable for the experimental regime.

## 3. Barren-plateau scaling probe (the Phase-3 deliverable)

Reuploading ansatz, L=2, local ⟨Z₀⟩ cost, 64 random inits/point:

| qubits | Var[∂C/∂θ] (mean) | Meyer-Wallach Q | expressibility KL |
|---|---|---|---|
| 2 | 5.88e-02 | 0.428 | 0.217 |
| 4 | 1.58e-02 | 0.850 | 0.078 |
| 6 | 4.71e-03 | 0.909 | 0.031 |
| 8 | 1.07e-03 | 0.936 | 0.029 |
| 10 | 3.96e-04 | 0.942 | ~0 |
| 12 | 7.09e-05 | 0.935 | ~0 |

**Fit: variance shrinks ×0.52 per added qubit (R² visually excellent on
log scale)** — a clean exponential, i.e. the barren-plateau signature, and
it co-occurs exactly as theory predicts: entanglement saturates and the
ansatz approaches the Haar/2-design regime (KL→0). Extrapolation:

- n=16 → ~5e-6 · n=32 → ~1.5e-10 · n=64 → ~1e-19

Per the go/no-go thresholds: **this ansatz at this depth is barren well
before n=32.** Conclusions for the next iteration: (a) keep circuits at
4–8 qubits per block and scale by adding *blocks*, not qubits; (b) test
identity-block / small-angle initializations and layerwise training;
(c) the structural zero in θ₀ (first RZ on |0⟩) means single-parameter
variance is misleading — the probe fits the mean (fixed in
`benchmarks/scaling_probe.py`).

Artifacts: `results/scaling_reuploading_L2.{csv,png}`, fit JSON, all rows
also in MLflow experiment `qllm-scaling` (step = qubit count).

## 4. Infrastructure findings (planning-doc deltas)

1. **MLflow 3.13 hard-deprecates the `./mlruns` file store** → defaults
   switched to `sqlite:///mlflow.db`. `mlflow ui --backend-store-uri
   sqlite:///mlflow.db` to browse.
2. **JAX moved to 0.10.1** (plan anticipated 0.7.1). PennyLane 0.45 works
   with it; warning is benign, gradients verified three ways.
3. Quantum diagnostics for both 4q runs are identical by construction
   (same circuit spec) — they characterize the *circuit*, not the trained
   model. Logged once per run, queryable via `mlflow.search_runs`.

## 5. Generation sample (classical, 300 steps — vibes check only)

```
ROMEO:
Lof he we dl, sisthace risthes histhonnth
Thomur g t belo an ver,
The p med I theledomourouteses m, indwimar wing hou arolkir ound...
```

Character-level structure (line breaks, capitalized speaker tags) emerging
at 300 steps; needs ~5k steps for coherent words. Expected and fine — the
testbed optimizes for iteration speed, not fluency.

## 6. 2×2 ablation (quantum contribution isolation)

3 seeds × 300 steps, identical pipeline. Parameter-matched twin solved
automatically: d_ff=4 (51,657 params vs quantum's 51,705 — also
*structure*-matched: both FFNs are 64→4→64 bottlenecks, GELU vs 2-layer
entangled VQC at the same width).

| variant | params | val ppl (mean ± std) |
|---|---|---|
| quantum-trained | 51,705 | 12.281 ± 0.245 |
| quantum-frozen (random circuit) | 51,705 | 12.290 ± 0.270 |
| classical-matched (d_ff=4) | 51,657 | 12.353 ± 0.221 |
| classical-full (d_ff=256) | 116,673 | 11.726 ± 0.246 |

**Decision-rule outcomes (planning-doc threshold (b) triggered):**

1. **Trained ≈ frozen (Δppl +0.010, within noise).** Training the circuit
   weights adds nothing measurable at this scale/budget — the quantum
   layer behaves as a *fixed random feature map*, with the classical
   pre/post projections doing the learning.
2. **Trained quantum ≈ parameter-matched classical (Δppl +0.073, within
   noise).** No evidence of quantum contribution: the earlier "within 5%
   of classical at 44% params" observation is fully explained by the
   bottleneck architecture, not the circuit.
3. classical-full wins, as expected, by spending 2.3× parameters.

Caveats: 300 steps is far from convergence; one circuit spec (4q, L=2,
reuploading); single dataset. The framework's job was to make this test
cheap and fair — rerun with `python benchmarks/ablation.py --steps 5000`
for the longer-horizon version. Per the roadmap's go/no-go logic, the
next experiments that could change this verdict: deeper/wider circuits at
4–8 qubits (scaling by blocks, not qubits, per §3), alternative ansätze,
and identity-block initialization.

## 7. Advantage-potential probe (v0.3 — geometric difference + controls)

New instrument (`qllm/quantum/advantage.py`, Huang et al. Nat. Commun.
2021 methodology): instead of hoping a dataset has quantum structure,
measure **g(K_C‖K_Q)** — whether the quantum feature map can beat all
classical kernels on ANY labels — and *engineer* the maximal-advantage
labels as a positive control. N=240, reuploading L=2:

| qubits | g (vs best classical; √N=15.5) | R² engineered: Q vs C | R² classical labels: Q vs C | K_Q off-diag |
|---|---|---|---|---|
| 4 | 9.26 | **0.997** vs 0.665 | 0.938 vs **1.000** | 0.165 ± 0.176 |
| 6 | 11.39 | **0.824** vs 0.068 | 0.845 vs **0.989** | 0.053 ± 0.088 |
| 8 | 11.18 | **0.537** vs 0.069 | 0.535 vs **0.965** | 0.020 ± 0.044 |

**Both controls PASS** — the framework now demonstrably detects quantum
advantage when it exists (positive control) and doesn't hallucinate it
when it doesn't (negative control). Two live phenomena: (a) g ≫ 1, so
this feature map has real advantage *room* on this input distribution;
(b) exponential concentration is already visible (off-diagonals → 0),
degrading even the quantum kernel's own learnability with n — the
advantage/trainability tension, measured.

## 8. Sharp ablation (v0.3 architecture upgrades)

Upgrades: ZZ-correlator readout (10 feats/circuit), **linear dressing**
(circuit is the only nonlinearity), 4 parallel circuit heads, small-angle
init (0.3), plus live circuit-vs-classical gradient-norm logging.

| variant | params | val ppl (mean ± std) |
|---|---|---|
| quantum-trained | 58,017 | 12.276 ± 0.191 |
| quantum-frozen | 58,017 | 12.270 ± 0.190 |
| classical-matched (d_ff=29) | 58,107 | 12.119 ± 0.282 |
| classical-full | 116,673 | 11.726 ± 0.246 |

**Verdict unchanged — but now with mechanism.** Trained ≈ frozen to 0.01
ppl per seed. The new grad-norm telemetry rules out vanishing gradients:
‖g_circ‖/‖g_cls‖ ≈ 0.04 with 96 vs ~58k parameters ⇒ **per-parameter
gradient signal ≈ 1×** (healthy). The circuit's contribution is
*functionally redundant*: a frozen random VQC is already a good random
feature map, the post-projection learns on top of it, and tuning 96
circuit parameters cannot move the achievable feature set at this scale —
the quantum analog of the classical random-features phenomenon.

**Synthesis (the v0.3 thesis):** architecture was not the bottleneck;
*data* is. On classical text, labels do not live in the quantum-favored
subspace, so any quantum layer degenerates to random features (§8) even
though the feature map provably has advantage room (§7, g≫1). The search
for truer advantage must therefore move data-side: quantum-generated /
quantum-structured sequence datasets, scored first with g before any
training is spent. That toolkit now exists and is validated.

## 9. v0.4 — quantum-generated data: the full causal chain validates

Built: monitored kicked-Ising sequence generator (quantum HMM with
tunable monitoring), matched order-k Markov twin, dataset dispatch
(`data.kind`), task-alignment screen (s_K model complexity), and
entropy-based predictability/long-memory gates. See DATA.md for the
data strategy and full numbers.

Headline measurements: (a) the self-dual chaotic point is a ppl-4.0
ceiling for everyone (confirmed in training — maximal scrambling =
nothing to learn); (b) resonant weak monitoring (1 qubit / 2 periods,
θ_x=0.6) yields 0.121 bits/token of genuine beyond-3-gram structure;
(c) both quantum and classical transformers extract ALL of it (1.592
vs ideal 1.59; twin capped at 1.73 exactly as predicted); (d) no
quantum-model advantage — a 5-qubit memory is classically easy.
Conclusion: data pipeline and detection instruments are validated
end-to-end; separation now requires scaling the quantum memory and a
*recurrent* quantum cell (v0.5).

## 10. v0.6 — the comprehensive QNLP suite (every site hot-swappable)

Every LLM component now swaps to quantum via config alone: embedding
(`embed_type: quantum` — words-as-quantum-states, the transferable core
of Coecke's DisCoCat/lambeq line, which itself is grammar-compositional
and classification-oriented rather than autoregressive), attention
(`quantum_proj` | `quantum_qkv`), FFN, whole blocks, and the recurrent
core (`arch: qrnn`). All runs land in a SQLite results DB keyed by
(suite, variant, dataset, seed, steps) — re-invocations skip finished
work, so sweeps resume across crashes and machines (GPU-ready).

Leaderboard (tiny-shakespeare, 300 steps, 2 seeds, identical pipeline):

| variant | params | val ppl |
|---|---|---|
| gru-64 | 33,217 | **10.47 ± 0.21** |
| classical transformer | 116,673 | 11.72 ± 0.35 |
| q-attn-qkv | 96,505 | 11.78 ± 0.51 |
| q-attn-proj | 110,329 | 11.94 ± 0.56 |
| q-ffn | 52,473 | 12.15 ± 0.28 |
| q-embed | 113,501 | 12.80 ± 0.57 |
| q-block | 46,129 | 12.95 ± 0.34 |
| q-full | 42,957 | 14.10 ± 0.82 |

Readings: (1) no quantum swap beats classical on text — q-attn-qkv ties
within noise, and degradation grows monotonically with the number of
quantum insertion sites (q-full worst), consistent with §7-8: each site
is a random-feature bottleneck on classically-structured data. (2) A
plain GRU wins outright at this scale — short training favors
recurrence on char-level text; a reminder that the classical reference
class matters as much as the quantum variants. (3) Caveats: 300 steps,
2 seeds, one circuit spec (4q/L2/zz); the suite exists precisely so
these cells can be re-filled at depth on a GPU with one resumable
command. Plot: results/qnlp_suite_text.png.

## 11. v0.7 — quality battery, generative output checks, weight transplant

**Battery** (`benchmarks/model_report.py`, metrics table in the DB):
val-ppl + memory-gain vs order-k Markov floors + calibration (ECE/NLL) +
**generative fidelity** — sample from the trained model (multi-prompt
short rollouts; single long rollouts absorb into degenerate basins, now
measured as `gen_max_runlen_frac`) and compare statistics to held-out
truth. On the quantum (ising) data, 500 steps, seed 0:

| metric | gru-64 | classical | qrnn |
|---|---|---|---|
| val ppl | **1.577** | 1.616 | 1.634 |
| memory gain vs markov-3 | **0.119** | 0.080 | 0.062 |
| ECE | **0.011** | 0.017 | 0.034 |
| generated 4-gram TV ↓ | 0.153 | 0.141 | **0.139** |
| generated entropy gap | −0.134 | −0.094 | **−0.063** |
| absorption (max-run frac) ↓ | 0.170 | 0.167 | **0.084** |

Split verdict = the finding: classical models win likelihood, but the
**QRNN's sampled output is distributionally most faithful** to the true
quantum process — least under-dispersed, half the absorption — an
inductive-bias signature invisible to perplexity alone. (1 seed, 500
steps; rerun deeper on GPU.)

**Weight transplant** (`benchmarks/weight_transplant.py`): donor FFNs
contracted to 4-qubit cores (45% Frobenius energy retained — the
SLM-scale contraction cost, measured), polar-decomposed, rotation
compiled into the gate set (global-phase-parameterized, multi-restart:
0.97 fidelity at 3 qubits/L=6; 0.28 at 4 qubits/L=6 — needs L≳10),
surgically warm-started as `ffn_type: quantum_linear`. Text, 300 steps,
seed 0: donor 11.47 (117k) | warm zero-shot 36.7 | **warm-ft 11.11** |
cold-ft 12.09 — at 54.8k params. Warm beats cold by ~1.0 ppl AND edges
the classical donor at 47% of its parameters. Honest framing: the
amplitude-space unitary FFN is exactly classically simulable — this is
quantum-INSPIRED compression (MPO/disentangler family), not advantage.

## 12. v0.8 — controls, hardening, and the corrected record

**Transplant-v2 (the causal control, 3 seeds, text, 300 steps).** Added
the missing twin: a classical rank-16 LINEAR bottleneck (`ffn_type:
lowrank`) warm-started from the SAME SVD factors as the quantum layer.

| variant | params | val ppl |
|---|---|---|
| lr-warm (classical SVD warm-start) | 54.6k | **11.16 ± 0.26** |
| q-warm (quantum transplant, L=20, fid≈0.986) | 55.1k | **11.17 ± 0.26** |
| donor (full classical) | 116.7k | 11.73 ± 0.25 |
| q-cold / lr-cold | ~55k | 12.23 / 12.30 |

Identical zero-shots per seed (e.g. 24.30 vs 24.28). **Corrected
attribution: the v0.7 win is low-rank + warm-start; the circuit adds
nothing beyond its classical twin once compiled faithfully.** Supporting
fixes: row-convention transpose bug in warm-start (pinned by a
round-trip test), compile-depth law at 4 qubits (fid 0.28/0.56/0.82/0.98
at L=6/10/14/20 vs dim O(16)=120), `lax.scan` over circuit layers
(compile O(1) in depth; was 10+ min via telemetry at L=20), and a
`log_grad_norms` gate.

**Battery hardened (3 seeds ising + Markov-twin control).** What
survived: (a) **inductive-bias asymmetry** — QRNN memory-gain is
positive on quantum data in every seed (+0.046 avg) but NEGATIVE on the
classical twin (−0.061), while GRU is positive on quantum (+0.099) and
~0 on the twin (its max there): the quantum cell is genuinely
specialized for quantum-generated structure. (b) **absorption
resistance** — QRNN has the lowest max-run fraction in every seed (mean
0.057 vs 0.088/0.100). NOT survived: the seed-0 generated-TV/entropy-gap
advantage (noise). Also robust: QRNN is consistently worst-calibrated
(ECE ≈ 0.036) and worst on likelihood.

**Exact-ansatz QRNN (ZZ-ring, 2000 steps): 1.637** — identical to the
approximate ansatz. The model class provably contains the generator yet
optimization plateaus ~0.06 above the ~1.58 floor: a landscape problem
(BPTT through 64 collapses), not expressivity. GPU follow-up: lr/init/
length sweep.

**Separation flagship ready**: `benchmarks/memory_sweep.py` sweeps
memory qubits m (belief filter ~4^m vs m+1 model qubits), records
memory-gain vs per-m Markov floors, resumable; CPU smoke at m=8 passes.
GPU: `python benchmarks/memory_sweep.py --memory-qubits 6 8 10 12
--steps 2000 --seeds 0 1`.

## 13. v0.9 — landscape, resonance, and the first separation curve

**Representability bug found and fixed.** The QRNN's mandatory per-layer
CNOT ring is an entangling Clifford the kicked-Ising generator does not
contain — single-qubit rotations cannot cancel it, so "exact" was only
approximate. New Ising-form ansatz (`ansatz: ising`: ZZ-ring + Rot only)
contains U_F^k exactly at L=k.

**Planted-solution diagnostic.** Setting the generator's own parameters
(verified Euler angles, zero injection): val ppl **1.527** at the old
m=5 setting — below every trained model and below the old "~1.59 floor"
estimate. No trained model had been extracting all the quantum memory.

**Landscape study (suite qrnn-landscape-v1).** Planted basin is stable
under fine-tuning (1.542) and robust to sigma=0.3 parameter noise; ALL
12 random-init runs (L in {2,3} x lr in {3e-3,1e-2,3e-2} x 2 seeds)
converge to a distinct sub-optimal attractor at 1.604-1.616. A clean
bad-basin (not vanishing-gradient) optimization failure. Consequence:
the flagship uses the PLANTED filter as the quantum reference —
training-free, exact, and sidesteps the landscape entirely.

**Resonance search (suite resonance-v1).** The old setting's
markov-vs-planted gap collapses with m (0.17 -> 0.02 by m=8). Scanning
(theta_x x steps_per_token) per m found a strong ridge at theta_x=0.75
(near the self-dual point), spt=2: gap = 0.67 / 0.63 / 0.57 / 0.43 at
m = 6/8/10/12 — the record looks ~random to 3-grams (~1.98 ppl) while
the exact filter predicts at 1.27-1.55. Generator rewritten gate-wise
(contiguous einsum; exact to 1e-12 vs dense; ~30x faster; m<=18 viable).

**First separation curve (suite memory-sweep-v2, full-size data, seed 0,
2000-step GRUs).** At a FIXED small capacity, the classical recurrence
degrades relative to the planted quantum filter as memory grows: gru8
(450 params) sits 0.06 below the filter at m=6 but 0.21 / 0.27 below at
m=8 / m=10 — the separation signature. Larger GRUs (gru16/32, 1.7k/6.4k
params) still track the filter through m=10, so no separation for
well-provisioned classical models in this range YET; whether the
required capacity diverges is what the m=12-16 extension tests. The
planted reference is conservative (uniform-prior window-sync penalty —
over-provisioned GRUs edge above it), so "match" is defined generously
and the honest signal is the fixed-capacity gain-retention curve
(results/separation_curve.png, left), not a threshold crossing. Caveats:
1 seed; fixed 2000-step budget. GPU continuation: m=12-16 (data gen
~5 min/m, cached), gru128-512 rungs, 3 seeds, steps-robustness, and the
trained ising-ansatz QRNN track (vs the planted upper bound).

## 14. v0.10 — a contextuality track: aiming for UNCONDITIONAL advantage

Motivated by Anschuetz–Hu–Huang–Gao (arXiv:2209.14353) and Zhao–Deng
(npj QI 2025): generic quantum dynamics (our Ising track) gives at best
constant-factor, optimization-limited separations. The UNCONDITIONAL,
provable separation has a named source — quantum **contextuality** —
which forces classical models into an Omega(n^2) memory wall where a
quantum recurrence needs O(n). Crucially this is an EXPRESSIVITY/memory
separation, immune to the barren-plateau and bad-basin problems that
capped our Ising QRNN, because the hard instance is explicit.

**New task (`qllm/data/contextual.py`).** Interleaved parity-context
streams: a cue token reveals an observable id, a value token reveals its
bit; `n_live` contexts are open at once and each context's final bit is
parity-FORCED by its earlier bits. Predicting forced bits requires
recalling the full live measurement context — the discrete analog of
contextuality's "measurement outcomes depend on prior commuting
measurements". Marginal token entropy is near-max (structure invisible
without memory); ~25% of value tokens are constrained. Scored by
`constrained_accuracy` (parity bits only; chance = 0.5).

**Classical memory wall measured (suite contextual-v1, 2000 steps, seed
0; results/contextual_wall.png).** Constrained-token accuracy vs GRU
size:

| GRU params | n_live=2 | n_live=3 | n_live=4 |
|---|---|---|---|
| 654 | 0.557 | 0.571 | 0.588 |
| 2,062 | 0.615 | 0.579 | 0.603 |
| 7,182 | 0.654 | 0.659 | 0.649 |
| 26,638 | 0.660 | 0.733 | 0.746 |
| 102,414 | 0.840 | 0.881 | 0.873 |

Accuracy climbs monotonically with capacity and NO GRU — even 102k
params — reaches the ceiling on bits that are fully determined by
context: a capacity-bound wall, not an optimization one (unconstrained
bits sit near chance throughout, confirming the models extract memory,
not spurious correlations). Honest caveats: the n_live ordering is
noisier than theory's clean version (random per-context observable
sampling dilutes effective contextuality — the robust axis is capacity,
not depth); 1 seed; the quantum side is not yet run (the QRNN's contextual
variant needs a parity-register encoding and vocab=2^k, next session).
This establishes the classical wall the quantum model must beat — the
cleanest shot at unconditional advantage in the whole project.

## 15. v0.11 — interference output head (a NEW idea, not in the literature)

Every QLM result I found puts the quantum part in the recurrent MEMORY
(CRNNs, our QRNN) or a feature map; the output projection is always a
classical softmax. This section tests a different, untried location: a
quantum OUTPUT head exploiting amplitude interference.

**The primitive.** A classical mixture head forms p(t) = sum_h w_h
p_h(t) with w_h, p_h >= 0 — it can only ADD evidence. A coherent head
forms p(t) = | sum_h c_h a_h(t) |^2 with complex c_h, a_h, squaring
AFTER the sum, so hypotheses can DESTRUCTIVELY INTERFERE: "allowed under
reading A, allowed under reading B, forbidden when both" is a single-layer
constraint here and provably not a single positive-mixture layer.
(`qllm/quantum/interference_head.py`: InterferenceHead, MixtureHead,
LinearHead. interference-K and mixture-2K are exactly parameter-matched —
complex doubles per-branch params.)

**Head-only probe (suite interference-head-v1, 3 seeds).** To isolate the
head, inputs are purely LINEAR encodings of k binary context features and
the target allowed-token set applies XOR-cancellation on conflict groups
(allowed iff an ODD number of features fire). No body precomputes the
conjunction. Excess cross-entropy over the exact entropy floor:

| head | params (k=2) | excess CE, k=1 / k=2 / k=3 |
|---|---|---|
| interference-1/2/4 | 136-544 | **0.0000 / 0.0000 / 0.0000** |
| mixture-2 | 136 | 0.0008 / 0.0021 / 0.0035 |
| mixture-4 | 272 | 0.0004 / 0.0013 / 0.0015 |
| mixture-8 | 544 | 0.0004 / 0.0009 / 0.0010 |
| linear | 64 | 0.0036 / 0.0405 / 0.0060 |

Interference reaches the floor EXACTLY at every arity — even a single
complex branch (interference-1, ~100-170 params) does what mixture-8
(4-8x the branches) cannot — with zero seed variance. The separation is
real and consistent (results/interference_head.png).

**Honest calibration.** (1) The gap MAGNITUDE is modest here (~0.001-0.004
nats); this is a clean EXPRESSIVITY demonstration, not yet a large
perplexity win. (2) It is a single-LAYER claim: a 2-layer classical MLP
head can also represent XOR, so the result says interference buys the
cancellation primitive in ONE layer / O(1) depth, not that classical
networks can never express it. (3) The effect only appears when the BODY
cannot precompute the conjunction: on the full-transformer interference
data task all three heads tied (~9.3 ppl) because attention computes the
features and any head then suffices — so the head-only probe is the
correct instrument. (4) Fully classically simulable at this size (the
point is an inductive-bias/expressivity primitive, not hardware speedup).
What is novel: interference at the OUTPUT as a depth-efficient primitive
for cancellation-structured constraints — a direction the QLM literature
has not explored.

## 16. v0.12 — does interference compound at the sequence level? (honest: no, here)

Following the v0.11 head-only result, the critical question: does the
interference head's single-step expressivity edge convert to a
SYSTEM-LEVEL perplexity win in a sequence model? Three experiments,
reported regardless of outcome.

**Sequential cancellation task** (`qllm/data/seq_cancellation.py`):
allowed-token sets governed by a running parity of recent tokens, with a
tunable cancellation DENSITY (fraction of vocab in the conflict tail).
Cancellation structure recurs every step.

**Result 1 — frozen random body (suite seq-interference-v1).** A fixed
random GRU featurizes the stream; only the head trains. ALL heads fail
badly (ppl ~14 vs floor 3.77): a random recurrent map does not preserve
the parity bits in a head-readable form, so no head — interference
included — can recover them. Interference shows no edge (slightly worse).

**Result 2 — trained full-width body (suite interference-v1, d_model=64).**
All heads TIE at 9.32 ppl: a trained body precomputes the conjunction,
after which any head suffices. linear 9.318, mixture-4 9.320,
interference-2 9.329 — within noise.

**Result 3 — trained bottlenecked body (suite interference-width,
d_model=8).** Even when the body is too narrow to cheaply precompute all
conjunctions, no head separates: mixture-4 11.36, linear 11.44,
interference-2 11.51 (±0.1-0.3). The bottleneck hurts all heads equally
rather than rewarding interference.

**Conclusion (negative, and clarifying).** The interference head's
expressivity advantage is real and reproducible IN ISOLATION (v0.11,
exact-floor at matched params on linear-feature XOR), but across frozen,
full-width, and bottlenecked sequence models we found NO regime where it
yields a perplexity gain. The advantage requires features that linearly
expose the cancellation inputs while denying the conjunction — a
condition the head-only probe constructs by hand but that trained bodies
either satisfy trivially (then any head wins) or destroy (then none do).
This bounds the v0.11 claim precisely: interference is a genuine
single-layer expressivity primitive, not a demonstrated end-to-end
advantage for autoregressive LMs. The testbed now records both the
positive isolation result and this negative system-level result — the
honest scientific state of the idea.

## 17. v0.13 — contextual quantum cell: a phase-accumulator memory (partial)

Built a purpose-designed quantum recurrent cell for the contextuality
task (`qllm/quantum/contextual_cell.py`, arch `contextual_qrnn`),
targeting the O(n)-qubit side of the Omega(n^2)-classical memory wall
(v0.10). Design: an n_phase-qubit register where each qubit starts in |+>,
tokens imprint RZ phases (a pi phase flips |+> <-> |->, encoding PARITY in
the X-basis sign — the linear-in-history quantity contextuality needs),
and readout applies Hadamard-back + Born measurement so parity becomes a
decodable 0/1 amplitude via interference.

**What works (verified, in tests).** On a pure running-parity task (track
the cumulative parity of a bit stream) the cell reaches **100% accuracy,
CE -> 0.003** with 3 qubits / 1 layer — exactly the interference-based
parity readout the contextuality escape relies on. A first design that
imprinted diagonal phases without the Hadamard-back read FAILED (chance
accuracy); the corrected RZ-on-|+> + Hadamard mechanism is the fix and is
now a regression test.

**What does NOT yet work (honest).** On the full interleaved contextual-
parity task (v0.10), the cell scores constrained-token accuracy 0.52-0.54
— barely above chance, BELOW the GRU ladder (0.58-0.66). Diagnosis: the
task interleaves n_live contexts and uses cue tokens to mark observable
identity, so the cell must accumulate parity PER CONTEXT, selecting which
qubit(s) to phase based on the cue. The current cell applies the same
phase op regardless of cue, so interleaved contexts scramble into a single
register and the per-context parities are unrecoverable. The missing
ingredient is cue-CONDITIONED routing (a learned map from cue token ->
which phase qubit), i.e. a controlled-phase keyed on the observable id.

**State of the contextuality track.** The classical memory wall is
established (v0.10) and the quantum cell's core parity-by-interference
primitive is verified (running-parity = 100%), but the routing needed to
beat the classical ladder on the interleaved task is not yet built. This
is a concrete, well-scoped next step — controlled-phase routing keyed on
cue tokens — not a dead end. Reported as partial rather than claimed.

## 18. v0.14 — cue-conditioned routing: the contextual cell, improved (still partial)

Built the routing the v0.13 diagnosis called for
(`RoutedContextualQRNN`, arch `routed_contextual`). Mechanism: a learned
soft map `cue_to_qubit` (n_cue x n_phase, softmaxed) sends each cue token
to a qubit distribution; the cell carries a running `selector` that a CUE
token REPLACES (with its qubit distribution) and a VALUE token HOLDS; the
value bit is then imprinted as an RZ phase weighted by the selector, so
parity accumulates on the active context's qubit. Readout is the verified
Hadamard-back + Born parity read.

**Measured improvement.** On the interleaved contextual task (live=3, the
v0.10 benchmark), routing lifts constrained-token accuracy to 0.569-0.579
(n_phase 5-7, 1.4k-5.5k params) from the unrouted cell's 0.52-0.54 — now
MATCHING the smallest GRU (gru8 = 0.571) at comparable parameters. In an
isolated 2-stream routed-parity control (explicit cues, two interleaved
parities) the cell reaches 0.76 value-step accuracy, well above chance and
now a regression test — confirming the routing mechanism functions.

**Still not a win (honest).** The routed cell matches small GRUs but does
NOT beat the classical ladder's larger rungs (gru32-128 = 0.66-0.88). Two
identified limits: (1) the selector is SOFT — it blends qubits rather than
cleanly partitioning streams, so contexts still partially interfere; (2)
qubit budget — n_phase must exceed n_live with margin for clean separation,
and even the 2-stream isolated test (3 qubits, 2 streams) only hit 0.76,
not ~1.0. Likely fixes for a future pass: harder (lower-temperature or
straight-through) routing, n_phase >> n_live, and a controlled-phase that
entangles the selector with the value qubit rather than weighting angles.

**Contextuality track, full state.** classical memory wall established
(v0.10) ✓; quantum parity-by-interference primitive verified at 100%
(v0.13) ✓; cue-conditioned routing built and improves results, reaching
small-GRU parity and passing an isolated routing test (v0.14) ✓; but a
clean separation BEATING the classical ladder is not yet achieved. The
track has gone from "wall + unbuilt cell" to "wall + working primitive +
partial routing", with the remaining gap precisely localized to soft-
routing fidelity and qubit budget. Reported as honest, incremental
progress on the hardest and most theoretically promising of the three
advantage tracks.

## 19. v0.15 — self-hosted dashboard (own MLflow replacement)

Built a local experiment UI over the project's own SQLite store — no
MLflow dependency. React + Vite frontend, FastAPI backend, per-step
curves logged directly into `qllm_results.db`.

**DB extensions.** Three new tables: `steps` (per-step training curves),
`live_runs` (in-flight progress registry: current/total step, status,
last loss/ppl), keeping the existing `runs`/`metrics` summary tables.
`fit()` gains opt-in dashboard logging via `TrackingConfig.dashboard_*`
fields (or the `with_dashboard(cfg, ...)` helper) — when set, every run
streams train/val curves and progress to the DB live.

**Backend** (`qllm/dashboard/server.py`, `queries.py`): FastAPI serving
`/api/suites`, `/api/suite/{name}` (leaderboard with merged metrics),
`/api/run/{id}` (config + metrics + curve), `/api/live` + `/api/live/
{key}/curve` (polled), and the existing PNG plots. SPA-fallback route so
client-side deep links resolve. Verified against the live 230-run / 13-
suite DB.

**Frontend** (React + Vite + recharts): Suites overview (cards), Suite
leaderboard (sortable table + bar chart, dataset filter chips), Run
detail (config + metrics + training curve), Live (2s-polling table with
progress bars + dual-axis live loss/ppl curve). One-command launch:
`python -m qllm.dashboard.run --port 8000`.

Bugs found and fixed during bring-up: StaticFiles 404 on deep links
(added SPA fallback) and a React hooks-order violation (#310) from a
useMemo after an early return (hoisted above the guard). Both caught by
screenshotting the running UI, not just unit tests. Roadmap (in
dashboard/README.md): compare view, live-rendered signature plots from
the DB, CSV export, remote-host auth.

## 20. v0.16 — historical two-stream v1 (rerun required)

Tested the idea "a quantum sentence transformer feeds/guides a classical
word transformer." Architecture (`qllm/models/two_stream.py`, arch
`two_stream`): a sentence encoder mean-pools the context into a small
vector s, injected into a standard causal word transformer three ways —
FiLM (per-block gain/bias from s), virtual token (s prepended as token 0),
or global bias (s added to every embedding). Encoder is quantum
(pool -> VQC -> measured s), classical (pool -> MLP -> s, param-matched to
~0.7%, slightly LARGER = conservative), or none. Run on both classical
text and quantum Ising data, strict param-matched control.

**Protocol correction (2026-07-11).** The `two-stream-v1` encoder computed one
summary from the full token window and reused it at every prediction position.
Although the word transformer itself was causal, that summary included future
tokens. Its perplexities therefore have metric type
`teacher_forced_side_information`, not strict autoregressive perplexity. The
numbers and plot below are preserved as historical provenance, but the suite is
`rerun_required` and supports no strict autoregressive conclusion.

**Historical text metric (4 seeds for bias, 2 for others).** Under the v1
side-information metric, every encoded variant recorded a lower perplexity than
none (9.95), and the quantum/classical ordering varied by conditioning mode:

| variant | val ppl (mean ± std) |
|---|---|
| quantum-bias | 9.29 ± 0.37 |
| classical-bias | 9.64 ± 0.44 |
| classical-film | 9.56 ± 0.22 |
| quantum-film | 9.70 ± 0.19 |
| quantum-token | 9.91 ± 0.13 |
| classical-token | 10.01 ± 0.11 |

Within that historical metric, quantum-bias recorded 0.35 lower perplexity
than its classical control. At 2 seeds the values were 9.01 vs 9.35; at 4
seeds the error bars overlap heavily (±0.37 vs ±0.44). Because every v1 value
uses future side information, neither this difference nor the other
conditioning-dependent orderings are evidence of a causal language-model edge.

**Historical Ising metric (2 seeds).** At the resonant theta_x=0.75, all
encoders recorded perplexity around 1.072 and the unconditioned variant recorded
1.093. These values use the same full-window side information and therefore do
not establish a strict causal conditioning benefit or a quantum/classical
separation.

**Corrected status.** `two-stream-v1` is historical, non-causal evidence only.
It establishes neither that sentence conditioning improves strict
autoregressive prediction nor that a quantum encoder beats a matched classical
encoder. The required next test is the causal-prefix `two-stream-causal-v2`
protocol with powered paired seeds, the parameter-matched classical encoder,
the no-conditioning ablation, and complete resource accounting. Historical
plot: results/two_stream.png.
