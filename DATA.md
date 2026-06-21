# Data strategy for quantum-advantage search (v0.4)

## Why data, not architecture

v0.3 showed: the quantum feature map has advantage *room* (g ≫ 1) but on
classical text the labels never use it — quantum layers degenerate to
random features regardless of architecture. The search variable is data.

## The menu (ranked by theoretical grounding)

| domain | why it could be quantum-favored | status |
|---|---|---|
| **Monitored quantum dynamics** (measurement records; quantum HMMs) | unmeasured qubits = literal quantum memory; quantum stochastic processes provably need less memory than any classical generator (quantum ε-machines line of work) | **IMPLEMENTED** (`monitored_ising_sequences`) |
| **Matched Markov twin** | order-k chain fitted TO the quantum corpus: same k-grams, deeper correlations destroyed — the mandatory control | **IMPLEMENTED** (`markov_control_sequences`) |
| Cryptographic PRG sequences (modexp / Blum-Micali) | provable classical hardness (DLP); Shor-structured feature maps are the known quantum win (Liu-Arunachalam-Temme 2021) | roadmap (both-fail control at simulable scale) |
| Classical shadows / state-property data | learning-from-experiments advantages (Huang et al. Science 2022) — NB: provable versions need quantum access, ours is classical-out | roadmap |
| Classical text | established negative baseline | baseline |

## The screening protocol (run BEFORE training)

1. **Predictability gate**: conditional entropies H_k. If H_k ≈ log2(V)
   (self-dual chaotic point!) there is nothing to learn for anyone; if
   H_1 ≈ H_∞ the structure is short-range and classical.
2. **Long-memory gate**: H_3 − H_7 on the data vs its Markov twin. The
   twin's gap is pure finite-sample bias; the excess is real
   beyond-k-gram structure.
3. **Kernel screen** (`screen_sequence_dataset`): g_min (advantage room)
   and s_ratio = s_C/s_Q (do the actual targets live in the
   quantum-favored subspace?).

## v0.4 measured results (monitored kicked Ising, 6 qubits)

Regime scan: measuring 2/6 qubits every period over-monitors (no
long-range structure at any θ_x); the self-dual point θ_x=π/4 is
information-theoretically unpredictable (H=2.0 bits = ppl ceiling 4.0,
confirmed in training). **Resonant regime: measure 1 qubit every 2
Floquet periods, θ_x=0.6** → H3−H7 = 0.121 bits vs twin's 0.001.

Information floors: ideal ≈ 2^0.669 = **1.59 ppl**; any order-3 model ≥
2^0.79 ≈ **1.73 ppl**.

| dataset | quantum-trained | classical-matched | floor |
|---|---|---|---|
| ising (resonant) | 1.592 ± 0.000 | 1.592 ± 0.003 | 1.59 ideal |
| markov-3 twin | 1.727 ± 0.009 | 1.725 ± 0.011 | 1.73 3-gram |

Reading: both transformers FULLY extract the quantum-memory bits (hit
the ideal floor, beat the 3-gram bound by exactly the predicted 0.12
bits/token); the twin confirms the structure was real. But no
quantum-vs-classical separation: a 5-qubit memory's belief-state filter
is a ≤1024-real-dim (effectively much smaller) classical object — easily
learned by a 50k-param classical FFN.

## What separation would require (the v0.5 thesis)

1. **Grow the quantum memory** m so the classical belief-state filter
   (dim ~ 4^m) outpaces classical capacity while the generating process
   stays m qubits. Needs TensorCircuit-NG MPS generation for m ≳ 10.
2. **Match the model class to the process**: our quantum layer is
   feed-forward per token; the data-generating process is *recurrent
   quantum memory*. The right architecture is a quantum recurrent cell
   whose qubit state persists across tokens — it can represent the
   generator natively in m qubits where classical RNNs need 4^m.
3. Keep every result gated by: predictability + long-memory screens,
   the Markov twin, parameter matching, and the frozen-circuit control.
