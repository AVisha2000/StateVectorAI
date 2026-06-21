# GPU run queue

Every command below addresses something explicitly deferred for lack of
GPU compute across v0.1–v0.16 (see the matching RESULTS.md section for
full context on each). Ordered by priority — top items are the highest-
value, most decisive experiments. All scripts are resumable: re-running a
command skips finished `(suite, variant, dataset, seed, steps)` cells, so
you can stop/restart freely and run these across multiple sessions.

Each command's dashboard logging is OFF by default (matches the existing
CPU-era scripts). Add `--dashboard` where shown (a small flag I've added
to the priority scripts) to stream live curves to `qllm_results.db` and
watch them in the UI (`python -m qllm.dashboard.run`) while they run.

## 1. The separation flagship (RESULTS §13, highest priority)

The whole project's central question: does classical memory cost diverge
from the quantum filter's O(m) qubits as memory size m grows? Established
on CPU through m=10; this is the experiment GPU access was specifically
acquired for.

```bash
python benchmarks/memory_sweep.py \
  --memory-qubits 6 8 10 12 14 16 \
  --models planted gru16 gru32 gru64 gru128 gru256 gru512 \
  --seeds 0 1 2 \
  --dashboard
python benchmarks/plot_separation.py
```
Estimated cost: m=16 data generation is the long pole (gate-wise, ~30x
faster than dense, but still O(2^17) per token); budget a few hours for
the full grid at 3 seeds. Run m=12/14 first if you want a faster checkpoint.

## 2. Extend the resonance ridge (RESULTS §13, feeds #1)

Confirms the theta_x=0.75 ridge (gap 0.67->0.43 at m=6->12) holds at the
m values #1 needs, BEFORE spending compute training there. Training-free
(planted filter only), so cheap — run this first if you haven't already
confirmed m=14/16 specifically.

```bash
python benchmarks/resonance_search.py --memory-qubits 14 16 \
  --theta-x 0.7 0.75 0.785 --spt 2
```

## 3. QRNN optimization-landscape, properly (RESULTS §13)

On CPU, ALL 12 random-init runs converged to the same bad basin (~1.61)
regardless of lr/depth/seed — never reaching the planted optimum (1.527).
With GPU throughput, sweep wider and longer to determine if this is a
true local-optimum trap or just needs more compute:

```bash
python benchmarks/qrnn_landscape.py \
  --variants L2-lr0.003 L2-lr0.01 L2-lr0.03 L3-lr0.003 L3-lr0.01 L3-lr0.03 \
             planted-ft planted-noise0.3-ft \
  --seeds 0 1 2 3 4 5 6 7 \
  --steps 5000
```
Add new variants for wider lr/depth coverage if the existing grid still
plateaus (e.g. `L4-lr0.1`, warmup schedules — these aren't wired in
`GRID` in `qrnn_landscape.py` yet, add them there first).

## 4. Exact-ansatz QRNN, full horizon (RESULTS §11/§13)

The exact-representability (Ising-form) QRNN plateaued at 1.637 after
2000 steps on CPU. Confirm whether more steps closes the gap to the
planted floor (1.527) given #3's landscape findings:

```bash
python benchmarks/recurrent_floor.py --dataset ising --models qrnn \
  --steps 10000 --seeds 0 1 2
python benchmarks/recurrent_floor.py --dataset markov --models qrnn \
  --steps 10000 --seeds 0 1 2   # the classical-twin control
```

## 5. Contextual cell routing, with real qubit budget (RESULTS §17/§18)

The routed contextual cell matched small GRUs (0.57) but not the ladder's
top end (0.66-0.88), with two named limits: soft (not hard) routing, and
qubit budget. GPU lets you test both at once — wider n_phase and many more
training steps:

```bash
python benchmarks/contextual_sweep.py --live 2 3 4 5 6 \
  --models rqrnn8 rqrnn10 rqrnn12 gru32 gru64 gru128 gru256 \
  --seeds 0 1 2 --steps 8000
python benchmarks/plot_contextual.py
```
If routing still lags, the next code change (not yet built) is a
straight-through/hard-argmax selector instead of the softmax — flag this
back if the soft version caps out, and I'll build the hard-routing variant.

## 6. Two-stream: settle the quantum-bias lean (RESULTS §20)

The most statistically fragile result in the project: quantum-bias led
its param-matched classical control 9.29 vs 9.64 ppl over 4 seeds, with
heavily overlapping error bars. This needs real seed count to know if
it's signal or noise:

```bash
python benchmarks/two_stream_probe.py --dataset text \
  --variants quantum-bias classical-bias \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --steps 3000 --dashboard
```
Also worth the contrast at higher capacity (does the lean persist if both
streams get bigger?):
```bash
python benchmarks/two_stream_probe.py --dataset text \
  --variants quantum-bias classical-bias --seeds 0 1 2 3 4 5 \
  --steps 3000   # then edit BODY/d_sent in the script to 2x and rerun
```

## 7. QNLP suite at real depth (RESULTS §10)

The original comprehensive suite ran 300 steps / 2 seeds on CPU — enough
to see the monotone "more quantum = worse on text" trend, not enough to
trust small gaps (e.g. q-attn-qkv tying classical). Rerun at the depth
originally planned:

```bash
python benchmarks/qnlp_suite.py --dataset text --steps 5000 --seeds 0 1 2
python benchmarks/qnlp_suite.py --dataset ising --steps 5000 --seeds 0 1 2
python benchmarks/plot_suite.py --dataset text
python benchmarks/plot_suite.py --dataset ising
```

## 8. Transplant: more seeds, deeper compile (RESULTS §11/§12)

transplant-v2's causal control (lr-warm == q-warm) was solid at 3 seeds;
worth confirming it holds with more seeds and at higher compile depth
(L=20 already gets fidelity ~0.985; push to see if it matters):

```bash
python benchmarks/weight_transplant.py --seeds 0 1 2 3 4 5 6 7 \
  --layers 24 --compile-steps 1200
```

## 9. Interference head: does the gap widen at higher arity? (RESULTS §15/§16)

The head-only probe held exactly at the floor through k=3. Push further
(k=4,5,6) to see if the interference-vs-mixture gap grows, shrinks, or
plateaus with more seeds for tighter error bars:

```bash
python benchmarks/interference_head_probe.py --features 1 2 3 4 5 6 \
  --seeds 0 1 2 3 4 --steps 4000
```

---

## After the queue: regenerate the consolidated plots

```bash
python benchmarks/plot_separation.py
python benchmarks/plot_contextual.py
python benchmarks/plot_suite.py --dataset text
python -m qllm.dashboard.run --port 8000   # browse everything
```

Report back whichever results land — especially #1 and #6, the two
genuinely open questions this project has left on the table.
