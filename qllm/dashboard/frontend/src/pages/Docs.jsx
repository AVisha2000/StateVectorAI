export default function Docs() {
  return (
    <div>
      <h1>Docs</h1>
      <h2>What QLLM Lab is running and how to read the numbers.</h2>

      <section className="panel docs">
        <h3>Model presets</h3>
        <p><b>Classical presets</b> are the control group: transformer or GRU models with no quantum circuit.</p>
        <p><b>Quantum FFN / Attention</b> swap one transformer component for a variational quantum circuit while keeping the rest of the pipeline comparable.</p>
        <p><b>QRNN</b> uses quantum recurrent memory. Treat it as a slower research probe, not a guaranteed improvement.</p>
        <p><b>Two-stream quantum bias</b> quantum-encodes each cumulative token prefix and applies a causal, per-position conditioning signal. Historical full-window runs are labeled rerun-required.</p>
      </section>

      <section className="panel docs">
        <h3>Metrics</h3>
        <p><b>train_loss</b>: cross-entropy on sampled training batches. It should generally go down.</p>
        <p><b>val_loss</b>: cross-entropy on held-out validation batches. This is the main generalization signal.</p>
        <p><b>val_ppl</b>: perplexity, computed from validation loss. Lower is better and easier to compare across runs.</p>
        <p><b>val_bpc</b>: bits per character. Lower means the model is assigning better probabilities to the next character.</p>
        <p><b>grad_norm_ratio</b>: quantum circuit gradient norm divided by classical gradient norm. Very tiny values can mean the circuit is not receiving useful signal.</p>
        <p><b>n_params</b>: trainable parameter count. Use this to avoid giving one model an unfair capacity advantage.</p>
        <p><b>wall_seconds</b>: elapsed training time. Quantum simulations can be much slower even with fewer parameters.</p>
      </section>

      <section className="panel docs">
        <h3>Dataset assumptions</h3>
        <p>Hugging Face imports are converted into one local text corpus and trained with the existing character-level language modeling path.</p>
        <p>For v1, imports are public only. Use a split and text column that contain natural text, not labels or metadata.</p>
      </section>

      <section className="panel docs">
        <h3>Interpreting quantum results</h3>
        <p>A lower perplexity from a quantum preset is interesting, but it is not an advantage claim by itself. Compare against parameter-matched classical controls, multiple seeds, and wall time.</p>
        <p>Good signs: stable validation gains, nonzero circuit gradients, and improvement that persists across seeds. Bad signs: overlapping error bars, frozen-circuit parity, or much slower runs for the same quality.</p>
      </section>

      <section className="panel docs">
        <h3>Run Workspace</h3>
        <p>Jobs launched from the Run tab open into a workspace with live curves, preset explanation, dataset provenance, artifacts, and final metrics.</p>
        <p>Quantum and hybrid presets can automatically queue a linked classical twin. The comparison panel reports candidate minus baseline, so negative loss, perplexity, bpc, or wall time deltas are better.</p>
      </section>

      <section className="panel docs">
        <h3>GPU mode</h3>
        <p>The GPU target is guarded by the status API. If JAX only reports CPU devices, GPU queueing is blocked before training starts.</p>
        <p>Use the GPU tab and GPU_SETUP.md to install and verify a CUDA-enabled JAX wheel. The UI reports the same JAX devices the worker will use.</p>
      </section>
    </div>
  )
}
