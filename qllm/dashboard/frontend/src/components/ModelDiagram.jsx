function metaValue(value) {
  if (value == null || value === '') return null
  return String(value)
}

export default function ModelDiagram({ graph, title = 'Model architecture' }) {
  const nodes = graph?.nodes || []
  const edges = graph?.edges || []
  const nextById = Object.fromEntries(edges.map(([from, to]) => [from, to]))
  if (!nodes.length) {
    return (
      <div className="model-diagram empty-chart">
        <div className="pill">{title}</div>
        <p className="muted">Architecture graph is unavailable.</p>
      </div>
    )
  }
  return (
    <div className="model-diagram">
      <div className="diagram-head">
        <div className="pill">{title}</div>
        {graph?.summary?.uses_quantum && <span className="badge quantum">quantum components</span>}
      </div>
      <div className="diagram-chain">
        {nodes.map((node) => (
          <div key={node.id} className="diagram-step">
            <div className={`diagram-node ${node.kind}`}>
              <div className="diagram-label">{node.label}</div>
              <div className="diagram-kind">{node.kind}</div>
              <div className="diagram-meta">
                {Object.entries(node.meta || {}).map(([key, value]) => (
                  metaValue(value) && <span key={key}>{key}: {metaValue(value)}</span>
                ))}
              </div>
            </div>
            {nextById[node.id] && <div className="diagram-edge">-&gt;</div>}
          </div>
        ))}
      </div>
      {graph?.quantum && (
        <div className="quantum-summary compact-summary">
          <span className="badge">qubits {graph.quantum.n_qubits}</span>
          <span className="badge">depth {graph.quantum.n_circuit_layers}</span>
          <span className="badge">{graph.quantum.backend}</span>
          <span className="badge">{graph.quantum.shots == null ? 'analytic' : `${graph.quantum.shots} shots`}</span>
        </div>
      )}
    </div>
  )
}
