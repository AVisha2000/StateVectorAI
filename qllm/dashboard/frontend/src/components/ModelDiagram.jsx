import { useMemo, useState } from 'react'

function metaValue(value) {
  if (value == null || value === '') return null
  if (typeof value === 'object') return null
  return String(value)
}

function visibleNodes(graph, levelId) {
  const nodes = graph?.nodes || []
  const level = (graph?.levels || []).find((item) => item.id === levelId)
  if (!level || level.id === 'overview') return nodes
  const ids = new Set(level.node_ids || [])
  return nodes.filter((node) => ids.has(node.id))
}

function filteredEdges(edges, nodes) {
  const ids = new Set(nodes.map((node) => node.id))
  return (edges || []).filter(([from, to]) => ids.has(from) && ids.has(to))
}

function NodeDetail({ node }) {
  if (!node) return <p className="muted">Select a component to inspect its config path and resources.</p>
  const meta = node.meta || {}
  const resource = meta.resource
  return (
    <div className="diagram-detail">
      <div>
        <div className="pill">{node.label}</div>
        <h3>{meta.component_type || node.kind}</h3>
      </div>
      <div className="kv compact">
        <div className="k">kind</div><div className="v">{node.kind}</div>
        <div className="k">config path</div><div className="v">{meta.config_path || '-'}</div>
        <div className="k">block</div><div className="v">{meta.block_index == null ? '-' : Number(meta.block_index) + 1}</div>
        <div className="k">qubits</div><div className="v">{resource?.n_qubits ?? meta.n_qubits ?? '-'}</div>
        <div className="k">depth</div><div className="v">{resource?.n_circuit_layers ?? meta.circuit_depth ?? '-'}</div>
        <div className="k">backend</div><div className="v">{resource?.backend || '-'}</div>
        <div className="k">shots</div><div className="v">{resource?.shots == null ? 'analytic' : resource.shots}</div>
      </div>
      {node.circuit?.template?.length > 0 && (
        <div className="circuit-list compact-circuit">
          {node.circuit.template.slice(0, 6).map((gate, index) => (
            <div key={`${gate.gate}-${index}`} className="circuit-gate">
              <b>{gate.gate}</b>
              <span>{gate.layer ? `layer ${gate.layer}` : 'input'} {gate.wires ? `wires ${gate.wires.join(',')}` : gate.pattern || ''}</span>
              <span className={`badge ${gate.trainable ? 'done' : ''}`}>{gate.trainable ? 'trainable' : 'fixed'}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function ModelDiagram({
  graph,
  title = 'Model architecture',
  initialLevel = 'overview',
  showControls = true,
  onSelectNode,
}) {
  const levels = graph?.levels?.length
    ? graph.levels
    : [{ id: 'overview', label: 'Overview', node_ids: (graph?.nodes || []).map((node) => node.id) }]
  const [levelId, setLevelId] = useState(initialLevel)
  const [selectedId, setSelectedId] = useState(null)
  const nodes = graph?.nodes || []
  const shownNodes = useMemo(() => visibleNodes(graph, levelId), [graph, levelId])
  const shownEdges = useMemo(() => filteredEdges(graph?.edges, shownNodes), [graph, shownNodes])
  const nextById = Object.fromEntries(shownEdges.map(([from, to]) => [from, to]))
  const selectedNode = nodes.find((node) => node.id === selectedId) || shownNodes[0]
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
      {showControls && levels.length > 1 && (
        <div className="diagram-controls">
          {levels.map((level) => (
            <button
              key={level.id}
              type="button"
              className={`chip ${levelId === level.id ? 'on' : ''}`}
              onClick={() => { setLevelId(level.id); setSelectedId(null) }}
              disabled={(level.node_ids || []).length === 0}
              title={level.description}
            >
              {level.label}
            </button>
          ))}
        </div>
      )}
      <div className="diagram-chain">
        {shownNodes.map((node) => (
          <div key={node.id} className="diagram-step">
            <button
              type="button"
              className={`diagram-node ${node.kind} ${selectedNode?.id === node.id ? 'selected' : ''}`}
              onClick={() => { setSelectedId(node.id); onSelectNode?.(node) }}
            >
              <div className="diagram-label">{node.label}</div>
              <div className="diagram-kind">{node.kind}</div>
              <div className="diagram-meta">
                {Object.entries(node.meta || {}).map(([key, value]) => (
                  metaValue(value) && <span key={key}>{key}: {metaValue(value)}</span>
                ))}
              </div>
            </button>
            {nextById[node.id] && <div className="diagram-edge">-&gt;</div>}
          </div>
        ))}
        {shownNodes.length === 0 && <p className="muted">No components at this level yet.</p>}
      </div>
      {showControls && <NodeDetail node={selectedNode} />}
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
