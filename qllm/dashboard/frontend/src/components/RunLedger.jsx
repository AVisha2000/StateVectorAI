import { capabilityRows, displayValue } from '../evidenceView'

function Row({ label, value, mono = false }) {
  return <><div className="k">{label}</div><div className={`v evidence-value ${mono ? 'mono' : ''}`}>{displayValue(value)}</div></>
}

export default function RunLedger({ manifest, durability, resourceLedger, backendCapabilities, title = 'Run durability and resource ledger' }) {
  const identity = durability?.immutable_identity || {}
  const checkpoint = durability?.checkpoint || {}
  const recovery = durability?.recovery || {}
  const worker = durability?.worker || {}
  const timing = resourceLedger?.timing || {}
  const device = resourceLedger?.execution_device || {}
  const precision = resourceLedger?.precision || {}
  const memory = resourceLedger?.memory || resourceLedger?.peak_memory || {}
  const capabilities = capabilityRows(backendCapabilities)
  return (
    <section className="panel run-ledger">
      <div className="workspace-header"><h3>{title}</h3><span className="badge">{durability?.status || 'pending'}</span></div>
      <div className="ledger-grid">
        <div className="kv compact">
          <h4>Immutable identity</h4>
          {['experiment_uuid', 'run_uuid', 'manifest_hash', 'config_hash', 'code_hash', 'data_hash', 'environment_hash', 'seed_axes_hash'].map((key) => <Row key={key} label={key.replaceAll('_', ' ')} value={identity[key] ?? manifest?.[key]} mono />)}
        </div>
        <div className="kv compact">
          <h4>Checkpoint and recovery</h4>
          <Row label="latest checkpoint" value={checkpoint.latest} mono />
          <Row label="best checkpoint" value={checkpoint.best} mono />
          <Row label="resume from" value={checkpoint.resume_from} mono />
          <Row label="completed step" value={checkpoint.completed_step} />
          <Row label="attempt / recovery" value={recovery.attempt_count == null && recovery.recovery_count == null ? null : `${displayValue(recovery.attempt_count)} / ${displayValue(recovery.recovery_count)}`} />
          <Row label="parent run UUID" value={recovery.parent_run_uuid} mono />
          <Row label="worker / heartbeat" value={worker.id == null && worker.heartbeat_ts == null ? null : `${displayValue(worker.id)} / ${displayValue(worker.heartbeat_ts)}`} mono />
          <Row label="lease expires" value={worker.lease_expires_ts} mono />
        </div>
        <div className="kv compact">
          <h4>Recorded resources</h4>
          <Row label="compile + first step" value={timing.compile_plus_first_executed_train_step_seconds ?? timing.compile_seconds} />
          <Row label="steady-state time" value={timing.steady_state_train_step_seconds_total ?? timing.steady_state_seconds} />
          <Row label="fit wall time" value={timing.fit_wall_seconds ?? resourceLedger?.wall_seconds} />
          <Row label="parameters" value={resourceLedger?.parameters?.value ?? resourceLedger?.n_params} />
          <Row label="state dimension" value={resourceLedger?.state_dimension?.value ?? resourceLedger?.state_dim} />
          <Row label="circuit calls" value={resourceLedger?.circuit_calls == null ? null : `${resourceLedger.circuit_calls} (${resourceLedger.circuit_calls_kind || 'kind unavailable'})`} />
          <Row label="device" value={device.resolved?.platform ?? device.requested ?? resourceLedger?.device} />
          <Row label="backend" value={resourceLedger?.quantum_backend?.actual ?? resourceLedger?.quantum_backend?.configured ?? resourceLedger?.backend} />
          <Row label="precision" value={precision.parameter_dtypes ?? precision.dtype ?? resourceLedger?.precision} />
          <Row label="peak memory" value={resourceLedger?.peak_memory_bytes ?? memory.peak_bytes} />
          <Row label="available memory" value={resourceLedger?.available_memory_bytes ?? memory.available_bytes} />
          <Row label="memory status" value={memory.status ?? resourceLedger?.peak_memory_status} />
        </div>
      </div>
      <h4>Backend capabilities</h4>
      {capabilities.length ? (
        <div className="table-scroll"><table><thead><tr><th>Component</th><th>Capability</th><th>Status</th><th>Exactness</th></tr></thead><tbody>{capabilities.map((row, index) => <tr key={`${row.component}-${row.capability}-${index}`}><td>{row.component}</td><td>{row.capability}</td><td>{row.status}</td><td>{row.exactness || 'unavailable'}</td></tr>)}</tbody></table></div>
      ) : <p className="muted">Backend capability metadata is pending or unavailable.</p>}
      <details><summary>Full manifest and resource ledger</summary><pre className="code-block evidence-raw">{JSON.stringify({ manifest: manifest ?? null, resource_ledger: resourceLedger ?? null }, null, 2)}</pre></details>
    </section>
  )
}
