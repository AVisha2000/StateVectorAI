// Small shared presentational helpers for the redesigned surfaces.

export function PageHeader({ title, sub, actions }) {
  return (
    <div className="row" style={{ alignItems: 'flex-start' }}>
      <div>
        <h1 className="h1">{title}</h1>
        {sub ? <p className="sub">{sub}</p> : null}
      </div>
      {actions ? <div className="row spacer">{actions}</div> : null}
    </div>
  )
}

export function Loading({ label = 'Loading…' }) {
  return <div className="state">{label}</div>
}

export function ErrorState({ error, label = 'Could not load this data.' }) {
  const detail = error?.message || String(error || '')
  return (
    <div className="state err">
      {label}
      {detail ? <div className="hint" style={{ marginTop: 6 }}>{detail}</div> : null}
      <div className="hint" style={{ marginTop: 6 }}>
        Is the dashboard API running? Start it and this will refresh automatically.
      </div>
    </div>
  )
}

export function Empty({ children }) {
  return <div className="state">{children}</div>
}

const STATUS_TAG = {
  running: 'good',
  done: 'plain',
  queued: 'plain',
  error: 'crit',
  cancelled: 'plain',
}

export function StatusTag({ status }) {
  const cls = STATUS_TAG[status] || 'plain'
  const label = status === 'error' ? 'failed' : status
  return <span className={`tag ${cls}`}>{status === 'running' ? '● ' : ''}{label}</span>
}

// A short, honest "planned surface" scaffold used until a surface is wired.
export function Scaffold({ phase, blurb, bullets }) {
  return (
    <div className="scaffold">
      <h3>{phase}</h3>
      <p style={{ margin: 0, lineHeight: 1.6 }}>{blurb}</p>
      {bullets?.length ? (
        <ul>{bullets.map((b) => <li key={b}>{b}</li>)}</ul>
      ) : null}
      <p className="hint" style={{ marginTop: 12 }}>
        Design reference: <code>docs/design/statevector-cockpit-v3.html</code> · plan:
        {' '}<code>docs/UI_REDESIGN_PLAN.md</code>
      </p>
    </div>
  )
}
