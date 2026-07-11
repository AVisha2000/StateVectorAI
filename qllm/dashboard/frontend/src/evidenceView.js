export function stableValue(value) {
  if (Array.isArray(value)) return `[${value.map(stableValue).join(',')}]`
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableValue(value[key])}`).join(',')}}`
  }
  return JSON.stringify(value)
}

export function uniqueWarnings(warnings = []) {
  const seen = new Set()
  return warnings.filter((warning) => {
    if (!warning || typeof warning !== 'object') return false
    const key = stableValue({
      code: warning.code,
      severity: warning.severity,
      title: warning.title,
      message: warning.message,
      evidence: warning.evidence,
    })
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

export function displayValue(value, fallback = 'unavailable') {
  if (value == null || value === '') return fallback
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

export function capabilityRows(capabilities) {
  if (!capabilities || typeof capabilities !== 'object') return []
  return Object.entries(capabilities).flatMap(([component, payload]) => {
    if (!payload || typeof payload !== 'object') return [{ component, capability: 'record', status: displayValue(payload) }]
    const entries = payload.capabilities && typeof payload.capabilities === 'object'
      ? payload.capabilities
      : payload
    const metadata = new Set(['schema_version', 'backend', 'device', 'mode', 'exactness', 'result_kind'])
    const rows = Object.entries(entries)
      .filter(([name]) => !metadata.has(name))
      .map(([capability, value]) => ({
        component,
        capability,
        status: displayValue(value?.status ?? value?.supported ?? value),
        exactness: displayValue(value?.exactness ?? payload.exactness, 'unavailable'),
      }))
    return rows.length ? rows : [{ component, capability: 'implementation', status: displayValue(payload.mode ?? payload.backend), exactness: displayValue(payload.exactness) }]
  })
}
