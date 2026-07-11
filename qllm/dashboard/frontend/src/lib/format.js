// Small display formatters shared across the data surfaces. Missing values read
// as an em dash, never a fabricated 0.

export const DASH = '—'

export function fmtNum(value, digits = 2) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return DASH
  return value.toLocaleString(undefined, { maximumFractionDigits: digits })
}

// Compact scientific notation for very small/large magnitudes (e.g. grad var).
export function fmtSci(value, digits = 1) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return DASH
  if (value === 0) return '0'
  const abs = Math.abs(value)
  if (abs >= 1e-3 && abs < 1e4) return fmtNum(value, 3)
  return value.toExponential(digits).replace('e', 'e')
}

// Signed percentage, e.g. -5.3%. `ratio` is a fraction (delta/base).
export function fmtPct(ratio, digits = 1) {
  if (typeof ratio !== 'number' || !Number.isFinite(ratio)) return DASH
  const sign = ratio > 0 ? '+' : ''
  return `${sign}${(ratio * 100).toFixed(digits)}%`
}

export function fmtSeconds(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return DASH
  if (value < 90) return `${value.toFixed(1)}s`
  const m = value / 60
  if (m < 90) return `${m.toFixed(1)}m`
  return `${(m / 60).toFixed(1)}h`
}
