import { useMemo } from 'react'
import { columns } from '../lib/circuitModel.js'

// Hand-authored SVG circuit diagram (renders reliably everywhere). Draws qubit
// wires and the gate layout derived from the ansatz family — a read view of the
// registry-runnable circuit, not a free-form editor pretending to round-trip.
const COL_W = 52
const ROW_H = 46
const PAD_L = 58
const PAD_T = 26
const BOX_W = 34
const BOX_H = 26

export default function CircuitSvg({ circuit }) {
  const cols = useMemo(() => columns(circuit), [circuit])
  const n = circuit?.n_qubits || 0
  const gates = circuit?.gates || []
  const colIndex = useMemo(() => new Map(cols.map((c, i) => [c, i])), [cols])

  const width = PAD_L + cols.length * COL_W + 24
  const height = PAD_T + n * ROW_H + 16
  const xOf = (col) => PAD_L + colIndex.get(col) * COL_W + COL_W / 2
  const yOf = (q) => PAD_T + q * ROW_H + ROW_H / 2

  return (
    <div className="circuit-wrap card">
      <svg viewBox={`0 0 ${Math.max(width, 200)} ${Math.max(height, 120)}`} width="100%" height={Math.max(height, 120)}
        role="img" preserveAspectRatio="xMinYMin meet" aria-label={`Quantum circuit: ${circuit?.ansatz}, ${n} qubits, depth ${circuit?.depth}`}>
        {/* wires + qubit labels */}
        {Array.from({ length: n }, (_, q) => (
          <g key={`wire-${q}`}>
            <line x1={PAD_L - 30} y1={yOf(q)} x2={width - 12} y2={yOf(q)} stroke="var(--axis)" strokeWidth={1.5} />
            <text x={16} y={yOf(q) + 4} fontSize="11" fill="var(--ink2)" className="mono">q{q}</text>
          </g>
        ))}
        {/* gates */}
        {gates.map((g) => {
          const x = xOf(g.col)
          if (g.kind === 'two') {
            const yc = yOf(g.control)
            const yt = yOf(g.qubit)
            return (
              <g key={g.id}>
                <line x1={x} y1={yc} x2={x} y2={yt} stroke="var(--q)" strokeWidth={1.6} />
                <circle cx={x} cy={yc} r={4} fill="var(--q)" />
                {g.type === 'CNOT' ? (
                  <g>
                    <circle cx={x} cy={yt} r={9} fill="none" stroke="var(--q)" strokeWidth={1.6} />
                    <line x1={x - 9} y1={yt} x2={x + 9} y2={yt} stroke="var(--q)" strokeWidth={1.6} />
                    <line x1={x} y1={yt - 9} x2={x} y2={yt + 9} stroke="var(--q)" strokeWidth={1.6} />
                  </g>
                ) : (
                  <g>
                    <rect x={x - BOX_W / 2} y={yt - BOX_H / 2} width={BOX_W} height={BOX_H} rx={5}
                      fill="var(--q-soft)" stroke="var(--q)" strokeWidth={1.4} />
                    <text x={x} y={yt + 4} textAnchor="middle" fontSize="10" fill="var(--q)">{g.type}</text>
                  </g>
                )}
              </g>
            )
          }
          const y = yOf(g.qubit)
          return (
            <g key={g.id}>
              <rect x={x - BOX_W / 2} y={y - BOX_H / 2} width={BOX_W} height={BOX_H} rx={5}
                fill="var(--surface2)" stroke="var(--accent)" strokeWidth={1.2} />
              <text x={x} y={y + 4} textAnchor="middle" fontSize="10" fill="var(--ink)">{g.type}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
