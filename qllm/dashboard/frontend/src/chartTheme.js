export const chartAxisTick = Object.freeze({
  fill: 'var(--chart-axis-text)',
  fontSize: 11,
})

export const chartGridStroke = 'var(--chart-grid)'
export const chartMutedText = 'var(--chart-axis-text)'

export const chartSeries = Object.freeze({
  accent: 'var(--chart-series-accent)',
  blue: 'var(--chart-series-blue)',
  green: 'var(--chart-series-green)',
  amber: 'var(--chart-series-amber)',
  pink: 'var(--chart-series-pink)',
})

export const chartTooltipProps = Object.freeze({
  contentStyle: Object.freeze({
    backgroundColor: 'var(--chart-tooltip-bg)',
    border: '1px solid var(--chart-tooltip-border)',
    borderRadius: 8,
    color: 'var(--chart-tooltip-text)',
    fontFamily: 'var(--mono)',
    fontSize: 12,
  }),
  labelStyle: Object.freeze({
    color: 'var(--chart-tooltip-text)',
  }),
  cursor: Object.freeze({
    stroke: 'var(--chart-cursor)',
    strokeDasharray: '3 3',
  }),
})
