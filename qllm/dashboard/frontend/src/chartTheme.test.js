import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

import {
  chartAxisTick,
  chartGridStroke,
  chartMutedText,
  chartSeries,
  chartTooltipProps,
} from './chartTheme.js'

const sourceRoot = dirname(fileURLToPath(import.meta.url))
const chartPages = [
  'Live.jsx',
  'Comparison.jsx',
  'ResearchResults.jsx',
  'Run.jsx',
  'RunWorkspace.jsx',
  'ScalingTest.jsx',
  'Study.jsx',
  'Suite.jsx',
]
const chartVariables = [
  '--chart-axis-text',
  '--chart-grid',
  '--chart-cursor',
  '--chart-tooltip-bg',
  '--chart-tooltip-border',
  '--chart-tooltip-text',
  '--chart-series-accent',
  '--chart-series-blue',
  '--chart-series-green',
  '--chart-series-amber',
  '--chart-series-pink',
  '--band-light-bg',
  '--band-light-text',
  '--band-medium-bg',
  '--band-medium-text',
  '--band-heavy-bg',
  '--band-heavy-text',
  '--band-research-bg',
  '--band-research-text',
]

test('chart presentation tokens remain CSS-variable based and static', () => {
  assert.deepEqual(chartAxisTick, {
    fill: 'var(--chart-axis-text)',
    fontSize: 11,
  })
  assert.equal(chartGridStroke, 'var(--chart-grid)')
  assert.equal(chartMutedText, 'var(--chart-axis-text)')
  assert.ok(Object.isFrozen(chartAxisTick))
  assert.ok(Object.isFrozen(chartSeries))
  assert.ok(Object.isFrozen(chartTooltipProps))

  const serialized = JSON.stringify({
    chartAxisTick,
    chartGridStroke,
    chartMutedText,
    chartSeries,
    chartTooltipProps,
  })
  assert.doesNotMatch(serialized, /#[0-9a-f]{3,8}/i)
  assert.match(serialized, /var\(--chart-tooltip-text\)/)
})

test('owned chart pages contain no legacy dark-only chart literals', () => {
  const forbidden = /#(?:8b949e|161b22|30363d|a371f7|2f81f7|3fb950|d29922|f778ba)\b/i
  for (const page of chartPages) {
    const source = readFileSync(join(sourceRoot, 'pages', page), 'utf8')
    assert.doesNotMatch(source, forbidden, page)
    assert.match(source, /chartAxisTick/, page)
    assert.match(source, /chartTooltipProps/, page)
  }
})

test('chart and band CSS variables are defined for dark and light themes', () => {
  const styles = readFileSync(join(sourceRoot, 'styles.css'), 'utf8')
  for (const variable of chartVariables) {
    const occurrences = styles.match(new RegExp(`${variable}:`, 'g')) ?? []
    assert.equal(occurrences.length, 2, `${variable} must be defined in both themes`)
  }
  assert.match(
    styles,
    /\.quantum-band\.research\s*\{[^}]*var\(--band-research-text\)/,
  )
})
