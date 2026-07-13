import { test, expect } from '@playwright/test'
import { mockApi } from './fixtures.js'

test.beforeEach(async ({ page }) => { await mockApi(page) })

// The default mock serves the canonical backend ontology (captured verbatim
// from /api/atlas/ontology), so these tests exercise the shipped contract.
const LIVE_SWAPS_LABEL = 'Variational quantum embedding, attention, FFN, and full-block swaps'

test('Atlas list: live canonical ontology, all outcome tiles, 19 cells', async ({ page }) => {
  await page.goto('/atlas')
  // live data → no seed-fallback notice
  await expect(page.getByText(/seed ontology/i)).toHaveCount(0)
  // every outcome bucket has a summary tile, nulls included at full weight
  for (const label of ['Quantum candidate', 'Quantum-only paradigm', 'Classical holds · no advantage', 'Open · gated', 'Unexplored']) {
    await expect(page.locator('.atlas-summary-tile', { hasText: label })).toBeVisible()
  }
  await expect(page.getByText('19 of 19 cells')).toBeVisible()
  // legend spells out the four encoding channels
  await expect(page.getByText(/Claim level → border width/i)).toBeVisible()
  await expect(page.getByText(/Replication → border style/i)).toBeVisible()
})

test('Atlas falls back to the bundled seed when the live ontology is absent', async ({ page }) => {
  await mockApi(page, { '/atlas/ontology': null })
  await page.goto('/atlas')
  await expect(page.getByText(/bundled seed ontology/i)).toBeVisible()
  await expect(page.getByText('19 of 19 cells')).toBeVisible() // seed still renders the full map
})

test('Atlas filters by claim level', async ({ page }) => {
  await page.goto('/atlas')
  await page.locator('label').filter({ hasText: 'Claim' }).locator('select').selectOption('untested')
  await expect(page.getByText('19 of 19 cells')).toHaveCount(0)
  await expect(page.getByText(/of 19 cells/)).toBeVisible()
})

test('Atlas graph: 19 clickable cells; click opens detail with claim/replication distinct', async ({ page }) => {
  await page.goto('/atlas')
  await page.getByRole('button', { name: 'Graph' }).click()
  const cells = page.locator('.atlas-graph-svg g[role="button"]')
  await expect(cells).toHaveCount(19)
  await cells.first().click()
  const detail = page.locator('.atlas-side')
  await expect(detail.getByText('Claim level (map)')).toBeVisible()
  await expect(detail.getByText('Replication', { exact: true })).toBeVisible()
  await expect(detail.getByText(/no composite advantage score/i)).toBeVisible()
})

test('Atlas: ?node= deep-link selects a cell from the URL', async ({ page }) => {
  await page.goto('/atlas?node=c_variational_swaps')
  await expect(page.locator('.atlas-side').getByText(LIVE_SWAPS_LABEL)).toBeVisible()
})

test('Atlas: selecting a cell writes it to the URL (shareable)', async ({ page }) => {
  await page.goto('/atlas')
  await page.getByText('Quantum recurrent model representability and optimization').click()
  await expect(page).toHaveURL(/node=c_qrnn/)
  await expect(page.locator('.atlas-side').getByText('Claim level (map)')).toBeVisible()
})

test('Atlas: collapse all hides graph cells; expand all restores them', async ({ page }) => {
  await page.goto('/atlas')
  await page.getByRole('button', { name: 'Graph' }).click()
  const cells = page.locator('.atlas-graph-svg g[role="button"]')
  await expect(cells).toHaveCount(19)
  await page.getByRole('button', { name: 'Collapse all' }).click()
  await expect(cells).toHaveCount(0)
  await page.getByRole('button', { name: 'Expand all' }).click()
  await expect(cells).toHaveCount(19)
})

test('Atlas: list domain header collapses its cells', async ({ page }) => {
  await page.goto('/atlas')
  await expect(page.getByText(LIVE_SWAPS_LABEL)).toBeVisible()
  await page.locator('.atlas-group-toggle').first().click()
  await expect(page.getByText(LIVE_SWAPS_LABEL)).toHaveCount(0)
})

test('Atlas: graph cells are keyboard-operable', async ({ page }) => {
  await page.goto('/atlas')
  await page.getByRole('button', { name: 'Graph' }).click()
  await page.locator('.atlas-graph-svg g[role="button"]').first().focus()
  await page.keyboard.press('Enter')
  await expect(page).toHaveURL(/node=/)
})

test('Atlas: a classical-holds (null) cell is styled with the classical, non-green token', async ({ page }) => {
  await page.goto('/atlas')
  const chip = page.locator('.atlas-oc-classical_holds').first()
  await expect(chip).toBeVisible()
  const color = await chip.evaluate((el) => getComputedStyle(el).color)
  // classical blue, never the good/green token
  expect(color).not.toBe('rgb(63, 191, 63)')
})

// ---- research-map upgrade (force-cluster layout, territories, routes, zoom) --

test('Atlas map: zoom toolbar zooms and resets (data-zoom reflects the level)', async ({ page }) => {
  await page.goto('/atlas')
  await page.getByRole('button', { name: 'Graph' }).click()
  const frame = page.locator('.atlas-map-frame')
  await expect(frame).toHaveAttribute('data-zoom', '1.00')
  await page.locator('.atlas-zoom-in').click()
  await expect(frame).not.toHaveAttribute('data-zoom', '1.00')
  await page.locator('.atlas-zoom-reset').click()
  await expect(frame).toHaveAttribute('data-zoom', '1.00')
})

test('Atlas map: selection ring never overwrites the claim/replication border', async ({ page }) => {
  await page.goto('/atlas?node=c_variational_swaps')
  await page.getByRole('button', { name: 'Graph' }).click()
  await expect(page.locator('.atlas-sel-ring')).toHaveCount(1)
  const cell = page.locator('[data-cell-id="c_variational_swaps"]')
  await expect(cell).toHaveAttribute('aria-pressed', 'true')
  // the outcome/claim border on the face survives selection (integrity fix)
  const stroke = await cell.locator('.atlas-cell-face').evaluate((el) => getComputedStyle(el).stroke)
  const ring = await page.locator('.atlas-sel-ring').evaluate((el) => getComputedStyle(el).stroke)
  expect(stroke).not.toBe(ring)
})

test('Atlas map: domain territories render; collapse swaps hulls for inert seals', async ({ page }) => {
  await page.goto('/atlas')
  await page.getByRole('button', { name: 'Graph' }).click()
  await expect(page.locator('.atlas-hull')).toHaveCount(6)
  await page.getByRole('button', { name: 'Collapse all' }).click()
  await expect(page.locator('.atlas-hull')).toHaveCount(0)
  await expect(page.locator('.atlas-seal')).toHaveCount(6)
  // seals are inert — the role=button count contract stays 0 when collapsed
  await expect(page.locator('.atlas-graph-svg g[role="button"]')).toHaveCount(0)
  await page.getByRole('button', { name: 'Expand all' }).click()
  await expect(page.locator('.atlas-hull')).toHaveCount(6)
})

test('Atlas map: typed routes render with dash + terminal semantics', async ({ page }) => {
  await page.goto('/atlas')
  await page.getByRole('button', { name: 'Graph' }).click()
  const routes = page.locator('.atlas-route')
  await expect(routes).toHaveCount(10) // the canonical ontology's relations
  // association-like routes are dashed with a chevron
  const dashed = page.locator('.atlas-route[data-relation="motivates"]').first()
  await expect(dashed).toHaveAttribute('stroke-dasharray', '6 4')
  // constraint-like routes are solid with an inhibition bar
  const solid = page.locator('.atlas-route[data-relation="constrains"]').first()
  await expect(solid).not.toHaveAttribute('stroke-dasharray', /./)
  await expect(solid).toHaveAttribute('marker-end', 'url(#atlas-bar)')
})

test('Atlas map: null-outcome cards keep full resting prominence in the new skin', async ({ page }) => {
  await page.goto('/atlas')
  await page.getByRole('button', { name: 'Graph' }).click()
  const face = page.locator('.atlas-node-oc-classical_holds .atlas-cell-face').first()
  await expect(face).toBeVisible()
  const { stroke, opacity } = await face.evaluate((el) => {
    const cs = getComputedStyle(el)
    return { stroke: cs.stroke, opacity: cs.opacity }
  })
  expect(opacity).toBe('1') // never dimmed by default
  expect(stroke).not.toBe('rgb(63, 191, 63)') // never green
})

test('Atlas map: keyboard zoom on the frame; Enter on a cell still selects', async ({ page }) => {
  await page.goto('/atlas')
  await page.getByRole('button', { name: 'Graph' }).click()
  const frame = page.locator('.atlas-map-frame')
  await frame.focus()
  await page.keyboard.press('+')
  await expect(frame).not.toHaveAttribute('data-zoom', '1.00')
  await page.keyboard.press('0')
  await expect(frame).toHaveAttribute('data-zoom', '1.00')
  // frame-level keys must not swallow cell activation
  await page.locator('.atlas-graph-svg g[role="button"]').first().focus()
  await page.keyboard.press('Enter')
  await expect(page).toHaveURL(/node=/)
})
