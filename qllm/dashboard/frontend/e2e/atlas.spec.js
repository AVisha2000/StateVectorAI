import { test, expect } from '@playwright/test'
import { mockApi } from './fixtures.js'

test.beforeEach(async ({ page }) => { await mockApi(page) })

test('Atlas list: seed-ontology notice, all outcome tiles, 19 cells', async ({ page }) => {
  await page.goto('/atlas')
  await expect(page.getByText(/frontend seed ontology/i)).toBeVisible()
  // every outcome bucket has a summary tile, nulls included at full weight
  for (const label of ['Quantum candidate', 'Quantum-only paradigm', 'Classical holds · no advantage', 'Open · gated', 'Unexplored']) {
    await expect(page.locator('.atlas-summary-tile', { hasText: label })).toBeVisible()
  }
  await expect(page.getByText('19 of 19 cells')).toBeVisible()
  // legend spells out the four encoding channels
  await expect(page.getByText(/Claim level → border width/i)).toBeVisible()
  await expect(page.getByText(/Replication → border style/i)).toBeVisible()
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

test('Atlas: a classical-holds (null) cell is styled with the classical, non-green token', async ({ page }) => {
  await page.goto('/atlas')
  const chip = page.locator('.atlas-oc-classical_holds').first()
  await expect(chip).toBeVisible()
  const color = await chip.evaluate((el) => getComputedStyle(el).color)
  // classical blue, never the good/green token
  expect(color).not.toBe('rgb(63, 191, 63)')
})
