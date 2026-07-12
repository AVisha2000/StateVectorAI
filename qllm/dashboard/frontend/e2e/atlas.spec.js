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

test('Atlas: ?node= deep-link selects a cell from the URL', async ({ page }) => {
  await page.goto('/atlas?node=c_variational_swaps')
  await expect(page.locator('.atlas-side').getByText('Variational embedding / attention / FFN / block swaps')).toBeVisible()
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
  await expect(page.getByText('Variational embedding / attention / FFN / block swaps')).toBeVisible()
  await page.locator('.atlas-group-toggle').first().click()
  await expect(page.getByText('Variational embedding / attention / FFN / block swaps')).toHaveCount(0)
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
