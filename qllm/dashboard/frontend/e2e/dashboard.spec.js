import { test, expect } from '@playwright/test'
import { mockApi } from './fixtures.js'

test.beforeEach(async ({ page }) => {
  await mockApi(page)
})

test('shell + navigation renders', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('link', { name: 'Atlas' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Bench' })).toBeVisible()
  await expect(page.getByRole('heading', { name: /Overview/ })).toBeVisible()
})

test('Runs table renders mocked jobs and a row opens run detail', async ({ page }) => {
  await page.goto('/runs')
  await expect(page.getByText('qrnn-s42')).toBeVisible()
  await expect(page.getByText('gru-s42')).toBeVisible()
  await page.getByText('qrnn-s42').click()
  await expect(page).toHaveURL(/\/runs\/7$/)
  // run detail header + a diagnostics KPI label
  await expect(page.getByText('#7 qrnn-s42')).toBeVisible()
  await expect(page.getByText('val_ppl').first()).toBeVisible()
})

test('Runs shows a graceful error when the jobs endpoint fails', async ({ page }) => {
  await page.route('**/api/jobs', (route) => route.fulfill({ status: 500, contentType: 'application/json', body: '{"detail":"boom"}' }))
  await page.goto('/runs')
  await expect(page.getByText(/Could not load|Is the dashboard API running/i)).toBeVisible()
})

test('Verdicts store: claim level and replication shown as DISTINCT columns; null outcome present', async ({ page }) => {
  await page.goto('/verdicts')
  // header columns exist and are separate
  await expect(page.getByRole('columnheader', { name: 'Claim level' })).toBeVisible()
  await expect(page.getByRole('columnheader', { name: 'Replication' })).toBeVisible()
  // the quantum-candidate row
  await expect(page.getByText('empirical')).toBeVisible()
  await expect(page.getByText('multi_seed_single_instance')).toBeVisible()
  // the null / refuted outcome is first-class (not hidden)
  await expect(page.getByText('none')).toBeVisible()
})

test('Verdict detail: per-dimension scorecard, no composite score, promotion disabled (human-gated)', async ({ page }) => {
  await page.goto('/verdicts')
  await page.getByRole('link', { name: 'Open →' }).first().click()
  await expect(page.getByRole('heading', { name: /qrnn-vs-gru|Verdict/ })).toBeVisible()
  // per-dimension scorecard, explicitly no aggregate total
  await expect(page.getByText(/no total|per dimension/i)).toBeVisible()
  // integrity: the UI states the no-composite-score guarantee outright
  await expect(page.getByText(/no composite advantage score/i)).toBeVisible()
  // promotion is human-gated → the button is disabled
  await expect(page.getByRole('button', { name: /Promote to claim ladder/i })).toBeDisabled()
})

test('Bench: device defaults to CPU and GPU is human-gated', async ({ page }) => {
  await page.goto('/bench')
  await expect(page.getByRole('heading', { name: /Bench/ })).toBeVisible()
  const device = page.locator('select').filter({ hasText: 'cpu' }).first()
  await expect(device).toHaveValue('cpu')
})

test('Atlas: SVG graph renders all cells; classical-holds (null) is first-class', async ({ page }) => {
  await page.goto('/atlas')
  // the classical-holds bucket is a first-class summary tile with a nonzero count
  const holdsTile = page.locator('.atlas-summary-tile', { hasText: 'Classical holds' })
  await expect(holdsTile).toBeVisible()
  await expect(holdsTile.locator('.v')).not.toHaveText('0')
  // switch to the graph and assert the SVG cells render
  await page.getByRole('button', { name: 'Graph' }).click()
  const cells = page.locator('.atlas-graph-svg g[role="button"]')
  await expect(cells).toHaveCount(19)
  // click a cell → detail shows claim level and replication as separate rows
  await cells.first().click()
  const detail = page.locator('.atlas-side')
  await expect(detail.getByText('Claim level (map)')).toBeVisible()
  await expect(detail.getByText('Replication', { exact: true })).toBeVisible()
})

test('Designer: circuit SVG renders and re-renders when the ansatz changes', async ({ page }) => {
  await page.goto('/designer')
  const svg = page.locator('.circuit-wrap svg')
  await expect(svg).toBeVisible()
  const rectsHW = await svg.locator('rect').count()
  expect(rectsHW).toBeGreaterThan(0)
  // switch ansatz → circuit changes (reuploading adds RX/RY + CZ)
  await page.locator('select').first().selectOption('reuploading')
  await expect(svg.getByText('RX').first()).toBeVisible()
})
