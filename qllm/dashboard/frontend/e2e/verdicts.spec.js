import { test, expect } from '@playwright/test'
import { mockApi, JOBS } from './fixtures.js'

test.beforeEach(async ({ page }) => { await mockApi(page) })

test('Verdicts store: distinct claim/replication columns; null outcome first-class', async ({ page }) => {
  await page.goto('/verdicts')
  await expect(page.getByRole('columnheader', { name: 'Claim level' })).toBeVisible()
  await expect(page.getByRole('columnheader', { name: 'Replication' })).toBeVisible()
  await expect(page.getByRole('columnheader', { name: 'Assessment' })).toBeVisible()
  await expect(page.getByText('empirical')).toBeVisible()
  await expect(page.getByText('multi_seed_single_instance')).toBeVisible()
  await expect(page.getByText('none')).toBeVisible() // the refuted/null snapshot is shown
})

test('Verdicts falls back to derived verdicts when the store is absent', async ({ page }) => {
  await mockApi(page, { '/verdicts': null }) // store 404s → derive from jobs with a comparison
  await page.goto('/verdicts')
  await expect(page.getByText(/persistent verdict store.*isn.t reachable|derived on the fly/i)).toBeVisible()
  await expect(page.getByText('#7 qrnn-s42')).toBeVisible()
})

test('Verdict detail (snapshot): scorecard has no composite score; promotion human-gated', async ({ page }) => {
  await page.goto('/verdicts')
  await page.getByRole('link', { name: /Open/ }).first().click()
  await expect(page).toHaveURL(/\/verdicts\/101$/)
  await expect(page.getByRole('heading', { name: /qrnn-vs-gru|Verdict/ })).toBeVisible()
  await expect(page.getByText(/no total|per dimension/i)).toBeVisible()
  await expect(page.getByText(/no composite advantage score/i)).toBeVisible()
  // canonical claim + replication both present as distinct chips/rows
  await expect(page.getByText(/replication:/i)).toBeVisible()
  await expect(page.getByRole('button', { name: /Promote to claim ladder/i })).toBeDisabled()
})

test('Verdict detail (comparison fallback): a run without a snapshot renders its pair', async ({ page }) => {
  // /verdicts/7 has no snapshot → falls back to /jobs/7/comparison
  await page.goto('/verdicts/7')
  await expect(page.getByText(/Advantage scorecard|Candidate vs its matched control|Perplexity/i).first()).toBeVisible()
  await expect(page.getByText(/simulator/i).first()).toBeVisible() // wall-time labeled simulator cost
})

test('Scaling: KPIs, charts, and the barren-plateau fit light up from diagnostics', async ({ page }) => {
  await page.goto('/runs/scaling/scale-grp')
  await expect(page.getByRole('heading', { name: 'Scaling', exact: true })).toBeVisible()
  await expect(page.locator('.kpi', { hasText: 'Progress' }).locator('.v')).toHaveText('3/3')
  await expect(page.locator('.chart-wrap svg').first()).toBeVisible()
  await expect(page.getByText(/simulator cost/i).first()).toBeVisible()
  // scaling_fit dimension (from /jobs/7/diagnostics) renders the measured fit
  await expect(page.getByText('Barren-plateau scaling fit')).toBeVisible()
  await expect(page.getByText('Variance decay / qubit')).toBeVisible()
  await expect(page.getByText('detected')).toBeVisible()
})
