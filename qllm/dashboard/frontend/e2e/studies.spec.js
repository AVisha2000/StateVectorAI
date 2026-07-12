import { test, expect } from '@playwright/test'
import { mockApi } from './fixtures.js'

test.beforeEach(async ({ page }) => { await mockApi(page) })

test('Studies list renders and links to a study', async ({ page }) => {
  await page.goto('/studies')
  await expect(page.getByRole('heading', { name: /Studies — multi-seed rigor/ })).toBeVisible()
  await expect(page.getByText('qffn-multiseed')).toBeVisible()
  await page.getByRole('link', { name: /Open/ }).first().click()
  await expect(page).toHaveURL(/\/studies\/1$/)
})

test('Studies is reachable from the sidebar', async ({ page }) => {
  await page.goto('/')
  await page.locator('.sidebar').getByRole('link', { name: 'Studies' }).click()
  await expect(page).toHaveURL(/\/studies$/)
})

test('Study detail: multi-seed KPIs, delta strip, ladder, and integrity framing', async ({ page }) => {
  await page.goto('/studies/1')
  await expect(page.getByRole('heading', { name: 'qffn-multiseed' })).toBeVisible()
  // aggregate KPIs — replication (fair pairs) distinct from the claim label
  await expect(page.locator('.kpi', { hasText: 'Replication' }).locator('.v')).toHaveText('4')
  await expect(page.locator('.kpi', { hasText: 'Consistency' })).toBeVisible()
  await expect(page.locator('.kpi', { hasText: 'Mean Δ val_ppl' })).toBeVisible()
  // the multi-seed per-pair strip renders (recharts svg)
  await expect(page.locator('.chart-wrap svg').first()).toBeVisible()
  // integrity: no composite score; explicitly multi-seed spread
  await expect(page.getByText(/no composite advantage score/i)).toBeVisible()
  await expect(page.getByText(/multi-seed spread, not a single verdict/i)).toBeVisible()
  // evidence ladder + study runs
  await expect(page.getByText('Multiple seeds')).toBeVisible()
  await expect(page.getByText('q4/d2').first()).toBeVisible()
})

test('Study detail: seed-band aggregates per-seed val_ppl trajectories', async ({ page }) => {
  await page.goto('/studies/1')
  const band = page.locator('.card', { hasText: 'Seed-band val_ppl over steps' })
  await expect(band).toBeVisible()
  // three per-seed workspaces → a real band renders (composed area+line svg)
  await expect(band.locator('.chart-wrap svg')).toBeVisible()
  // integrity: the spread is variance, not a claim
  await expect(band.getByText(/seed-to-seed variance of the trajectory itself, not a claim/i)).toBeVisible()
})

test('Study detail: seed-band degrades when per-seed curves are absent', async ({ page }) => {
  // strip the per-seed workspaces → 404 → graceful note, no crash
  await mockApi(page, { '/jobs/201/workspace': null, '/jobs/202/workspace': null, '/jobs/203/workspace': null })
  await page.goto('/studies/1')
  const band = page.locator('.card', { hasText: 'Seed-band val_ppl over steps' })
  await expect(band).toBeVisible()
  await expect(band.getByText(/No per-seed training curves are available/i)).toBeVisible()
})

test('Studies empty state points to the Bench', async ({ page }) => {
  await mockApi(page, { '/studies': [] })
  await page.goto('/studies')
  await expect(page.getByText(/No studies yet/i)).toBeVisible()
})
