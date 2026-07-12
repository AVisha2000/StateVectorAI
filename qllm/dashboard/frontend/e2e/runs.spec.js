import { test, expect } from '@playwright/test'
import { mockApi } from './fixtures.js'

test.beforeEach(async ({ page }) => { await mockApi(page) })

test('Runs table lists all jobs and filters by status', async ({ page }) => {
  await page.goto('/runs')
  await expect(page.getByText('qrnn-s42')).toBeVisible()
  await expect(page.getByText('gru-s42')).toBeVisible()
  await expect(page.getByText('qattn-s77')).toBeVisible()
  // filter to Failed → only the error job remains
  await page.getByRole('button', { name: 'Failed' }).click()
  await expect(page.getByText('qattn-s77')).toBeVisible()
  await expect(page.getByText('qrnn-s42')).toHaveCount(0)
  // filter to Running
  await page.getByRole('button', { name: 'Running' }).click()
  await expect(page.getByText('qrnn-s42')).toBeVisible()
  await expect(page.getByText('gru-s42')).toHaveCount(0)
})

test('Runs shows a graceful error when /jobs fails', async ({ page }) => {
  await page.route('**/api/jobs', (r) => r.fulfill({ status: 500, contentType: 'application/json', body: '{"detail":"boom"}' }))
  await page.goto('/runs')
  await expect(page.getByText(/Could not load|Is the dashboard API running/i)).toBeVisible()
})

test('Runs empty state when there are no jobs', async ({ page }) => {
  await mockApi(page, { '/jobs': [] })
  await page.goto('/runs')
  await expect(page.getByText(/No runs match this filter/i)).toBeVisible()
})

test('Run detail: header, diagnostics KPIs, charts, and warnings', async ({ page }) => {
  await page.goto('/runs')
  await page.getByText('qrnn-s42').click()
  await expect(page).toHaveURL(/\/runs\/7$/)
  await expect(page.getByText('#7 qrnn-s42')).toBeVisible()
  // status + type tags
  await expect(page.getByText('QUANTUM', { exact: true })).toBeVisible()
  // diagnostics KPIs populate from /jobs/7/diagnostics (labeled as diagnostics)
  await expect(page.locator('.kpi', { hasText: 'Grad variance' })).toBeVisible()
  await expect(page.locator('.kpi', { hasText: 'Grad SNR' })).toBeVisible()
  await expect(page.locator('.kpi', { hasText: 'Entanglement' })).toBeVisible()
  await expect(page.getByText(/diagnostics \/ mechanism candidates|mechanism candidates/i)).toBeVisible()
  // two chart panels render an SVG (recharts)
  await expect(page.locator('.chart-wrap svg').first()).toBeVisible()
  // backend interpretation warning surfaced
  await expect(page.getByText('Interpretation warnings')).toBeVisible()
  // compare-with-twin deep link
  await expect(page.getByRole('link', { name: /Compare with twin/i })).toBeVisible()
})

test('Run detail: unknown id degrades gracefully', async ({ page }) => {
  await page.goto('/runs/424242')
  await expect(page.getByText(/Could not load this run|not found/i)).toBeVisible()
})
