import { test, expect } from '@playwright/test'
import { mockApi } from './fixtures.js'

test.beforeEach(async ({ page }) => { await mockApi(page) })

const deviceSelect = (page) => page.locator('label').filter({ hasText: 'Device' }).locator('select')

test('Bench: candidate + matched-control cards and CPU-default protocol', async ({ page }) => {
  await page.goto('/portal/bench')
  await expect(page.getByRole('heading', { name: /Bench/ })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Candidate' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Matched control' })).toBeVisible()
  await expect(deviceSelect(page)).toHaveValue('cpu')
  // matched-control card shows the curated twin, framed as a proposal not a pass
  await expect(page.getByText(/proposal — fairness is gated after runs/i)).toBeVisible()
})

test('Bench: an Atlas proposal requires an explicit preset rather than inheriting a default', async ({ page }) => {
  await page.goto('/portal/bench?source=atlas&area_id=qrnn_representability_and_training&hypothesis=Evaluate%20QRNN%20training.')
  const preset = page.locator('.card').filter({ has: page.getByRole('heading', { name: 'Candidate' }) }).locator('select')
  const queue = page.getByRole('button', { name: /Queue \d+ run/i })
  await expect(preset).toHaveValue('')
  await expect(queue).toBeDisabled()
  await expect(page.getByText(/Atlas proposal — not evidence/i)).toBeVisible()
  await preset.selectOption('quantum-ffn-4q')
  await expect(queue).toBeEnabled()
})

test('Bench: an incomplete Atlas handoff stays fail-closed after preset selection', async ({ page }) => {
  await page.goto('/portal/bench?source=atlas&area_id=qrnn_representability_and_training')
  const preset = page.locator('.card').filter({ has: page.getByRole('heading', { name: 'Candidate' }) }).locator('select')
  const queue = page.getByRole('button', { name: /Queue \d+ run/i })
  await expect(preset).toHaveValue('')
  await expect(page.getByText(/Incomplete Atlas handoff/i)).toBeVisible()
  await preset.selectOption('quantum-ffn-4q')
  await expect(queue).toBeDisabled()
})

test('Bench: switching device to GPU blocks queueing (human gate)', async ({ page }) => {
  await page.goto('/portal/bench')
  const queue = page.getByRole('button', { name: /Queue \d+ run/i })
  await expect(queue).toBeEnabled()
  await deviceSelect(page).selectOption('gpu')
  await expect(queue).toBeDisabled()
  await expect(page.getByText(/GPU.*human gate|GPU\/auto targets are a human gate/i)).toBeVisible()
})

test('Bench: rigor selector changes the run estimate', async ({ page }) => {
  await page.goto('/portal/bench')
  await page.locator('.rig', { hasText: 'Quick probe' }).click()
  await expect(page.getByRole('button', { name: /Queue 1 run/i })).toBeVisible()
  await page.locator('.rig', { hasText: 'Standard pair' }).click()
  await expect(page.getByRole('button', { name: /Queue 10 runs/i })).toBeVisible()
})

test('Bench: quantum controls render and ride along as quantum_overrides', async ({ page }) => {
  let body = null
  await page.route('**/api/jobs', (route) => {
    if (route.request().method() === 'POST') { body = route.request().postDataJSON(); return route.fulfill({ status: 200, contentType: 'application/json', body: '{"id":99,"status":"queued"}' }) }
    return route.fallback()
  })
  await page.goto('/portal/bench')
  await expect(page.getByRole('heading', { name: 'Quantum controls' })).toBeVisible()
  await page.locator('label').filter({ hasText: 'Qubits' }).locator('input').fill('6')
  await page.getByRole('button', { name: /Queue \d+ run/i }).click()
  await expect(page).toHaveURL(/\/runs$/)
  expect(body?.quantum_overrides?.n_qubits).toBe(6)
})

test('Bench: queueing a CPU standard pair posts jobs and navigates to Runs', async ({ page }) => {
  let posts = 0
  await page.route('**/api/jobs', (route) => {
    if (route.request().method() === 'POST') { posts += 1; return route.fulfill({ status: 200, contentType: 'application/json', body: '{"id":99,"status":"queued"}' }) }
    return route.fallback()
  })
  await page.goto('/portal/bench')
  await page.getByRole('button', { name: /Queue 10 runs/i }).click()
  await expect(page).toHaveURL(/\/runs$/)
  expect(posts).toBeGreaterThan(0)
})
