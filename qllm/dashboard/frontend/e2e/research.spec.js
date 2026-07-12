import { test, expect } from '@playwright/test'
import { mockApi, CAPABILITIES } from './fixtures.js'

test.beforeEach(async ({ page }) => { await mockApi(page) })

test('Library: D4 capability gate reads the shipped capabilities honestly', async ({ page }) => {
  await page.goto('/library')
  await expect(page.getByRole('heading', { name: /Library/ })).toBeVisible()
  await expect(page.getByText('D4 boundary')).toBeVisible()
  // the paid pieces are shown gated, not enabled
  await expect(page.getByText('none — human-gated').first()).toBeVisible()
  await expect(page.getByText('unset — human-gated')).toBeVisible()
})

test('Library: a bounded arXiv scan lists papers and shows the quota', async ({ page }) => {
  await page.goto('/library')
  await page.getByRole('button', { name: /Scan/ }).click()
  await expect(page.getByText('Reuploading circuits resist barren plateaus')).toBeVisible()
  await expect(page.getByText('Quantum models as random features')).toBeVisible()
  await expect(page.getByText(/48 remaining/)).toBeVisible()
})

test('Library: degrades to the gate explainer when the service is unreachable', async ({ page }) => {
  await mockApi(page, { '/research/capabilities': null })
  await page.goto('/library')
  await expect(page.getByText(/research service isn.t reachable/i)).toBeVisible()
  await expect(page.locator('body')).not.toContainText('undefined')
})

test('Discover: copilot + idea queue are human-gated while paid services are off', async ({ page }) => {
  await page.goto('/discover')
  await expect(page.getByRole('heading', { name: /Discover/ })).toBeVisible()
  await expect(page.getByText(/copilot.*human-gated|human-gated/i).first()).toBeVisible()
  await expect(page.getByPlaceholder(/Ask about papers/i)).toBeDisabled()
  await expect(page.getByRole('button', { name: 'Send' })).toBeDisabled()
})

test('Discover: copilot lights up when a provider is enabled', async ({ page }) => {
  await mockApi(page, { '/research/capabilities': { ...CAPABILITIES, paid_services_enabled: true, llm_provider: 'anthropic' } })
  await page.goto('/discover')
  await expect(page.getByPlaceholder(/Ask about papers/i)).toBeEnabled()
  await expect(page.getByRole('button', { name: 'Send' })).toBeEnabled()
})
