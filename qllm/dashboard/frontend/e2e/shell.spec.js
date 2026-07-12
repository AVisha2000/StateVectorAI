import { test, expect } from '@playwright/test'
import { mockApi } from './fixtures.js'

test.beforeEach(async ({ page }) => { await mockApi(page) })

test('sidebar shows all three nav groups and every surface link', async ({ page }) => {
  await page.goto('/')
  const sidebar = page.locator('.sidebar')
  for (const name of ['Overview', 'Discover', 'Library', 'Atlas', 'Designer', 'Bench', 'Runs', 'Studies', 'Verdicts', 'Datasets', 'Queue & Backends']) {
    await expect(sidebar.getByRole('link', { name })).toBeVisible()
  }
})

test('theme toggle flips and persists to localStorage', async ({ page }) => {
  await page.goto('/')
  const toggle = page.getByRole('button', { name: /Switch to (dark|light) theme/ })
  const before = await page.evaluate(() => document.documentElement.dataset.theme)
  await toggle.click()
  const after = await page.evaluate(() => document.documentElement.dataset.theme)
  expect(after).not.toBe(before)
  expect(await page.evaluate(() => localStorage.getItem('qllm-theme'))).toBe(after)
})

test('legacy routes redirect to their new surfaces', async ({ page }) => {
  const redirects = [['/launch', '/bench'], ['/jobs', '/runs'], ['/results', '/verdicts'], ['/models', '/designer'], ['/explore', '/atlas'], ['/gpu', '/system']]
  for (const [from, to] of redirects) {
    await page.goto(from)
    await expect(page).toHaveURL(new RegExp(`${to}$`))
  }
})

test('unknown route shows Not Found without breaking the shell', async ({ page }) => {
  await page.goto('/nope-nope')
  await expect(page.getByRole('link', { name: 'Atlas' })).toBeVisible()
  await expect(page.getByText(/not found/i)).toBeVisible()
})

test('Overview: tiles reflect job counts and the running table lists live runs', async ({ page }) => {
  await page.goto('/')
  const running = page.locator('.tile', { hasText: 'Running' })
  await expect(running.locator('.v')).toHaveText('1')
  await expect(page.locator('.tile', { hasText: 'Failed' }).locator('.v')).toHaveText('1')
  await expect(page.getByText('#7 qrnn-s42')).toBeVisible()
})

test('Overview: Latest-verdicts strip lists recent snapshots with claim + replication', async ({ page }) => {
  await page.goto('/')
  const card = page.locator('.card', { hasText: 'Latest verdicts' })
  await expect(card).toBeVisible()
  await expect(card.getByText('empirical')).toBeVisible()
  await expect(card.getByText('multi_seed_single_instance')).toBeVisible()
  await expect(card.getByRole('link', { name: /Open/ }).first()).toBeVisible()
  // integrity framing: positive claims are candidates, not established
  await expect(card.getByText(/candidates.*not established/i)).toBeVisible()
})

test('Overview: Latest-verdicts degrades when the store is unreachable', async ({ page }) => {
  await mockApi(page, { '/verdicts': null })
  await page.goto('/')
  await expect(page.getByText(/verdict store isn.t reachable/i)).toBeVisible()
})

test('System: /status five fields render and quantum backends are listed', async ({ page }) => {
  await page.goto('/system')
  await expect(page.getByText('CPU · active')).toBeVisible()
  await expect(page.locator('.kpi', { hasText: 'Runs recorded' }).locator('.v')).toHaveText('312')
  await expect(page.locator('.kpi', { hasText: 'Running' }).locator('.v')).toHaveText('1')
  for (const b of ['pennylane', 'tensorcircuit', 'tensorcircuit_mps']) {
    await expect(page.getByText(b, { exact: true })).toBeVisible()
  }
})

test('Datasets: table renders the registered datasets', async ({ page }) => {
  await page.goto('/datasets')
  await expect(page.getByText('monitored_ising')).toBeVisible()
  await expect(page.getByText('contextual')).toBeVisible()
})
