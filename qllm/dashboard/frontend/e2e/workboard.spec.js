import { test, expect } from '@playwright/test'
import { DECISIONS, mockApi } from './fixtures.js'

test('Workboard shows dense counts, shared work, and human decision framing', async ({ page }) => {
  await mockApi(page)
  await page.goto('/portal')
  await expect(page.getByRole('heading', { name: 'Workboard' })).toBeVisible()
  await expect(page.getByLabel('Workboard counts')).toContainText('Available')
  await expect(page.getByLabel('Workboard counts')).toContainText('Running')
  await expect(page.getByText('Map matched controls')).toBeVisible()
  await expect(page.getByText('Agent recommendation: Linear readout.')).toBeVisible()
  await expect(page.getByText('decision-controls', { exact: true }).first()).toBeVisible()
  await expect(page.getByRole('link', { name: 'decision-controls' })).toHaveAttribute('href', '/portal/decisions/decision-controls')
  await expect(page.getByText('Blocked: human decision due')).toBeVisible()
  await expect(page.getByText('actor.agent.gamma')).toBeVisible()
  await expect(page.getByText('prior-noise', { exact: true })).toBeVisible()
  await expect(page.getByText('Revision requested')).toBeVisible()
  await expect(page.getByText('Evidence reviewed for human handoff').first()).toBeVisible()
  await expect(page.getByText(/This page never submits a claim, decision, or experiment/i)).toBeVisible()
})

test('Workboard names an empty queue', async ({ page }) => {
  await mockApi(page, { '/work': [], '/decisions': [] })
  await page.goto('/portal')
  await expect(page.getByText('No agent work or human decisions are open yet.')).toBeVisible()
})

test('Workboard excludes expired decisions from human review', async ({ page }) => {
  const expired = { ...DECISIONS[0], status: 'expired', question: 'Expired control-family choice' }
  await mockApi(page, { '/decisions': [expired] })
  await page.goto('/portal')
  const openDecisionCount = page.locator('.workboard-count').filter({ hasText: 'Open decisions' })
  await expect(openDecisionCount).toContainText('0')
  await expect(page.getByText('No open human decisions.')).toBeVisible()
  await expect(page.getByText('Expired control-family choice')).not.toBeVisible()
})

test('Workboard identifies disabled coordination and leaves the lab reachable', async ({ page }) => {
  await mockApi(page)
  await page.route('**/api/work', (route) => route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'coordination disabled' }) }))
  await page.route('**/api/decisions', (route) => route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'coordination disabled' }) }))
  await page.goto('/portal')
  await expect(page.getByText(/Agent coordination is disabled/i)).toBeVisible()
  await expect(page.locator('.state').getByRole('link', { name: 'Lab overview' })).toBeVisible()
})

test('Workboard names an unexpected API failure', async ({ page }) => {
  await mockApi(page, { '/work': { detail: 'temporary backend failure' } })
  await page.route('**/api/work', (route) => route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'temporary backend failure' }) }))
  await page.goto('/portal')
  await expect(page.getByText('Could not load the Workboard.')).toBeVisible()
})

test('narrow viewport exposes a keyboard-operable menu', async ({ page }) => {
  await mockApi(page)
  await page.setViewportSize({ width: 390, height: 700 })
  await page.goto('/portal')
  await expect.poll(() => page.evaluate(() => document.body.scrollWidth <= window.innerWidth)).toBe(true)
  const menu = page.getByRole('button', { name: 'Menu' })
  await expect(menu).toBeVisible()
  const menuBox = await menu.boundingBox()
  expect(menuBox?.height).toBeGreaterThanOrEqual(44)
  await menu.press('Enter')
  await expect(menu).toHaveAttribute('aria-expanded', 'true')
  await expect(page.locator('#mobile-navigation').getByRole('link', { name: 'Lab overview' })).toBeVisible()
  await page.locator('#mobile-navigation').getByRole('link', { name: 'Lab overview' }).press('Enter')
  await expect(page).toHaveURL(/\/lab$/)
  await expect(page.getByRole('heading', { name: 'Overview' })).toBeVisible()
})
