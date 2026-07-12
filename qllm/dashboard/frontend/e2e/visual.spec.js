import { test, expect } from '@playwright/test'
import { mockApi } from './fixtures.js'

// Visual-regression snapshots for the key surfaces in both themes. Tagged
// @visual so they are EXCLUDED from the default `npm run test:e2e` (and CI) —
// baselines are platform-specific, so run `npm run test:e2e:visual:update` on the
// target OS (Linux for CI) to (re)generate them. See the frontend README/e2e.
const THEMES = ['light', 'dark']

async function setup(page, theme) {
  await page.addInitScript((t) => localStorage.setItem('qllm-theme', t), theme)
  await mockApi(page)
}

// Mask the live/sync status chip (its label depends on fetch/stream timing).
const mask = (page) => [page.locator('.chip')]

test.describe('visual', { tag: '@visual' }, () => {
  for (const theme of THEMES) {
    test(`Overview ${theme}`, async ({ page }) => {
      await setup(page, theme)
      await page.goto('/')
      await expect(page.getByRole('heading', { name: /Overview/ })).toBeVisible()
      await expect(page).toHaveScreenshot(`overview-${theme}.png`, { fullPage: true, mask: mask(page) })
    })

    test(`Runs ${theme}`, async ({ page }) => {
      await setup(page, theme)
      await page.goto('/runs')
      await expect(page.getByText('qrnn-s42')).toBeVisible()
      await expect(page).toHaveScreenshot(`runs-${theme}.png`, { fullPage: true, mask: mask(page) })
    })

    test(`Verdict detail ${theme}`, async ({ page }) => {
      await setup(page, theme)
      await page.goto('/verdicts/101')
      await expect(page.getByRole('heading', { name: /qrnn-vs-gru|Verdict/ })).toBeVisible()
      await expect(page).toHaveScreenshot(`verdict-detail-${theme}.png`, { fullPage: true, mask: mask(page) })
    })

    test(`Bench ${theme}`, async ({ page }) => {
      await setup(page, theme)
      await page.goto('/bench')
      await expect(page.getByRole('heading', { name: 'Candidate' })).toBeVisible()
      await expect(page).toHaveScreenshot(`bench-${theme}.png`, { fullPage: true, mask: mask(page) })
    })

    test(`Designer ${theme}`, async ({ page }) => {
      await setup(page, theme)
      await page.goto('/designer')
      await expect(page.locator('.circuit-wrap svg')).toBeVisible()
      await expect(page).toHaveScreenshot(`designer-${theme}.png`, { fullPage: true, mask: mask(page) })
    })

    test(`Atlas graph ${theme}`, async ({ page }) => {
      await setup(page, theme)
      await page.goto('/atlas')
      await page.getByRole('button', { name: 'Graph' }).click()
      await expect(page.locator('.atlas-graph-svg g[role="button"]')).toHaveCount(19)
      await expect(page).toHaveScreenshot(`atlas-graph-${theme}.png`, { fullPage: true, mask: mask(page) })
    })
  }
})
