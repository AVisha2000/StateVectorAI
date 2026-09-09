import { test, expect } from '@playwright/test'

import { mockApi } from './fixtures.js'

const playground = process.env.VITE_PLAYGROUND_MODE === '1'

test.use({ baseURL: process.env.PLAYWRIGHT_TEST_BASE_URL || 'http://localhost:4174' })

test('decision confirmation names whether guidance was persisted', async ({ page }) => {
  await mockApi(page)
  await page.goto('/portal/decisions/decision-controls')
  await page.getByRole('radio', { name: 'Linear readout', exact: true }).check()
  await page.getByLabel('Reasoning').fill('Keep the review bounded to the matched linear control.')
  await page.getByRole('button', { name: 'Record guidance' }).click()

  const status = page.getByRole('status')
  await expect(status).toContainText(playground
    ? 'Your guidance was simulated in this playground and was not persisted.'
    : 'Your guidance was recorded for this coordination decision.')
  await expect(page.getByText(playground ? 'Simulated answer:' : 'Recorded answer:', { exact: false })).toBeVisible()
  await expect(page.getByText(playground ? 'Recorded answer:' : 'Simulated answer:', { exact: false })).toHaveCount(0)
})
