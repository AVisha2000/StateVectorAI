import { defineConfig, devices } from '@playwright/test'

// E2E harness for the dashboard frontend. Tests drive the real production build
// in headless Chromium and stub the backend with page.route('**/api/**', …), so
// they run with no FastAPI/GPU present and stay deterministic. See e2e/fixtures.js.
export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  expect: {
    timeout: 7_000,
    // Visual-regression tolerance. Baselines are platform-specific (Playwright
    // stores one per OS, e.g. -linux.png / -win32.png). Generate/refresh with
    // `npm run test:e2e:visual:update`; CI (Linux) needs its own -linux baselines.
    toHaveScreenshot: { maxDiffPixelRatio: 0.02, animations: 'disabled', scale: 'css', caret: 'hide' },
  },
  fullyParallel: true,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:4174',
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    // Build then serve the production bundle (what ships), reusing a running one.
    command: 'npm run build && npm run preview:e2e',
    url: 'http://localhost:4174',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})
