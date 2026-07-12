# Dashboard frontend — E2E tests (Playwright)

Drives the **production build** in headless Chromium and stubs the backend via
`page.route('**/api/**', …)` (see `fixtures.js`), so the suite runs with **no
FastAPI/GPU** and stays deterministic.

## Run

```bash
npm run test:e2e            # functional specs (34) — this is what CI runs
npm run test:e2e:visual     # visual-regression snapshots (compare)
npm run test:e2e:visual:update   # (re)generate visual baselines for THIS OS
```

Functional specs are grouped by surface (`shell`, `runs`, `verdicts`, `bench`,
`atlas`, `designer`, `research`) and assert rendering, key flows, graceful
loading/error/empty states, and the research-integrity invariants (no composite
advantage score, claim-level vs replication kept distinct, nulls first-class,
human-gated promotion/GPU, diagnostics labeled as diagnostics, simulator-cost
labeling).

## Visual regression & platform baselines

Screenshots are **OS-specific** (font rendering differs), so Playwright stores
one baseline per platform — e.g. `overview-dark-chromium-linux.png` vs
`…-win32.png`. Visual specs are tagged `@visual` and are **excluded from
`test:e2e`** (and therefore from CI), because CI runs on Linux and needs its own
`-linux.png` baselines.

**To enable the visual check in CI (Linux) once:** on a Linux machine (or the
official `mcr.microsoft.com/playwright` image), run
`npm ci && npm run test:e2e:visual:update`, commit the generated `*-linux.png`
files, then add a `visual` job that runs `npm run test:e2e:visual`. Tolerance is
`maxDiffPixelRatio: 0.02` with animations disabled (see `playwright.config.js`).

The committed `*-win32.png` baselines make `npm run test:e2e:visual` pass on
Windows today.
