import { test, expect } from '@playwright/test'

import { DECISIONS, LINEAGE_DECISION, WORK, WORK_DECISION_DETAIL, mockApi } from './fixtures.js'

test('decision inbox deep-links to one bounded human question', async ({ page }) => {
  await mockApi(page)
  await page.goto('/portal/decisions')
  await expect(page.getByRole('heading', { name: 'Decision inbox' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Open first' })).toBeVisible()
  await expect(page.getByText('Which control family should be prioritised?')).toBeVisible()
  await page.getByRole('link', { name: 'Review question' }).press('Enter')
  await expect(page).toHaveURL(/\/decisions\/decision-controls$/)
  await expect(page.getByRole('heading', { name: 'Which control family should be prioritised?' })).toBeVisible()
  await expect(page.getByText(/does not make or promote a scientific claim/i)).toBeVisible()
  await page.getByRole('link', { name: 'View research record' }).click()
  await expect(page).toHaveURL(/\/research\/work-decision$/)
  await expect(page.getByRole('heading', { name: 'Choose the next control family' })).toBeVisible()
  await expect(page.getByRole('alert')).not.toBeVisible()
})

test('authorized synthetic human can resolve a decision from the keyboard', async ({ page }) => {
  let resolutionRequest
  page.on('request', (request) => {
    if (request.url().endsWith('/api/decisions/decision-controls/resolve')) resolutionRequest = request
  })
  await mockApi(page)
  await page.goto('/portal/decisions/decision-controls')
  await page.getByRole('radio', { name: 'Linear readout', exact: true }).check()
  await page.getByLabel('Reasoning').fill('Keep the next correction bounded to the matched linear control.')
  await page.getByRole('button', { name: 'Record guidance' }).press('Enter')
  await expect(page.getByText(/guidance was recorded/i)).toBeVisible()
  await expect(page.getByText(/Recorded answer:/)).toContainText('Linear readout')
  expect(resolutionRequest?.headers()['x-statevector-test-actor']).toBe('human-reviewer')
  const payload = resolutionRequest?.postDataJSON()
  expect(payload.expected_revision).toBe(1)
  expect(payload.idempotency_key).toBeTruthy()
})

test('resolving guidance refreshes the cached Workboard projection without a page reload', async ({ page }) => {
  let resolved = false
  let workGets = 0
  let decisionGets = 0
  let documentRequests = 0
  page.on('request', (request) => {
    if (request.isNavigationRequest() && request.resourceType() === 'document') documentRequests += 1
  })
  const resolvedDecision = {
    ...DECISIONS[0], revision: 2, status: 'resolved', answer: 'Linear readout',
    rationale: 'Keep the review bounded to the matched linear control.', resolver_actor_id: 'actor.human.reviewer',
    allowed_actions: [], actionable_next_state: null,
  }
  const resolvedWork = WORK.map((item) => item.id === 'work-decision'
    ? { ...item, revision: 5, blocking_decision_ids: [], actionable_next_state: null }
    : item)
  const resolvedWorkItem = resolvedWork.find((item) => item.id === 'work-decision')
  const resolvedWorkDetail = {
    ...WORK_DECISION_DETAIL,
    ...resolvedWorkItem,
    events: [...WORK_DECISION_DETAIL.events, {
      id: 25, work_id: 'work-decision', revision: 5, event_type: 'decision_resolved',
      actor_id: 'actor.human.reviewer', capability: 'decision:resolve', previous_state: 'running',
      next_state: 'running', payload: { decision_id: 'decision-controls' }, created_at: 1788344040,
    }],
    revisions: [...WORK_DECISION_DETAIL.revisions, {
      work_id: 'work-decision', revision: 5, state: 'running', payload: WORK_DECISION_DETAIL.payload,
      owner_actor_id: 'actor.agent.delta', lease_expires_at: 1788347460, created_at: 1788344040,
    }],
  }
  const resolvedLineage = {
    ...LINEAGE_DECISION,
    nodes: LINEAGE_DECISION.nodes.map((node) => node.type === 'work'
      ? { ...node, revision: 5, hash: 'resolved-work-hash' }
      : node.type === 'decision' ? { ...node, revision: 2, hash: 'resolved-decision-hash' } : node),
  }
  await mockApi(page)
  await page.route('**/api/work', (route) => {
    workGets += 1
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(resolved ? resolvedWork : WORK) })
  })
  await page.route('**/api/decisions', (route) => {
    decisionGets += 1
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(resolved ? [resolvedDecision, DECISIONS[1]] : DECISIONS) })
  })
  await page.route('**/api/decisions/decision-controls/resolve', (route) => {
    resolved = true
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(resolvedDecision) })
  })
  await page.route('**/api/work/work-decision', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(resolved ? resolvedWorkDetail : WORK_DECISION_DETAIL) }))
  await page.route('**/api/objects/work-decision/lineage', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(resolved ? resolvedLineage : LINEAGE_DECISION) }))

  await page.goto('/portal/research/work-decision')
  await expect(page.getByText('9999999999999999999999999999999999999999999999999999999999999999')).toBeVisible()
  await page.getByRole('link', { name: 'Workboard' }).first().click()
  await expect(page.getByRole('heading', { name: 'Workboard' })).toBeVisible()
  await expect(page.getByLabel('Workboard counts')).toContainText('Open decisions1')
  await expect(page.getByRole('link', { name: 'decision-controls', exact: true })).toBeVisible()
  const initialDocumentRequests = documentRequests

  await page.getByRole('link', { name: 'Which control family should be prioritised?' }).click()
  await page.getByRole('radio', { name: 'Linear readout', exact: true }).check()
  await page.getByLabel('Reasoning').fill('Keep the review bounded to the matched linear control.')
  await page.getByRole('button', { name: 'Record guidance' }).click()
  await expect(page.getByText(/guidance was recorded/i)).toBeVisible()
  await page.getByRole('link', { name: 'Workboard' }).first().click()
  await page.getByRole('link', { name: 'Choose the next control family' }).click()

  await expect(page.getByText('resolved-decision-hash')).toBeVisible()
  await expect(page.getByText('resolved-work-hash')).toBeVisible()
  await page.getByRole('link', { name: 'Workboard' }).first().click()

  await expect(page).toHaveURL(/\/portal$/)
  await expect(page.getByLabel('Workboard counts')).toContainText('Open decisions0')
  const workRow = page.locator('tr', { has: page.getByRole('link', { name: 'Choose the next control family' }) })
  await expect(workRow).toContainText('None')
  await expect(workRow.getByRole('link', { name: 'decision-controls', exact: true })).toHaveCount(0)
  await expect.poll(() => workGets).toBeGreaterThanOrEqual(2)
  await expect.poll(() => decisionGets).toBeGreaterThanOrEqual(2)
  expect(documentRequests).toBe(initialDocumentRequests)
})

test('stale decision recovery refreshes the current resolved revision and focuses the error', async ({ page }) => {
  await mockApi(page)
  let staleConflictRecorded = false
  const resolved = { ...DECISIONS[0], revision: 2, status: 'resolved', answer: 'Nonlinear readout', rationale: 'Another authorized reviewer already recorded the current guidance.', allowed_actions: [] }
  await page.route('**/api/decisions/decision-controls**', (route) => {
    if (route.request().method() === 'POST') {
      staleConflictRecorded = true
      return route.fulfill({ status: 409, contentType: 'application/json', body: JSON.stringify({ detail: 'synthetic stale decision revision' }) })
    }
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(staleConflictRecorded ? resolved : DECISIONS[0]) })
  })
  await page.goto('/portal/decisions/decision-controls')
  await page.getByRole('radio', { name: 'Linear readout', exact: true }).check()
  await page.getByLabel('Reasoning').fill('Keep the correction bounded and retry after refreshing the decision.')
  await page.getByRole('button', { name: 'Record guidance' }).press('Enter')
  const alert = page.getByRole('alert')
  await expect(alert).toContainText('synthetic stale decision revision')
  await expect(alert).toBeFocused()
  await expect(page.getByRole('button', { name: 'Refresh decision' })).toBeVisible()
  await page.getByRole('button', { name: 'Refresh decision' }).click()
  await expect(page.getByText(/Recorded answer:/)).toContainText('Nonlinear readout')
  await expect(page.getByText('Another authorized reviewer already recorded the current guidance.')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Record guidance' })).not.toBeVisible()
})

test('a transient resolution failure retries with the same idempotency key', async ({ page }) => {
  const keys = []
  let attempts = 0
  await mockApi(page)
  await page.route('**/api/decisions/decision-controls/resolve', (route) => {
    keys.push(route.request().postDataJSON().idempotency_key)
    attempts += 1
    if (attempts === 1) return route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'synthetic transient failure' }) })
    const payload = route.request().postDataJSON()
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ ...DECISIONS[0], revision: 2, status: 'resolved', answer: payload.answer, rationale: payload.rationale, allowed_actions: [] }) })
  })
  await page.goto('/portal/decisions/decision-controls')
  await page.getByRole('radio', { name: 'Linear readout', exact: true }).check()
  await page.getByLabel('Reasoning').fill('Retry the same bounded guidance after a transient error.')
  await page.getByRole('button', { name: 'Record guidance' }).click()
  await expect(page.getByRole('alert')).toBeFocused()
  await page.getByRole('button', { name: 'Record guidance' }).click()
  await expect(page.getByText(/guidance was recorded/i)).toBeVisible()
  expect(keys).toHaveLength(2)
  expect(keys[0]).toBeTruthy()
  expect(keys[1]).toBe(keys[0])
})

test('research record keeps overlap, null evidence, claim, and review provisional', async ({ page }) => {
  await mockApi(page)
  await page.goto('/portal')
  await page.getByRole('link', { name: 'Map matched controls' }).press('Enter')
  await expect(page).toHaveURL(/\/research\/work-controls$/)
  await expect(page.getByRole('heading', { name: 'Map matched controls' })).toBeVisible()
  await expect(page.getByText('Provisional coordination record.')).toBeVisible()
  await expect(page.getByText('Synthetic matched-control reference')).toBeVisible()
  await expect(page.getByText('Submission ID').first()).toBeVisible()
  await expect(page.getByText('Methods').first()).toBeVisible()
  await expect(page.getByText('Claim hash').first()).toBeVisible()
  await expect(page.getByText('Reviewer').first()).toBeVisible()
  for (const [kind, anchor] of [
    ['supporting', '7'],
    ['contradicting', '1'],
    ['null', '11111111-1111-4111-8111-111111111111'],
    ['failed', 'prior-controls'],
    ['excluded', '8'],
  ]) {
    const item = page.locator(`[data-evidence-kind="${kind}"]`)
    await expect(item).toBeVisible()
    await expect(item).toContainText(anchor)
  }
  await expect(page.getByText('Canonical claim updated').first()).toBeVisible()
  await expect(page.getByText('null', { exact: true }).first()).toBeVisible()
  await expect(page.getByText('This fixture is an unreviewed null result only.')).toBeVisible()
  await expect(page.getByText('The bounded synthetic provenance is internally consistent.')).toBeVisible()
  await expect(page.getByText('decision requested')).toBeVisible()
  await expect(page.getByText('classification-controls').first()).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Relationships' })).toBeVisible()
  await expect(page.getByText('classifies', { exact: true }).first()).toBeVisible()
})

test('decision surfaces name empty, missing, and disabled states', async ({ page }) => {
  await mockApi(page, { '/decisions': [] })
  await page.goto('/portal/decisions')
  await expect(page.getByText('No decisions have been requested.')).toBeVisible()

  await page.goto('/portal/decisions/not-present')
  await expect(page.getByText(/decision was not found/i)).toBeVisible()

  await page.route('**/api/decisions', (route) => route.fulfill({
    status: 503,
    contentType: 'application/json',
    body: JSON.stringify({ detail: 'synthetic coordination disabled' }),
  }))
  await page.goto('/portal/decisions')
  await expect(page.getByText('Could not load decisions.')).toBeVisible()
})

test('decision and research views fit a 375px viewport without horizontal overflow', async ({ page }) => {
  await mockApi(page)
  await page.setViewportSize({ width: 375, height: 700 })
  await page.goto('/portal/decisions/decision-controls')
  await expect(page.getByRole('button', { name: 'Record guidance' })).toBeVisible()
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)

  await page.goto('/portal/research/work-controls')
  await expect(page.getByRole('heading', { name: 'Map matched controls' })).toBeVisible()
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)

  await page.setViewportSize({ width: 812, height: 375 })
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await expect(page.getByRole('link', { name: 'Workboard' })).toHaveCSS('transition-duration', '0s')
})
