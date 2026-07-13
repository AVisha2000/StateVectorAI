import { test, expect } from '@playwright/test'
import { mockApi } from './fixtures.js'

test.beforeEach(async ({ page }) => { await mockApi(page) })

const circuit = (page) => page.locator('.circuit-wrap svg')
const select = (page, label) => page.locator('label').filter({ hasText: label }).locator('select')

test('Designer: circuit renders and each ansatz produces a distinct layout', async ({ page }) => {
  await page.goto('/designer')
  await expect(circuit(page)).toBeVisible()
  // hardware_efficient → RY gates
  await expect(circuit(page).getByText('RY').first()).toBeVisible()
  // reuploading → RX + CZ
  await select(page, 'Ansatz').selectOption('reuploading')
  await expect(circuit(page).getByText('RX').first()).toBeVisible()
  await expect(circuit(page).getByText('CZ').first()).toBeVisible()
  // ising → ZZ
  await select(page, 'Ansatz').selectOption('ising')
  await expect(circuit(page).getByText('ZZ').first()).toBeVisible()
})

test('Designer: readout offers only registry values (z, zz — never all)', async ({ page }) => {
  await page.goto('/designer')
  const options = await select(page, 'Readout').locator('option').allTextContents()
  expect(options.sort()).toEqual(['z', 'zz'])
})

test('Designer: ising (QRNN-only) pins backend/readout to compatibility values', async ({ page }) => {
  await page.goto('/designer')
  await select(page, 'Ansatz').selectOption('ising')
  // architecture=qrnn requirement surfaces, backend/readout lock to pennylane/z
  await expect(page.getByText(/QRNN-only family/i)).toBeVisible()
  await expect(select(page, 'Backend')).toBeDisabled()
  await expect(select(page, 'Backend')).toHaveValue('pennylane')
  await expect(select(page, 'Readout')).toBeDisabled()
  await expect(select(page, 'Readout')).toHaveValue('z')
})

test('Designer: tensorcircuit_mps demands a max bond dimension and labels it approximate', async ({ page }) => {
  await page.goto('/designer')
  await select(page, 'Backend').selectOption('tensorcircuit_mps')
  const bond = page.locator('label').filter({ hasText: 'Max bond dim' }).locator('input')
  await expect(bond).toBeVisible()
  await expect(page.getByText(/approximate/i).first()).toBeVisible()
  // other backends do not show the control
  await select(page, 'Backend').selectOption('pennylane')
  await expect(bond).toHaveCount(0)
})

test('Designer: live round-trip validates and shows registry-derived params', async ({ page }) => {
  await page.goto('/designer')
  await expect(page.locator('.card', { hasText: 'Round-trip' }).getByText('live')).toBeVisible()
  await page.getByRole('button', { name: /Validate against registry/i }).click()
  await expect(page.getByText(/✓ valid — registry-backed, side-effect-free/i)).toBeVisible()
  // the server-derived parameter count is authoritative and labeled as registry
  const props = page.locator('.card', { hasText: 'Properties' })
  await expect(props.getByText('registry', { exact: true })).toBeVisible()
  await expect(props.locator('.metric-row', { hasText: 'Trainable params' }).locator('b')).toHaveText('24')
})

test('Designer: a registry rejection is shown as a rejection, not an outage', async ({ page }) => {
  await mockApi(page, {
    'POST /designer/circuit': { status: 400, body: { detail: "ansatz 'ising' requires architecture='qrnn'." } },
  })
  await page.goto('/designer')
  await page.getByRole('button', { name: /Validate against registry/i }).click()
  await expect(page.getByText(/Rejected by the registry:.*requires architecture/i)).toBeVisible()
})

test('Designer: degrades gracefully when the endpoint is absent (older backend)', async ({ page }) => {
  await mockApi(page, { '/designer/circuit': null })
  await page.goto('/designer')
  await page.getByRole('button', { name: /Validate against registry/i }).click()
  await expect(page.getByText(/doesn.t serve/i)).toBeVisible()
})

test('Designer: properties update with size; classical toggle swaps the panel', async ({ page }) => {
  await page.goto('/designer')
  await expect(page.locator('.metric-row', { hasText: 'Param gates (drawn)' })).toBeVisible()
  await expect(page.locator('.metric-row', { hasText: 'Entangling gates (drawn)' })).toBeVisible()
  await page.getByRole('button', { name: 'Classical' }).click()
  await expect(page.getByText(/width-matched classical twin/i)).toBeVisible()
})

test('Designer → Bench: Send to Bench carries the circuit as a prefill', async ({ page }) => {
  await page.goto('/designer')
  await select(page, 'Ansatz').selectOption('reuploading')
  await page.getByRole('button', { name: /Send to Bench/i }).click()
  await expect(page).toHaveURL(/\/bench$/)
  await expect(page.getByText(/From the Designer/i)).toBeVisible()
  await expect(page.locator('.notice', { hasText: 'reuploading' })).toBeVisible()
})
