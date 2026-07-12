import { test, expect } from '@playwright/test'
import { mockApi } from './fixtures.js'

test.beforeEach(async ({ page }) => { await mockApi(page) })

const circuit = (page) => page.locator('.circuit-wrap svg')

test('Designer: circuit renders and each ansatz produces a distinct layout', async ({ page }) => {
  await page.goto('/designer')
  await expect(circuit(page)).toBeVisible()
  // hardware_efficient → RY gates
  await expect(circuit(page).getByText('RY').first()).toBeVisible()
  // reuploading → RX + CZ
  await page.locator('select').first().selectOption('reuploading')
  await expect(circuit(page).getByText('RX').first()).toBeVisible()
  await expect(circuit(page).getByText('CZ').first()).toBeVisible()
  // ising → ZZ
  await page.locator('select').first().selectOption('ising')
  await expect(circuit(page).getByText('ZZ').first()).toBeVisible()
})

test('Designer: properties update with size; classical toggle swaps the panel', async ({ page }) => {
  await page.goto('/designer')
  await expect(page.locator('.metric-row', { hasText: 'Trainable params' })).toBeVisible()
  await expect(page.locator('.metric-row', { hasText: 'Entangling gates' })).toBeVisible()
  await page.getByRole('button', { name: 'Classical' }).click()
  await expect(page.getByText(/width-matched classical twin/i)).toBeVisible()
})

test('Designer: Validate against the proposed round-trip degrades gracefully', async ({ page }) => {
  await page.goto('/designer')
  await page.getByRole('button', { name: /Validate against registry/i }).click()
  await expect(page.getByText(/isn.t on this branch yet|not computed|valid circuit spec/i)).toBeVisible()
})

test('Designer → Bench: Send to Bench carries the circuit as a prefill', async ({ page }) => {
  await page.goto('/designer')
  await page.locator('select').first().selectOption('reuploading')
  await page.getByRole('button', { name: /Send to Bench/i }).click()
  await expect(page).toHaveURL(/\/bench$/)
  await expect(page.getByText(/From the Designer/i)).toBeVisible()
  await expect(page.locator('.notice', { hasText: 'reuploading' })).toBeVisible()
})
