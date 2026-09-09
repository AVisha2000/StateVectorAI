import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import { runInNewContext } from 'node:vm'
import { transformWithEsbuild } from 'vite'
import { renderToStaticMarkup } from 'react-dom/server'

const source = await readFile(new URL('./MotionHarbourEntry.jsx', import.meta.url), 'utf8')
const { code } = await transformWithEsbuild(source, 'MotionHarbourEntry.jsx', { loader: 'jsx', format: 'cjs', jsx: 'automatic' })
const module = { exports: {} }
runInNewContext(code, { module, exports: module.exports, require: createRequire(import.meta.url) })
const render = (hostname, studioId = 'physics') => renderToStaticMarkup(module.exports.default({ hostname, studioId }))

test('local Physics exposes the separate Unity island with truthful service scope', () => {
  for (const hostname of ['127.0.0.1', 'localhost']) {
    const markup = render(hostname)
    assert.match(markup, /href="http:\/\/127\.0\.0\.1:4178\/"/)
    assert.match(markup, /Explore Physics island/)
    assert.match(markup, /Local service required/)
    assert.doesNotMatch(markup, /target=/, 'navigation stays in the current tab')
  }
})

test('public and lookalike hosts cannot expose a visitor-local service link', () => {
  for (const hostname of ['statevectorai.com', 'preview.example', 'localhost.example', '127.0.0.1.example', '', undefined]) {
    assert.equal(render(hostname), '')
  }
})

test('the Unity route does not replace other disciplines or the existing quantum pilot', () => {
  for (const id of ['chemistry', 'quantum-island', 'classical', undefined]) {
    assert.equal(render('localhost', id ?? null), '')
  }
})
