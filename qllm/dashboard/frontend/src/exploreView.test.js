import assert from 'node:assert/strict'
import test from 'node:test'

import { studiesForDomain, taskLinkForStudy } from './exploreView.js'

const studies = [
  { id: 'qnlp-study', domain: 'QNLP', task: 'Language modelling' },
  { id: 'memory-study', domain: 'Sequence memory', task: 'Sequence memory' },
]

test('study groups follow the selected Explore domain', () => {
  assert.deepEqual(studiesForDomain(studies, { name: 'QNLP' }), [studies[0]])
  assert.deepEqual(studiesForDomain(studies, null), studies)
  assert.deepEqual(studiesForDomain(null, { name: 'QNLP' }), [])
})

test('inferred groups link only through a canonical matching task route', () => {
  const tasks = [{
    domain: 'QNLP',
    domain_slug: 'qnlp',
    name: 'Language modelling',
    slug: 'language-modelling',
  }]
  assert.equal(
    taskLinkForStudy(studies[0], tasks),
    '/explore/task/language-modelling?domain=qnlp',
  )
  assert.equal(taskLinkForStudy(studies[1], tasks), null)
})
