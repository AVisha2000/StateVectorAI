import test from 'node:test'

import { assertCoordinationFixtureContract } from '../e2e/fixtures.js'

test('coordination browser fixtures match evidence anchors and review lifecycle invariants', () => {
  assertCoordinationFixtureContract()
})
