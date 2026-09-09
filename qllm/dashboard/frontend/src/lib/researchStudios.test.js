import test from 'node:test'
import assert from 'node:assert/strict'
import { RESEARCH_WORLD_SNAPSHOT, researchWorldView } from './researchWorld.js'
import { studiosForWorld, withResearcher, meetingReply, launchHeight } from './researchStudios.js'

test('concept studios are disclosed and are never presented as live agents', () => {
  const world = researchWorldView(RESEARCH_WORLD_SNAPSHOT),
    studios = studiosForWorld(world)
  assert.equal(studios.length, 6)
  assert.deepEqual(
    studios.filter((s) => s.preview).map((s) => s.id),
    ['chemistry', 'biology', 'ai-safety'],
  )
  assert.ok(studios.filter((s) => s.preview).every((s) => !s.agent && s.status === 'Concept studio'))
  assert.equal(studiosForWorld({ ...world, mode: 'live' }).length, 3)
  assert.deepEqual(studiosForWorld({ ...world, mode: 'live', agents: [] }), [])
})
test('selecting a second researcher preserves that researcher’s question and identity', () => {
  const world = researchWorldView(RESEARCH_WORLD_SNAPSHOT),
    studios = studiosForWorld(world)
  const nova = world.agents.find((agent) => agent.id === 'nova-ember')
  const selected = withResearcher(
    studios.find((s) => s.id === 'physics'),
    nova,
  )
  assert.equal(selected.name, 'Nova Ember')
  assert.equal(selected.title, nova.title)
  assert.equal(selected.agent.id, nova.id)
  assert.equal(selected.status, 'Reviewing')
})
test('an unfamiliar live discipline keeps its own domain and researcher', () => {
  const world = researchWorldView(RESEARCH_WORLD_SNAPSHOT)
  const studio = studiosForWorld({
    ...world,
    mode: 'live',
    regions: [{ id: 'climate', label: 'Climate', color: '#fff' }],
    agents: [{ ...world.agents[0], regionId: 'climate' }],
  })[0]
  assert.equal(studio.label, 'Climate')
  assert.equal(studio.name, world.agents[0].displayName)
  assert.equal(studio.preview, false)
})
test('scripted replies describe drafts and working notes without claiming execution', () => {
  const studio = studiosForWorld(researchWorldView(RESEARCH_WORLD_SNAPSHOT))[0]
  assert.equal(meetingReply(studio, 'Show me a paper').action, 'paper')
  assert.match(meetingReply(studio, 'Run this analysis').text, /draft/)
  assert.match(meetingReply(studio, 'Why does that follow?').text, /scripted responses/)
})
test('rocket launch has a deterministic rest, ascent, and periodic reset', () => {
  assert.equal(launchHeight(5), 0)
  assert.equal(launchHeight(6), 0)
  assert.equal(launchHeight(8), 0.72)
  assert.equal(launchHeight(10), 0)
  assert.equal(launchHeight(22), launchHeight(8))
})
