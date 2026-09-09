import assert from 'node:assert/strict'
import test from 'node:test'
import { RESEARCH_STATUS, RESEARCH_WORLD_SNAPSHOT, agentMotionTimeAt, agentStrideAt, agentsForRegion, cappedPixelRatio, circlePatch, deferredCleanup, regionForAgent, researchStatus, researchWorldSnapshotIsValid, researchWorldView, setAgentMotionPaused } from './researchWorld.js'

test('snapshot status mapping is stable and unknown values stay neutral', () => {
  assert.deepEqual(Object.fromEntries(['planning', 'researching', 'reviewing', 'waiting'].map((status) => [status, researchStatus(status).label])), {
    planning: 'Planning', researching: 'Researching', reviewing: 'Reviewing', waiting: 'Waiting',
  })
  assert.equal(researchStatus('future-state').label, 'Status unavailable')
  assert.equal(researchStatus('future-state').color, RESEARCH_STATUS.unknown.color)
})

test('snapshot top-level validation is strict enough for the source seam', () => {
  assert.equal(researchWorldSnapshotIsValid(RESEARCH_WORLD_SNAPSHOT), true)
  assert.equal(researchWorldSnapshotIsValid({ schema_version: '1', regions: {}, agents: [] }), false)
  assert.equal(researchWorldSnapshotIsValid({ ...RESEARCH_WORLD_SNAPSHOT, regions: [{ ...RESEARCH_WORLD_SNAPSHOT.regions[0], latitude: 'bad' }] }), false)
  assert.equal(researchWorldSnapshotIsValid({ ...RESEARCH_WORLD_SNAPSHOT, agents: [{ ...RESEARCH_WORLD_SNAPSHOT.agents[0], has_question: 'yes' }] }), false)
  assert.equal(researchWorldSnapshotIsValid(null), false)
})

test('client contract accepts bounded long text and rejects schema boundary violations', () => {
  const longText = 'x'.repeat(500)
  const valid = { ...RESEARCH_WORLD_SNAPSHOT, agents: RESEARCH_WORLD_SNAPSHOT.agents.map((agent, index) => index === 0 ? { ...agent, summary: longText, question: longText } : agent) }
  assert.equal(researchWorldSnapshotIsValid(valid), true)
  assert.equal(researchWorldSnapshotIsValid({ ...valid, agents: valid.agents.map((agent, index) => index === 0 ? { ...agent, summary: `${longText}x` } : agent) }), false)
  assert.equal(researchWorldSnapshotIsValid({ ...valid, extra: true }), false)
  assert.equal(researchWorldSnapshotIsValid({ ...valid, regions: valid.regions.map((region, index) => index === 0 ? { ...region, extra: true } : region) }), false)
  assert.equal(researchWorldSnapshotIsValid({ ...valid, agents: valid.agents.map((agent, index) => index === 0 ? { ...agent, id: 'Bad Slug' } : agent) }), false)
  assert.equal(researchWorldSnapshotIsValid({ ...valid, agents: valid.agents.map((agent, index) => index === 0 ? { ...agent, summary: 'safe <markup>' } : agent) }), false)
  assert.equal(researchWorldSnapshotIsValid({ ...valid, generated_at: '2026-09-03T12:00:00' }), false)
  assert.equal(researchWorldSnapshotIsValid({ ...valid, generated_at: '2026-02-30T12:00:00Z' }), false)
  assert.equal(researchWorldSnapshotIsValid({ ...valid, generated_at: '2026-09-03T24:00:00Z' }), false)
  assert.equal(researchWorldSnapshotIsValid({ ...valid, generated_at: '2026-09-03T12:00:00+24:00' }), false)
  assert.equal(researchWorldSnapshotIsValid({ ...valid, generated_at: '2026-09-03T12:00:00+00:00' }), true)
})

test('Research World fixture contains exactly six agents in three valid regions', () => {
  const world = researchWorldView(RESEARCH_WORLD_SNAPSHOT)
  assert.equal(world.agents.length, 6)
  assert.equal(world.regions.length, 3)
  for (const agent of world.agents) assert.ok(regionForAgent(world, agent), `${agent.id} is unmapped`)
})

test('region allocation is explicit and the fixture has one reviewed question', () => {
  const world = researchWorldView(RESEARCH_WORLD_SNAPSHOT)
  assert.deepEqual(world.regions.map((region) => agentsForRegion(world, region.id).length), [2, 2, 2])
  assert.equal(world.agents.filter((agent) => agent.has_question).length, 1)
  assert.equal(world.unmapped_agent_count, 0)
  assert.equal(world.costLabel, 'Cost data unavailable')
})

test('empty snapshots are rejected by the source contract', () => {
  assert.throws(() => researchWorldView({ ...RESEARCH_WORLD_SNAPSHOT, regions: [], agents: [], unmapped_agent_count: 0, allocation: { metric: 'active_agent_count', by_region: {} }, cost: { ...RESEARCH_WORLD_SNAPSHOT.cost, coverage: { included_agents: 0, total_agents: 0 } } }), /Unsupported Research World snapshot/)
})

test('motion, render coordination, and pixel ratio policies are deterministic', () => {
  assert.notEqual(agentStrideAt(0, 10), agentStrideAt(330, 10))
  assert.equal(cappedPixelRatio(0.75), 1)
  assert.equal(cappedPixelRatio(1.25), 1.25)
  assert.equal(cappedPixelRatio(3), 1.5)
})

test('manual motion pause resumes from the frozen phase instead of jumping to wall time', () => {
  let clock = { paused: false, pausedAt: null, pausedDuration: 0 }
  assert.equal(agentMotionTimeAt(clock, 1000), 1000)
  clock = setAgentMotionPaused(clock, true, 1000)
  assert.equal(agentMotionTimeAt(clock, 5000), 1000)
  clock = setAgentMotionPaused(clock, false, 5000)
  assert.equal(agentMotionTimeAt(clock, 6000), 2000)
  assert.equal(agentStrideAt(agentMotionTimeAt(clock, 6000), 10), agentStrideAt(2000, 10))
})

test('region patches are closed clockwise rings instead of globe-sized complements', () => {
  const ring = circlePatch(35, -30, 20).coordinates[0]
  assert.deepEqual(ring[0], ring.at(-1))
  const signedArea = ring.slice(0, -1).reduce((area, point, index) => {
    const next = ring[(index + 1) % (ring.length - 1)]
    return area + point[0] * next[1] - next[0] * point[1]
  }, 0) / 2
  assert.ok(signedArea < 0)
})

test('deferred cleanup survives a StrictMode cleanup-remount cycle and disposes on real unmount', () => {
  const jobs = new Map()
  let nextId = 0
  let disposals = 0
  const schedule = (callback) => { nextId += 1; jobs.set(nextId, callback); return nextId }
  const cancel = (id) => jobs.delete(id)
  const cleanup = deferredCleanup(() => { disposals += 1 }, schedule, cancel)

  cleanup.mount()
  cleanup.unmount()
  cleanup.mount()
  for (const callback of jobs.values()) callback()
  assert.equal(disposals, 0)

  cleanup.unmount()
  for (const callback of [...jobs.values()]) callback()
  assert.equal(disposals, 1)
})
