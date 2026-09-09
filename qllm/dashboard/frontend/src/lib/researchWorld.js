export const RESEARCH_WORLD_SNAPSHOT = Object.freeze({
  schema_version: '1', generated_at: '2026-09-03T12:00:00Z', mode: 'fixture',
  regions: [
    { id: 'physics', label: 'Physics', description: 'Questions about physical systems and their models.', latitude: 35, longitude: -30, radius: 20, color: 'violet' },
    { id: 'mathematics', label: 'Mathematics', description: 'Questions about formal structures and methods.', latitude: 12, longitude: 40, radius: 18, color: 'gold' },
    { id: 'machine-learning', label: 'Machine Learning', description: 'Questions about learning systems and evaluation.', latitude: -25, longitude: 105, radius: 22, color: 'cyan' },
  ],
  agents: [
    { id: 'lyra-vale', display_name: 'Lyra Vale', region_id: 'physics', latitude: 39, longitude: -24, status: 'researching', title: 'Compare boundary conditions', summary: 'Preparing a bounded comparison of candidate boundary conditions.', has_question: true, question: 'Which boundary condition should we review before the next comparison?', updated_at: '2026-09-03T12:00:00Z' },
    { id: 'nova-ember', display_name: 'Nova Ember', region_id: 'physics', latitude: 30, longitude: -35, status: 'reviewing', title: 'Check model assumptions', summary: 'Reviewing the assumptions recorded for a small physical-system model.', has_question: false, question: null, updated_at: '2026-09-03T12:00:00Z' },
    { id: 'orion-reed', display_name: 'Orion Reed', region_id: 'mathematics', latitude: 16, longitude: 47, status: 'planning', title: 'Frame a proof sketch', summary: 'Planning a concise proof sketch and its required definitions.', has_question: false, question: null, updated_at: '2026-09-03T12:00:00Z' },
    { id: 'mira-quill', display_name: 'Mira Quill', region_id: 'mathematics', latitude: 7, longitude: 35, status: 'researching', title: 'Test a formal mapping', summary: 'Testing whether a formal mapping preserves the stated constraints.', has_question: false, question: null, updated_at: '2026-09-03T12:00:00Z' },
    { id: 'sol-hart', display_name: 'Sol Hart', region_id: 'machine-learning', latitude: -20, longitude: 111, status: 'researching', title: 'Inspect evaluation controls', summary: 'Inspecting matched evaluation controls for a learning-system study.', has_question: false, question: null, updated_at: '2026-09-03T12:00:00Z' },
    { id: 'echo-rowan', display_name: 'Echo Rowan', region_id: 'machine-learning', latitude: -31, longitude: 98, status: 'waiting', title: 'Await review input', summary: 'Waiting for an advisory review before the next planning step.', has_question: false, question: null, updated_at: '2026-09-03T12:00:00Z' },
  ],
  unmapped_agent_count: 0,
  allocation: { metric: 'active_agent_count', by_region: { physics: 2, mathematics: 2, 'machine-learning': 2 } },
  cost: { availability: 'unavailable', metric: null, unit: null, window: null, source: null, coverage: { included_agents: 0, total_agents: 6 }, by_region: null },
})

const REGION_COLORS = Object.freeze({ violet: '#b9a4ff', gold: '#f4d36d', cyan: '#62d4ff' })
export const RESEARCH_STATUS = Object.freeze({
  planning: Object.freeze({ label: 'Planning', color: '#f4d36d' }),
  researching: Object.freeze({ label: 'Researching', color: '#62d4ff' }),
  reviewing: Object.freeze({ label: 'Reviewing', color: '#b9a4ff' }),
  waiting: Object.freeze({ label: 'Waiting', color: '#a9b8c8' }),
  unknown: Object.freeze({ label: 'Status unavailable', color: '#a9b8c8' }),
})
export const RESEARCH_STATUS_ORDER = Object.freeze(['planning', 'researching', 'reviewing', 'waiting'])

export function researchStatus(status) { return RESEARCH_STATUS[status] || RESEARCH_STATUS.unknown }
export function researchWorldSnapshotIsValid(snapshot) {
  const object = (value) => value && typeof value === 'object' && !Array.isArray(value)
  const exact = (value, keys) => object(value) && Object.keys(value).length === keys.length && keys.every((key) => Object.prototype.hasOwnProperty.call(value, key))
  const text = (value, max) => typeof value === 'string' && value.length >= 1 && value.length <= max && !/[\u0000-\u001f<>]/.test(value)
  const slug = (value) => text(value, 80) && /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(value)
  const timestamp = (value) => {
    if (value === null) return true
    if (typeof value !== 'string') return false
    const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(Z|[+-](\d{2}):(\d{2}))$/.exec(value)
    if (!match) return false
    const [, yearText, monthText, dayText, hourText, minuteText, secondText, zone, offsetHourText, offsetMinuteText] = match
    const year = Number(yearText), month = Number(monthText), day = Number(dayText)
    const hour = Number(hourText), minute = Number(minuteText), second = Number(secondText)
    const daysInMonth = new Date(Date.UTC(year, month, 0)).getUTCDate()
    if (year < 1 || year > 9999 || month < 1 || month > 12 || day < 1 || day > daysInMonth || hour > 23 || minute > 59 || second > 59) return false
    if (zone !== 'Z' && (Number(offsetHourText) > 23 || Number(offsetMinuteText) > 59)) return false
    return Number.isFinite(Date.parse(value))
  }
  const integer = (value) => Number.isInteger(value) && value >= 0
  const regionValid = (region) => exact(region, ['id', 'label', 'description', 'latitude', 'longitude', 'radius', 'color']) && slug(region.id) && text(region.label, 80) && text(region.description, 240) && Number.isFinite(region.latitude) && region.latitude >= -90 && region.latitude <= 90 && Number.isFinite(region.longitude) && region.longitude >= -180 && region.longitude <= 180 && Number.isFinite(region.radius) && region.radius > 0 && region.radius <= 90 && ['violet', 'gold', 'cyan'].includes(region.color)
  const agentValid = (agent) => exact(agent, ['id', 'display_name', 'region_id', 'latitude', 'longitude', 'status', 'title', 'summary', 'has_question', 'question', 'updated_at']) && slug(agent.id) && text(agent.display_name, 80) && slug(agent.region_id) && Number.isFinite(agent.latitude) && agent.latitude >= -90 && agent.latitude <= 90 && Number.isFinite(agent.longitude) && agent.longitude >= -180 && agent.longitude <= 180 && ['planning', 'researching', 'reviewing', 'waiting'].includes(agent.status) && text(agent.title, 160) && text(agent.summary, 500) && typeof agent.has_question === 'boolean' && (agent.has_question ? text(agent.question, 500) : agent.question === null) && timestamp(agent.updated_at)
  if (!exact(snapshot, ['schema_version', 'generated_at', 'mode', 'regions', 'agents', 'unmapped_agent_count', 'allocation', 'cost']) || snapshot.schema_version !== '1' || !['fixture', 'live'].includes(snapshot.mode) || !timestamp(snapshot.generated_at) || !Array.isArray(snapshot.regions) || snapshot.regions.length < 1 || snapshot.regions.length > 20 || !snapshot.regions.every(regionValid) || new Set(snapshot.regions.map((region) => region.id)).size !== snapshot.regions.length || !Array.isArray(snapshot.agents) || snapshot.agents.length > 200 || !snapshot.agents.every(agentValid) || new Set(snapshot.agents.map((agent) => agent.id)).size !== snapshot.agents.length) return false
  const regionIds = new Set(snapshot.regions.map((region) => region.id))
  if (!snapshot.agents.every((agent) => regionIds.has(agent.region_id)) || !integer(snapshot.unmapped_agent_count)) return false
  if (!exact(snapshot.allocation, ['metric', 'by_region']) || snapshot.allocation.metric !== 'active_agent_count' || !object(snapshot.allocation.by_region)) return false
  const counts = Object.fromEntries(snapshot.regions.map((region) => [region.id, 0]))
  snapshot.agents.forEach((agent) => { counts[agent.region_id] += 1 })
  if (!sameKeys(snapshot.allocation.by_region, counts) || Object.entries(snapshot.allocation.by_region).some(([id, count]) => count !== counts[id] || !integer(count))) return false
  const cost = snapshot.cost
  if (!exact(cost, ['availability', 'metric', 'unit', 'window', 'source', 'coverage', 'by_region']) || !['unavailable', 'recorded'].includes(cost.availability) || !exact(cost.coverage, ['included_agents', 'total_agents']) || !integer(cost.coverage.included_agents) || !integer(cost.coverage.total_agents) || cost.coverage.total_agents !== snapshot.agents.length || cost.coverage.included_agents > cost.coverage.total_agents) return false
  if (cost.availability === 'unavailable') return cost.metric === null && cost.unit === null && cost.window === null && cost.source === null && cost.by_region === null && cost.coverage.included_agents === 0
  return text(cost.metric, 120) && text(cost.unit, 80) && text(cost.window, 120) && text(cost.source, 160) && object(cost.by_region) && sameKeys(cost.by_region, counts) && Object.values(cost.by_region).every((value) => Number.isFinite(value) && value >= 0)
}

function sameKeys(left, right) {
  const leftKeys = Object.keys(left).sort()
  const rightKeys = Object.keys(right).sort()
  return leftKeys.length === rightKeys.length && leftKeys.every((key, index) => key === rightKeys[index])
}

export function researchWorldView(snapshot) {
  if (!researchWorldSnapshotIsValid(snapshot)) throw new Error('Unsupported Research World snapshot.')
  const regions = snapshot.regions.map((region) => ({ ...region, color: REGION_COLORS[region.color] || '#eef5ff' }))
  const regionIds = new Set(regions.map((region) => region.id))
  const agents = snapshot.agents.filter((agent) => regionIds.has(agent.region_id)).map((agent) => ({ ...agent, displayName: agent.display_name, regionId: agent.region_id, updatedAt: agent.updated_at ? 'This snapshot' : 'Unavailable', statusLabel: researchStatus(agent.status).label, statusColor: researchStatus(agent.status).color }))
  return { ...snapshot, regions, agents, costLabel: snapshot.cost?.availability === 'recorded' ? 'Recorded cost data' : 'Cost data unavailable' }
}

export function regionForAgent(world, agent) { return world.regions.find((region) => region.id === agent.regionId) || null }
export function agentsForRegion(world, regionId) { return world.agents.filter((agent) => agent.regionId === regionId) }

export function agentStrideAt(timeMs, latitude) { return Math.sin(timeMs / 330 + latitude) }
export function agentMotionTimeAt(clock, nowMs) { return (clock.paused ? clock.pausedAt : nowMs) - clock.pausedDuration }
export function setAgentMotionPaused(clock, paused, nowMs) {
  if (clock.paused === paused) return clock
  if (paused) return { ...clock, paused: true, pausedAt: nowMs }
  return { paused: false, pausedAt: null, pausedDuration: clock.pausedDuration + Math.max(0, nowMs - clock.pausedAt) }
}
export function cappedPixelRatio(devicePixelRatio) { return Math.min(Math.max(Number(devicePixelRatio) || 1, 1), 1.5) }

export function deferredCleanup(cleanup, schedule = (callback) => setTimeout(callback, 0), cancel = clearTimeout) {
  let handle = null
  return {
    mount() {
      if (handle === null) return
      cancel(handle)
      handle = null
    },
    unmount() {
      if (handle !== null) cancel(handle)
      handle = schedule(() => { handle = null; cleanup() })
    },
  }
}

export function circlePatch(latitude, longitude, radius) {
  const points = []
  for (let step = 0; step < 18; step += 1) {
    const angle = -(step / 18) * Math.PI * 2
    points.push([longitude + Math.cos(angle) * radius, latitude + Math.sin(angle) * radius * 0.68])
  }
  points.push(points[0])
  return { type: 'Polygon', coordinates: [points] }
}
