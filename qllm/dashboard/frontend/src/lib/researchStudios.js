// Presentation metadata, never an authority for research results or live state.
// A playable destination, deliberately separate from snapshot research agents.
export const QUANTUM_ISLAND = Object.freeze({
  id: 'quantum-island', kind: 'quantum-game', label: 'Quantum computing', short: 'Quantum',
  color: '#c0a7f0', name: 'Quantum island', activity: 'Steer a quantum state',
  title: 'A small state. A new possibility.',
  description: 'Step onto the island, steer a qubit and turn a different path into your next experiment.',
  position: [-0.59, -0.47, 0.65], agent: null, preview: true, status: 'Playable prototype',
})

export const STUDIO_TYPES = [
  {
    id: 'mathematics',
    label: 'Mathematics',
    short: 'Maths',
    color: '#ebbd78',
    name: 'Mira',
    activity: 'At the chalkboard',
    title: 'Finding the shape of a proof',
    description:
      'A quiet place to explore definitions, test a conjecture, and work through a proof together.',
    position: [-0.54, 0.53, 0.65],
  },
  {
    id: 'physics',
    label: 'Physics',
    short: 'Physics',
    color: '#e3a18b',
    name: 'Lyra',
    activity: 'Modelling a trajectory',
    title: 'A little curiosity. A longer flight.',
    description: 'Change an assumption, follow the trajectory, and ask what the model leaves out.',
    position: [0.52, 0.53, 0.66],
  },
  {
    id: 'chemistry',
    label: 'Chemistry',
    short: 'Chemistry',
    color: '#b4a0e5',
    name: 'Iris',
    activity: 'At the reaction bench',
    title: 'Where small changes react',
    description: 'Explore reaction models and the evidence needed to distinguish a useful explanation.',
    position: [-0.78, -0.15, 0.6],
  },
  {
    id: 'biology',
    label: 'Biology',
    short: 'Biology',
    color: '#8cbda0',
    name: 'Fern',
    activity: 'Exploring living systems',
    title: 'Life, one question at a time',
    description: 'Study a tiny living world, trace a hypothesis, and make room for unexpected observations.',
    position: [0.63, -0.31, 0.71],
  },
  {
    id: 'ai-safety',
    label: 'AI safety',
    short: 'AI safety',
    color: '#8ec4d2',
    name: 'Sage',
    activity: 'Testing a robot',
    title: 'How do we know an agent is safe?',
    description:
      'Inspect assumptions, challenge a boundary, and think through a controlled safety evaluation.',
    position: [0.01, -0.64, 0.77],
  },
  {
    id: 'machine-learning',
    label: 'Machine learning',
    short: 'Machine learning',
    color: '#93a9e3',
    name: 'Sol',
    activity: 'Inspecting a neural network',
    title: 'Learning how learning works',
    description: 'Explore representations, compare controls, and look for the simplest explanation.',
    position: [0.04, 0.02, 1],
  },
]

export function studiosForWorld(world) {
  const types =
    world.mode === 'fixture'
      ? STUDIO_TYPES
      : world.regions.map(
          (region, index) =>
            STUDIO_TYPES.find((studio) => studio.id === region.id) || {
              ...STUDIO_TYPES[index % STUDIO_TYPES.length],
              id: region.id,
              label: region.label,
              short: region.label,
              color: region.color,
            },
        )
  return types
    .map((studio) => {
      const candidates = world.agents.filter((agent) => agent.regionId === studio.id)
      const agent = candidates.find((candidate) => candidate.status === 'researching') || candidates[0]
      return {
        ...studio,
        agent: agent || null,
        name: agent?.displayName || studio.name,
        title: agent?.title || studio.title,
        description: agent?.summary || studio.description,
        preview: !agent,
        status: agent?.statusLabel || 'Concept studio',
      }
    })
    .filter((studio) => world.mode === 'fixture' || studio.agent)
}

export function studioForAgent(studios, agent) {
  return studios.find((studio) => studio.id === agent.regionId)
}

export function withResearcher(studio, agent) {
  if (!studio || !agent || studio.id !== agent.regionId) return studio
  return {
    ...studio,
    agent,
    name: agent.displayName,
    title: agent.title,
    description: agent.summary,
    preview: false,
    status: agent.statusLabel,
  }
}

export function meetingReply(studio, question) {
  const q = question.toLowerCase()
  if (/paper|read|source|evidence/.test(q))
    return {
      text: 'Let’s put a working note on the table. You can read it alongside our conversation. It is an example note for this interface, with assumptions and open questions clearly marked.',
      action: 'paper',
    }
  if (/try|test|approach|what if|run|code|analysis/.test(q))
    return {
      text: 'Let’s turn that into a testable question. I’ve opened an experiment draft with your idea. In a connected research session, you would review the method and budget before a worker starts.',
      action: 'experiment',
    }
  if (/coffee|hello|hi\b/.test(q))
    return {
      text: `Welcome to the ${studio.label.toLowerCase()} studio. There’s a seat for you. We can explore the current question, read a working note together, or sketch the next test.`,
      action: null,
    }
  return {
    text: `I’ve kept your question in this session. A useful next step is to make the assumption explicit, identify what evidence would change our view, and choose a small test. This preview uses scripted responses; a research model is not connected yet.`,
    action: null,
  }
}

export function launchHeight(time) {
  const phase = ((time % 14) + 14) % 14
  if (phase < 6) return 0
  if (phase < 10) return 0.18 * (phase - 6) ** 2
  return 0 // The rocket respawns out of view after a brief launch.
}
