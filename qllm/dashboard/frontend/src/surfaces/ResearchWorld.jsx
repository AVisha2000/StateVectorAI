import { Component, lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { PORTAL_BASE } from '../appShell.js'
import { RESEARCH_WORLD_SNAPSHOT, researchWorldView } from '../lib/researchWorld.js'
import { studiosForWorld, withResearcher, QUANTUM_ISLAND } from '../lib/researchStudios.js'
import { useResearchWorldSnapshot } from '../lib/hooks.js'
import Icon from '../components/WorldIcon.jsx'
import MotionHarbourEntry from '../components/MotionHarbourEntry.jsx'
import '../research-world.css'

const ResearchPlanet = lazy(() => import('../components/ResearchPlanet.jsx'))
const ResearchSession = lazy(() => import('../components/ResearchSession.jsx'))
const QuantumIslandGame = lazy(() => import('../components/QuantumIslandGame.jsx'))
function useReducedMotion() {
  const [reduced, setReduced] = useState(
    () => globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true,
  )
  useEffect(() => {
    const query = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)')
    const update = () => setReduced(query.matches)
    query?.addEventListener('change', update)
    return () => query?.removeEventListener('change', update)
  }, [])
  return reduced
}

export default function ResearchWorld({ snapshot }) {
  const query = useResearchWorldSnapshot({ injectedSnapshot: snapshot })
  const world = useMemo(() => researchWorldView(query.data || RESEARCH_WORLD_SNAPSHOT), [query.data])
  const studios = useMemo(() => [...studiosForWorld(world), QUANTUM_ISLAND], [world])
  const reducedMotion = useReducedMotion(),
    [paused, setPaused] = useState(reducedMotion)
  const [selectedId, setSelectedId] = useState(null),
    [command, setCommand] = useState(null),
    [meeting, setMeeting] = useState(null),
    [view, setView] = useState('world'),
    [search, setSearch] = useState(''),
    [selectedAgentId, setSelectedAgentId] = useState(null)
  const [sessions, setSessions] = useState({}),
    studioButtons = useRef(new Map()),
    meetingTrigger = useRef(null)
  const selected = withResearcher(
    studios.find((studio) => studio.id === selectedId),
    world.agents.find((agent) => agent.id === selectedAgentId),
  )
  const issue = (action) => setCommand({ ...action, nonce: Date.now() })
  const select = (studio, agent = null) => {
    if (!studio) return
    setSelectedId(studio.id)
    setSelectedAgentId(agent?.id || null)
    issue({ type: 'focus', id: studio.id })
    if (window.matchMedia('(max-width: 600px)').matches) {
      requestAnimationFrame(() =>
        document
          .querySelector('.rw-field-note-selected')
          ?.scrollIntoView({ behavior: reducedMotion ? 'instant' : 'smooth', block: 'start' }),
      )
    }
  }
  const reset = () => {
    setSelectedId(null)
    issue({ type: 'reset' })
  }
  const enter = (studio, event) => {
    if (!studio) return
    meetingTrigger.current = event?.currentTarget || document.activeElement
    setMeeting(studio)
  }
  const leave = () => {
    setMeeting(null)
    requestAnimationFrame(() => meetingTrigger.current?.focus())
  }
  useEffect(() => {
    if (reducedMotion) setPaused(true)
  }, [reducedMotion])
  useEffect(() => {
    const key = (event) => {
      if (event.key === 'Escape' && !meeting && selectedId) {
        const id = selectedId
        setSelectedId(null)
        studioButtons.current.get(id)?.focus()
      }
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [selectedId, meeting])
  const visibleStudios = studios.filter((studio) =>
    `${studio.label} ${studio.name} ${studio.title}`.toLowerCase().includes(search.toLowerCase()),
  )
  return (
    <main className="rw" id="main-content">
      <a className="rw-skip" href="#research-studios">
        Skip to research studios
      </a>
      <header className="rw-header">
        <a className="rw-brand" href="/" aria-label="StateVectorAI home">
          <span className="rw-logomark">
            <Icon name="world" size={26} />
          </span>
          statevector<span>ai</span>
          <sup>LABS</sup>
        </a>
        <nav aria-label="Main navigation">
          <button className={view === 'world' ? 'active' : ''} onClick={() => setView('world')}>
            <Icon name="world" size={16} />
            The world
          </button>
          <button
            className={view === 'studios' ? 'active' : ''}
            onClick={() => {
              setView('studios')
              document
                .getElementById('research-studios')
                ?.scrollIntoView({ behavior: reducedMotion ? 'instant' : 'smooth' })
            }}
          >
            Research studios
          </button>
          <a href={`${PORTAL_BASE}/library`}>
            Library <span>↗</span>
          </a>
        </nav>
        <a className="rw-workspace" href={PORTAL_BASE}>
          Open workspace <Icon name="arrow" size={17} />
        </a>
      </header>
      <section className="rw-universe" aria-labelledby="research-world-title">
        <div className="rw-stars" aria-hidden="true" />
        <div className="rw-orbit rw-orbit-one" aria-hidden="true" />
        <div className="rw-orbit rw-orbit-two" aria-hidden="true" />
        <div className="rw-intro">
          <p className="rw-eyebrow">
            <span className="rw-spark" /> A PLACE FOR CURIOUS MINDS
          </p>
          <h1 id="research-world-title">
            A world of
            <br />
            <em>curiosity.</em>
          </h1>
          <p className="rw-description">
            Big questions. Small worlds.
            <br />A new way to do research, together.
          </p>
          <p className="rw-invitation">
            Find a researcher. Pull up a chair.
            <br />
            See where a question takes you.
          </p>
          <button
            className="rw-text-link"
            onClick={() => select(studios.find((studio) => studio.id === 'ai-safety') || studios[0])}
          >
            Explore the world <Icon name="arrow" size={18} />
          </button>
          <div className="rw-preview-label">
            <span />
            {world.mode === 'fixture' ? 'Interactive preview' : 'Research snapshot'}
            <span className="rw-preview-detail">Illustrative activity</span>
          </div>
        </div>
        <div className="rw-scene" aria-label="Interactive research world">
          <SceneBoundary>
            <Suspense
              fallback={
                <div className="rw-loading" role="status">
                  Growing a little world…
                </div>
              }
            >
              <ResearchPlanet
                studios={studios}
                selectedId={selectedId}
                onSelect={select}
                paused={paused || Boolean(meeting)}
                reducedMotion={reducedMotion}
                command={command}
              />
            </Suspense>
          </SceneBoundary>
        </div>
        <aside className={`rw-field-note ${selected ? 'rw-field-note-selected' : ''}`} aria-live="polite">
          {selected ? (
            <>
              <div className="rw-note-top">
                <span className="rw-eyebrow">{selected.kind === 'quantum-game' ? 'ON THE QUANTUM ISLAND' : `INSIDE THE ${selected.short.toUpperCase()} STUDIO`}</span>
                <button
                  className="rw-icon-button"
                  aria-label="Close studio details"
                  onClick={() => setSelectedId(null)}
                >
                  <Icon name="close" size={16} />
                </button>
              </div>
              <div className="rw-person">
                {selected.kind === 'quantum-game' ? <span className="rw-avatar" style={{ '--studio-color': selected.color }}><Icon name="quantum-island" size={28} /></span> : <Avatar studio={selected} />}
                <div>
                  <h2>{selected.name}</h2>
                  <span>
                    {selected.status}
                    {world.mode === 'fixture' && selected.agent ? ' · Sample agent' : ''}
                  </span>
                </div>
              </div>
              <h3>{selected.title}</h3>
              <p>{selected.description}</p>
              {selected.agent?.has_question ? <blockquote>{selected.agent.question}</blockquote> : null}
              <button className="rw-primary" onClick={(event) => enter(selected, event)}>
                <Icon name={selected.kind === 'quantum-game' ? 'play' : 'coffee'} size={18} />
                {selected.kind === 'quantum-game' ? 'Enter quantum island' : `Meet ${selected.name.split(' ')[0]}`}
                <Icon name="arrow" size={17} />
              </button>
              <span className="rw-small-note">{selected.kind === 'quantum-game' ? 'Playable one-qubit lab · Local simulation' : 'Enter a 3D studio · Demo conversation'}</span>
              <MotionHarbourEntry studioId={selected.id} />
            </>
          ) : (
            <>
              <span className="rw-note-index">FIELD NOTE / 001</span>
              <div className="rw-note-illustration">
                <Icon name="coffee" size={35} />
                <span className="rw-note-line" />
                <Icon name="paper" size={30} />
              </div>
              <h2>
                A seat at the
                <br />
                research table.
              </h2>
              <p>Drop into a studio, read a paper together, or turn a “what if” into the next experiment.</p>
              <button
                className="rw-text-link"
                onClick={(event) =>
                  enter(studios.find((studio) => studio.id === 'physics') || studios[0], event)
                }
              >
                Take a seat <Icon name="arrow" size={17} />
              </button>
              <span className="rw-note-footer">Human curiosity, meet machine possibility.</span>
            </>
          )}
        </aside>
        <div className="rw-world-bottom">
          <span className="rw-drag-hint">
            <Icon name="compass" size={19} />
            Drag to explore <span>·</span> Scroll to zoom <span>·</span> Every direction
          </span>
          <div className="rw-controls" aria-label="Globe controls">
            <button aria-label="Rotate globe left" onClick={() => issue({ type: 'rotate', direction: -1 })}>
              ←
            </button>
            <button
              aria-label="Rotate globe up"
              onClick={() => issue({ type: 'rotate', direction: 1, axis: 'vertical' })}
            >
              ↑
            </button>
            <span />
            <button aria-label="Zoom in" onClick={() => issue({ type: 'zoom', direction: 1 })}>
              <Icon name="plus" size={17} />
            </button>
            <button aria-label="Zoom out" onClick={() => issue({ type: 'zoom', direction: -1 })}>
              <Icon name="minus" size={17} />
            </button>
            <span />
            <button
              aria-label={paused ? 'Resume motion' : 'Pause motion'}
              aria-pressed={paused}
              onClick={() => setPaused(!paused)}
            >
              <Icon name={paused ? 'play' : 'pause'} size={16} />
            </button>
            <button aria-label="Reset view" onClick={reset}>
              <Icon name="reset" size={17} />
            </button>
          </div>
        </div>
      </section>
      <section className="rw-studios" id="research-studios" aria-labelledby="studios-title">
        <div className="rw-studios-heading">
          <div>
            <p className="rw-eyebrow">FOLLOW YOUR CURIOSITY</p>
            <h2 id="studios-title">A different world in every discipline.</h2>
          </div>
          <label className="rw-search">
            <Icon name="search" size={17} />
            <input
              type="search"
              aria-label="Search research studios"
              placeholder="Find your field"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </label>
        </div>
        <div className="rw-studio-grid">
          {visibleStudios.map((studio) => (
            <button
              key={studio.id}
              className={`rw-studio-card ${selectedId === studio.id ? 'selected' : ''}`}
              ref={(element) => {
                if (element) studioButtons.current.set(studio.id, element)
                else studioButtons.current.delete(studio.id)
              }}
              style={{ '--studio-color': studio.color }}
              aria-pressed={selectedId === studio.id}
              onClick={() => {
                select(studio)
                document
                  .querySelector('.rw-universe')
                  ?.scrollIntoView({ behavior: reducedMotion ? 'instant' : 'smooth' })
              }}
            >
              <span className="rw-studio-icon">
                <Icon name={studio.id} size={25} />
              </span>
              <span>
                <b>{studio.label}</b>
                <small>{studio.activity}</small>
              </span>
              <Icon name="arrow" size={16} />
            </button>
          ))}
        </div>
        {visibleStudios.length === 0 ? (
          <p className="rw-empty">No studios match “{search}”. Try another field or researcher.</p>
        ) : null}
      </section>
      <section className="rw-researchers" aria-labelledby="researchers-title">
        <div>
          <p className="rw-eyebrow">FROM THE RESEARCH SNAPSHOT</p>
          <h2 id="researchers-title">Meet the researchers</h2>
          <p>Explore each agent’s current question and recorded status.</p>
        </div>
        <div className="rw-researcher-list">
          {world.agents.map((agent) => (
            <button
              key={agent.id}
              onClick={() => {
                const studio = studios.find((s) => s.id === agent.regionId)
                if (studio) {
                  select(studio, agent)
                  document
                    .querySelector('.rw-universe')
                    ?.scrollIntoView({ behavior: reducedMotion ? 'instant' : 'smooth' })
                }
              }}
            >
              <span className="rw-initials">
                {agent.displayName
                  .split(' ')
                  .map((word) => word[0])
                  .join('')}
              </span>
              <span>
                <b>{agent.displayName}</b>
                <small>{agent.title}</small>
              </span>
              <span className="rw-agent-status">{agent.statusLabel}</span>
            </button>
          ))}
        </div>
      </section>
      <footer className="rw-footer">
        <span>
          STATEVECTORAI <span> / </span> A little world. An open question.
        </span>
        <div role="status">
          {query.isError ? (
            <>
              <span>Snapshot unavailable · showing sample data.</span>
              <button onClick={() => query.refetch()}>Retry snapshot</button>
            </>
          ) : query.isFetching ? (
            'Refreshing the research snapshot…'
          ) : world.mode === 'fixture' ? (
            `${world.agents.length} sample agents · Animated scenes are illustrative · No live research connected`
          ) : (
            'Snapshot activity · Animations are illustrative'
          )}
          {reducedMotion ? ' · Reduced motion' : ''}
        </div>
      </footer>
      {meeting ? (
        <Suspense
          fallback={
            <div role="status" className="rw-loading">
              Opening the studio…
            </div>
          }
        >
          {meeting.kind === 'quantum-game' ? <QuantumIslandGame
            visit={sessions[QUANTUM_ISLAND.id]}
            onVisitChange={(next) => setSessions((current) => ({ ...current, [QUANTUM_ISLAND.id]: next }))}
            onClose={leave}
            reducedMotion={reducedMotion}
          /> : <ResearchSession
            key={meeting.agent?.id || meeting.id}
            studio={meeting}
            onClose={leave}
            paused={paused || reducedMotion}
            reducedMotion={reducedMotion}
            session={sessions[meeting.agent?.id || meeting.id]}
            onSessionChange={(next) =>
              setSessions((current) => ({ ...current, [meeting.agent?.id || meeting.id]: next }))
            }
          />}
        </Suspense>
      ) : null}
    </main>
  )
}
export function Avatar({ studio }) {
  return (
    <span className="rw-avatar" style={{ '--studio-color': studio.color }} aria-hidden="true">
      <span className="rw-avatar-head" />
      <span className="rw-avatar-body" />
    </span>
  )
}
class SceneBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { failed: false }
  }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  render() {
    return this.state.failed ? (
      <div className="rw-loading" role="status">
        The 3D view is unavailable. Explore every studio below.
      </div>
    ) : (
      this.props.children
    )
  }
}
