import { useEffect, useRef, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useIsFetching } from '@tanstack/react-query'
import {
  NAV_GROUPS,
  THEME_STORAGE_KEY,
  resolveInitialTheme,
  navTitleForPath,
} from './appShell.js'
import { useJobsStream, useStreamActive } from './lib/stream.js'

function initialTheme() {
  const saved = globalThis.localStorage?.getItem?.(THEME_STORAGE_KEY)
  const prefersLight = globalThis.matchMedia?.('(prefers-color-scheme: light)').matches === true
  return resolveInitialTheme(saved, prefersLight)
}

export default function App() {
  const playground = import.meta.env.VITE_PLAYGROUND_MODE === '1'
  const [theme, setTheme] = useState(initialTheme)
  const [menuOpen, setMenuOpen] = useState(false)
  const mainRef = useRef(null)
  const location = useLocation()
  const fetching = useIsFetching()
  useJobsStream()
  const streaming = useStreamActive()

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem(THEME_STORAGE_KEY, theme)
  }, [theme])

  useEffect(() => {
    setMenuOpen(false)
    mainRef.current?.focus({ preventScroll: true })
  }, [location.pathname])

  const nextTheme = theme === 'dark' ? 'light' : 'dark'
  const crumb = navTitleForPath(location.pathname) || 'StateVector'

  return (
    <div className="app">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <aside className="sidebar">
        <div className="wordmark">
          <div className="wm-glyph">|ψ⟩</div>
          <div className="wm-text">StateVector<small>QUANTUM ML LAB</small></div>
        </div>
        <Navigation />
        <div className="foot">
          {playground ? (
            <><b>UAT playground</b><br />Simulated data · no jobs execute</>
          ) : (
            <>Signed in · <b>researcher</b><br />Public visitors see Atlas only</>
          )}
        </div>
      </aside>

      <div className="main">
        <div className="topbar">
          <button
            className="iconbtn mobile-menu-toggle"
            type="button"
            aria-expanded={menuOpen}
            aria-controls="mobile-navigation"
            onClick={() => setMenuOpen((open) => !open)}
          >
            Menu
          </button>
          <div className="crumb">{crumb}</div>
          <div className="right">
            {playground ? <span className="chip">PLAYGROUND</span> : null}
            <span className="chip" title={streaming ? 'Live SSE stream connected' : 'Polling for updates'}>
              <span className={`dot ${fetching || streaming ? 'run' : 'idle'}`} />
              {fetching ? 'syncing' : streaming ? 'streaming' : 'live'}
            </span>
            <button
              className="iconbtn"
              type="button"
              onClick={() => setTheme(nextTheme)}
              aria-label={`Switch to ${nextTheme} theme`}
            >
              ◐ {nextTheme}
            </button>
            <div className="avatar">AV</div>
          </div>
        </div>
        {menuOpen ? <nav id="mobile-navigation" className="mobile-navigation" aria-label="Primary navigation"><Navigation onNavigate={() => setMenuOpen(false)} /></nav> : null}
        <main id="main-content" className="content" tabIndex="-1" ref={mainRef}><Outlet /></main>
      </div>
    </div>
  )
}

function Navigation({ onNavigate }) {
  return NAV_GROUPS.map((group) => (
    <div key={group.title}>
      <div className="navsec">{group.title}</div>
      <nav className="nav" aria-label={group.title}>
        {group.items.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            onClick={onNavigate}
            className={({ isActive }) => (isActive ? 'active' : undefined)}
          >
            <span className="ic" aria-hidden="true">{item.icon}</span>
            {item.label}
            {item.badge ? <span className="pill">{item.badge}</span> : null}
          </NavLink>
        ))}
      </nav>
    </div>
  ))
}
