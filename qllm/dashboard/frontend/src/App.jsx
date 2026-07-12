import { useEffect, useState } from 'react'
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
  const [theme, setTheme] = useState(initialTheme)
  const location = useLocation()
  const fetching = useIsFetching()
  useJobsStream()
  const streaming = useStreamActive()

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem(THEME_STORAGE_KEY, theme)
  }, [theme])

  const nextTheme = theme === 'dark' ? 'light' : 'dark'
  const crumb = navTitleForPath(location.pathname) || 'StateVector'

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="wordmark">
          <div className="wm-glyph">|ψ⟩</div>
          <div className="wm-text">StateVector<small>QUANTUM ML LAB</small></div>
        </div>
        {NAV_GROUPS.map((group) => (
          <div key={group.title}>
            <div className="navsec">{group.title}</div>
            <nav className="nav">
              {group.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) => (isActive ? 'active' : undefined)}
                >
                  <span className="ic">{item.icon}</span>
                  {item.label}
                  {item.badge ? <span className="pill">{item.badge}</span> : null}
                </NavLink>
              ))}
            </nav>
          </div>
        ))}
        <div className="foot">
          Signed in · <b>researcher</b>
          <br />Public visitors see Atlas only
        </div>
      </aside>

      <div className="main">
        <div className="topbar">
          <div className="crumb">{crumb}</div>
          <div className="right">
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
        <div className="content"><Outlet /></div>
      </div>
    </div>
  )
}
