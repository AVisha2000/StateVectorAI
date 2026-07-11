import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { NAV_ITEMS, THEME_STORAGE_KEY, resolveInitialTheme } from './appShell.js'

function initialTheme() {
  const saved = globalThis.localStorage?.getItem?.(THEME_STORAGE_KEY)
  const prefersLight = globalThis.matchMedia?.('(prefers-color-scheme: light)').matches === true
  return resolveInitialTheme(saved, prefersLight)
}

export default function App() {
  const [theme, setTheme] = useState(initialTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem(THEME_STORAGE_KEY, theme)
  }, [theme])

  const nextTheme = theme === 'dark' ? 'light' : 'dark'

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">QLLM<span>.</span>Cockpit</div>
        <div className="tagline">quantum advantage research console</div>
        <button
          className="theme-toggle"
          type="button"
          onClick={() => setTheme(nextTheme)}
          aria-label={`Switch to ${nextTheme} theme`}
        >
          <span>{theme === 'dark' ? 'dark' : 'light'}</span>
          <b>{nextTheme}</b>
        </button>
        <nav className="nav">
          {NAV_ITEMS.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end}>{item.label}</NavLink>
          ))}
        </nav>
      </aside>
      <main className="main"><Outlet /></main>
    </div>
  )
}
