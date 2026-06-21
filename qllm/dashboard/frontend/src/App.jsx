import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'

function initialTheme() {
  const saved = localStorage.getItem('qllm-theme')
  if (saved === 'light' || saved === 'dark') return saved
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

export default function App() {
  const [theme, setTheme] = useState(initialTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('qllm-theme', theme)
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
          <NavLink to="/" end>Overview</NavLink>
          <NavLink to="/explore">Explore</NavLink>
          <NavLink to="/experiments">Experiments</NavLink>
          <NavLink to="/models">Model Builder</NavLink>
          <NavLink to="/studies">Studies</NavLink>
          <NavLink to="/results">Results</NavLink>
          <NavLink to="/datasets">Datasets & Tasks</NavLink>
          <NavLink to="/gpu">System</NavLink>
          <NavLink to="/docs">Docs</NavLink>
        </nav>
      </aside>
      <main className="main"><Outlet /></main>
    </div>
  )
}
