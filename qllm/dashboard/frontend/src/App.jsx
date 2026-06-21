import { NavLink, Outlet } from 'react-router-dom'

export default function App() {
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">QLLM<span>.</span>Lab</div>
        <div className="tagline">quantum-classical LM testbed</div>
        <nav className="nav">
          <NavLink to="/" end>Lab Overview</NavLink>
          <NavLink to="/experiments">Experiments</NavLink>
          <NavLink to="/launch">New Experiment</NavLink>
          <NavLink to="/scaling">Scaling Tests</NavLink>
          <NavLink to="/models">Model Builder</NavLink>
          <NavLink to="/datasets">Datasets & Tasks</NavLink>
          <NavLink to="/results">Results</NavLink>
          <NavLink to="/gpu">System</NavLink>
        </nav>
      </aside>
      <main className="main"><Outlet /></main>
    </div>
  )
}
