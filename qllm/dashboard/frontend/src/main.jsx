import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClientProvider } from '@tanstack/react-query'
import { queryClient } from './lib/queryClient.js'
import App from './App.jsx'
import { LEGACY_REDIRECTS, PORTAL_BASE, isPortalPath, portalPath, portalRelativePath } from './appShell.js'
import ResearchWorld from './surfaces/ResearchWorld.jsx'
import Overview from './surfaces/Overview.jsx'
import Workboard from './surfaces/Workboard.jsx'
import DecisionInbox from './surfaces/DecisionInbox.jsx'
import DecisionDetail from './surfaces/DecisionDetail.jsx'
import ResearchRecord from './surfaces/ResearchRecord.jsx'
import Discover from './surfaces/Discover.jsx'
import Library from './surfaces/Library.jsx'
import Atlas from './surfaces/Atlas.jsx'
import Designer from './surfaces/Designer.jsx'
import Bench from './surfaces/Bench.jsx'
import Runs from './surfaces/Runs.jsx'
import RunDetail from './surfaces/RunDetail.jsx'
import Scaling from './surfaces/Scaling.jsx'
import Studies from './surfaces/Studies.jsx'
import Verdicts from './surfaces/Verdicts.jsx'
import Datasets from './surfaces/Datasets.jsx'
import System from './surfaces/System.jsx'
import NotFound from './surfaces/NotFound.jsx'
import './styles.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <Root />
    </QueryClientProvider>
  </React.StrictMode>,
)

function Root() {
  const pathname = globalThis.location?.pathname || '/'
  if (pathname === '/') return <ResearchWorld />
  if (!isPortalPath(pathname)) return <LegacyPortalRedirect />
  return (
    <BrowserRouter basename={PORTAL_BASE}>
      <Routes>
          <Route path="/" element={<App />}>
            <Route index element={<Workboard />} />
            <Route path="decisions" element={<DecisionInbox />} />
            <Route path="decisions/:id" element={<DecisionDetail />} />
            <Route path="research/:workId" element={<ResearchRecord />} />
            <Route path="lab" element={<Overview />} />
            <Route path="discover" element={<Discover />} />
            <Route path="library" element={<Library />} />
            <Route path="atlas" element={<Atlas />} />
            <Route path="designer" element={<Designer />} />
            <Route path="bench" element={<Bench />} />
            <Route path="runs" element={<Runs />} />
            <Route path="runs/scaling/:groupId" element={<Scaling />} />
            <Route path="runs/:id" element={<RunDetail />} />
            <Route path="studies" element={<Studies />} />
            <Route path="studies/:id" element={<Studies />} />
            <Route path="verdicts" element={<Verdicts />} />
            <Route path="verdicts/:id" element={<Verdicts />} />
            <Route path="datasets" element={<Datasets />} />
            <Route path="system" element={<System />} />
            {/* Legacy routes redirect to their new surface (no dead links). */}
            {Object.entries(LEGACY_REDIRECTS).map(([from, to]) => (
              <Route key={from} path={from.slice(1)} element={<Navigate to={portalRelativePath(to)} replace />} />
            ))}
            <Route path="*" element={<NotFound />} />
          </Route>
      </Routes>
    </BrowserRouter>
  )
}

function LegacyPortalRedirect() {
  const target = `${portalPath(globalThis.location?.pathname || '/')}${globalThis.location?.search || ''}${globalThis.location?.hash || ''}`
  globalThis.location?.replace?.(target)
  return <main className="research-world research-world-redirect"><p>Opening the portal… <a href={target}>Continue</a></p></main>
}
