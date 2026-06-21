import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import App from './App.jsx'
import LabOverview from './pages/LabOverview.jsx'
import Explore from './pages/Explore.jsx'
import ResearchResults from './pages/ResearchResults.jsx'
import Suites from './pages/Suites.jsx'
import Suite from './pages/Suite.jsx'
import Run from './pages/Run.jsx'
import Launch from './pages/Launch.jsx'
import Datasets from './pages/Datasets.jsx'
import Docs from './pages/Docs.jsx'
import Live from './pages/Live.jsx'
import Jobs from './pages/Jobs.jsx'
import RunWorkspace from './pages/RunWorkspace.jsx'
import GPU from './pages/GPU.jsx'
import Comparison from './pages/Comparison.jsx'
import Models from './pages/Models.jsx'
import Studies from './pages/Studies.jsx'
import Study from './pages/Study.jsx'
import ScalingTest from './pages/ScalingTest.jsx'
import ScalingTests from './pages/ScalingTests.jsx'
import './styles.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />}>
          <Route index element={<LabOverview />} />
          <Route path="overview" element={<LabOverview />} />
          <Route path="explore" element={<Explore />} />
          <Route path="explore/dataset/:dataset" element={<ResearchResults mode="dataset" />} />
          <Route path="explore/task/:task" element={<ResearchResults mode="task" />} />
          <Route path="launch" element={<Launch />} />
          <Route path="experiments" element={<Jobs />} />
          <Route path="jobs" element={<Jobs />} />
          <Route path="jobs/:id" element={<RunWorkspace />} />
          <Route path="comparisons/:id" element={<Comparison />} />
          <Route path="studies" element={<Studies />} />
          <Route path="studies/:id" element={<Study />} />
          <Route path="scaling" element={<ScalingTests />} />
          <Route path="scaling/:groupId" element={<ScalingTest />} />
          <Route path="models" element={<Models />} />
          <Route path="datasets" element={<Datasets />} />
          <Route path="results" element={<Suites />} />
          <Route path="suite/:name" element={<Suite />} />
          <Route path="run/:id" element={<Run />} />
          <Route path="live" element={<Live />} />
          <Route path="docs" element={<Docs />} />
          <Route path="gpu" element={<GPU />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
)
