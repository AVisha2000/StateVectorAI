import { useQuery } from '@tanstack/react-query'
import { api } from '../api.js'
import { LIVE_REFETCH_MS } from './queryClient.js'
import { useStreamActive } from './stream.js'

// Thin typed-ish query hooks over the existing REST api. Live surfaces refetch
// on an interval; static ones do not. The /api/stream/jobs SSE stream replaces
// the interval whenever it is connected (docs/UI_REDESIGN_PLAN §8).

export function useOverview() {
  return useQuery({
    queryKey: ['overview'],
    queryFn: api.overview,
    refetchInterval: LIVE_REFETCH_MS,
  })
}

export function useStatus() {
  return useQuery({
    queryKey: ['status'],
    queryFn: api.status,
    refetchInterval: LIVE_REFETCH_MS,
  })
}

export function useJobs() {
  // When the SSE stream is connected it pushes fresh snapshots, so the interval
  // poll is disabled; it re-arms automatically if the stream drops.
  const streamActive = useStreamActive()
  return useQuery({
    queryKey: ['jobs'],
    queryFn: api.jobs,
    refetchInterval: streamActive ? false : LIVE_REFETCH_MS,
  })
}

export function useDatasets() {
  return useQuery({ queryKey: ['datasets'], queryFn: api.datasets })
}

export function usePresets() {
  return useQuery({ queryKey: ['presets'], queryFn: api.presets })
}

export function useConfigChoices() {
  return useQuery({ queryKey: ['config-choices'], queryFn: api.configChoices })
}

// A single run. Live-refetches while the job is in flight; a finished/failed run
// is static, so the caller can pass live=false to stop polling.
export function useJob(id, { live = true } = {}) {
  return useQuery({
    queryKey: ['job', id],
    queryFn: () => api.job(id),
    enabled: id != null && id !== '',
    refetchInterval: live ? LIVE_REFETCH_MS : false,
  })
}

export function useComparison(id) {
  return useQuery({
    queryKey: ['comparison', id],
    queryFn: () => api.comparison(id),
    enabled: id != null && id !== '',
  })
}

// The workspace payload bundles the run, its metric curve, and its inline
// comparison — the primary source for the run-detail surface.
export function useWorkspace(id, { live = true } = {}) {
  return useQuery({
    queryKey: ['workspace', id],
    queryFn: () => api.workspace(id),
    enabled: id != null && id !== '',
    refetchInterval: live ? LIVE_REFETCH_MS : false,
  })
}

// model-tests carries summary.quantum_diagnostics (grad variance, Meyer–Wallach,
// expressibility) persisted at train time — today's source for run diagnostics.
export function useModelTests(id) {
  return useQuery({
    queryKey: ['model-tests', id],
    queryFn: () => api.modelTests(id),
    enabled: id != null && id !== '',
    retry: false,
  })
}

// The run's architecture graph (nodes/edges) for the model-structure view.
export function useModelGraph(id) {
  return useQuery({
    queryKey: ['model-graph', id],
    queryFn: () => api.jobGraph(id),
    enabled: id != null && id !== '',
    retry: false,
  })
}

// Quiet-404 options for the research contracts: they are shipped on main, but
// an older backend build may not serve them yet. Don't retry or spam a 404;
// let the caller render a calm degraded state instead of a hard error
// (see isNotYetBuilt).
const quiet404QueryOptions = {
  retry: false,
  refetchOnWindowFocus: false,
}

export function isNotYetBuilt(error) {
  return error?.status === 404
}

// /jobs/{id}/diagnostics — per-run quantum diagnostics (retrieval-only).
export function useDiagnostics(id) {
  return useQuery({
    queryKey: ['diagnostics', id],
    queryFn: () => api.diagnostics(id),
    enabled: id != null && id !== '',
    ...quiet404QueryOptions,
  })
}

// /verdicts — persistent, append-only verdict store.
export function useVerdicts() {
  return useQuery({
    queryKey: ['verdicts'],
    queryFn: api.verdicts,
    ...quiet404QueryOptions,
  })
}

export function useVerdict(id) {
  return useQuery({
    queryKey: ['verdict', id],
    queryFn: () => api.verdict(id),
    enabled: id != null && id !== '',
    ...quiet404QueryOptions,
  })
}

export function useScalingTests() {
  return useQuery({ queryKey: ['scaling-tests'], queryFn: api.scalingTests })
}

// Multi-seed studies: the rigor view where a claim is tested across seeds.
export function useStudies() {
  return useQuery({ queryKey: ['studies'], queryFn: api.studies })
}

export function useStudy(id) {
  return useQuery({
    queryKey: ['study', id],
    queryFn: () => api.study(id),
    enabled: id != null && id !== '',
  })
}

// /atlas/ontology — the canonical curated domain→component ontology (shipped).
// The Atlas surface falls back to its bundled seed if the endpoint is absent.
export function useAtlasOntology() {
  return useQuery({
    queryKey: ['atlas-ontology'],
    queryFn: api.atlasOntology,
    ...quiet404QueryOptions,
  })
}

// GET /designer/circuit — registry-backed capabilities: choices, defaults, and
// constraints (bounds, qrnn-only architecture rule, MPS bond-dim requirement).
// The Designer prefers these over its static fallbacks.
export function useDesignerCapabilities() {
  return useQuery({
    queryKey: ['designer-capabilities'],
    queryFn: api.designerCapabilities,
    staleTime: 5 * 60 * 1000, // registry choices change only with a backend deploy
    ...quiet404QueryOptions,
  })
}

// Research-service capabilities (D4 boundary: which providers are enabled, cost
// budget, human gates — all still closed pending the user's D4 decision).
export function useResearchCapabilities() {
  return useQuery({
    queryKey: ['research-capabilities'],
    queryFn: api.researchCapabilities,
    ...quiet404QueryOptions,
  })
}

export function useScalingTest(groupId) {
  return useQuery({
    queryKey: ['scaling-test', groupId],
    queryFn: () => api.scalingTest(groupId),
    enabled: groupId != null && groupId !== '',
  })
}
