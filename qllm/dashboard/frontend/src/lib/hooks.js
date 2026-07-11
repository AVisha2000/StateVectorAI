import { useQuery } from '@tanstack/react-query'
import { api } from '../api.js'
import { LIVE_REFETCH_MS } from './queryClient.js'
import { useStreamActive } from './stream.js'

// Thin typed-ish query hooks over the existing REST api. Live surfaces refetch
// on an interval; static ones do not. (The interval is replaced by the
// /api/stream/jobs SSE stream once the backend ships it — docs/UI_REDESIGN_PLAN §8.)

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

// A "proposed" endpoint the backend has not shipped yet returns 404. Don't retry
// or spam it, and let the caller render a calm "awaiting backend" state instead
// of a hard error (see isNotYetBuilt).
const proposedQueryOptions = {
  retry: false,
  refetchOnWindowFocus: false,
}

export function isNotYetBuilt(error) {
  return error?.status === 404
}

// Proposed: /jobs/{id}/diagnostics — per-run quantum diagnostics.
export function useDiagnostics(id) {
  return useQuery({
    queryKey: ['diagnostics', id],
    queryFn: () => api.diagnostics(id),
    enabled: id != null && id !== '',
    ...proposedQueryOptions,
  })
}

// Proposed: /verdicts — persistent verdict store.
export function useVerdicts() {
  return useQuery({
    queryKey: ['verdicts'],
    queryFn: api.verdicts,
    ...proposedQueryOptions,
  })
}

export function useVerdict(id) {
  return useQuery({
    queryKey: ['verdict', id],
    queryFn: () => api.verdict(id),
    enabled: id != null && id !== '',
    ...proposedQueryOptions,
  })
}

export function useScalingTests() {
  return useQuery({ queryKey: ['scaling-tests'], queryFn: api.scalingTests })
}

export function useScalingTest(groupId) {
  return useQuery({
    queryKey: ['scaling-test', groupId],
    queryFn: () => api.scalingTest(groupId),
    enabled: groupId != null && groupId !== '',
  })
}
