import { useEffect } from 'react'
import { useSyncExternalStore } from 'react'
import { useQueryClient } from '@tanstack/react-query'

// Live job updates over Server-Sent Events (GET /api/stream/jobs), replacing the
// poll interval when the stream is connected. SQLite stays authoritative on the
// backend; each event carries a bounded snapshot (not a delta), so we can push it
// straight into the query cache. When SSE is unavailable or drops, the live hooks
// fall back to interval polling — see useStreamActive() in hooks.js.

// --- tiny external store: is the stream currently connected? -----------------
let active = false
const listeners = new Set()

function setActive(next) {
  if (next === active) return
  active = next
  for (const l of listeners) l()
}

function subscribe(listener) {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function useStreamActive() {
  return useSyncExternalStore(subscribe, () => active, () => false)
}

// Map a stream snapshot onto the query cache. The jobs stream carries the same
// bounded job rows the /jobs list returns; a snapshot may also carry live_runs.
// Accept the shapes the backend may use without guessing a single one.
export function applyJobsSnapshot(queryClient, snapshot) {
  if (!snapshot || typeof snapshot !== 'object') return
  const jobs = Array.isArray(snapshot) ? snapshot : snapshot.jobs ?? snapshot.data ?? snapshot.lab_jobs
  if (Array.isArray(jobs)) queryClient.setQueryData(['jobs'], jobs)
  // A live_runs projection, if present, refreshes the overview counts cheaply.
  if (snapshot.overview && typeof snapshot.overview === 'object') {
    queryClient.setQueryData(['overview'], snapshot.overview)
  }
}

// Open the jobs stream for the lifetime of the app shell. Idempotent per mount.
export function useJobsStream() {
  const queryClient = useQueryClient()
  useEffect(() => {
    if (typeof EventSource === 'undefined') return undefined
    let es
    try {
      es = new EventSource('/api/stream/jobs')
    } catch (_) {
      return undefined
    }
    const onSnapshot = (event) => {
      try {
        applyJobsSnapshot(queryClient, JSON.parse(event.data))
      } catch (_) {
        // ignore heartbeats / non-JSON keepalive comments
      }
    }
    es.onopen = () => setActive(true)
    es.onmessage = onSnapshot
    // Named events are also handled if the backend uses them.
    es.addEventListener('snapshot', onSnapshot)
    es.addEventListener('jobs', onSnapshot)
    es.onerror = () => {
      // The browser auto-reconnects; until it does, fall back to polling.
      setActive(false)
    }
    return () => {
      setActive(false)
      es.close()
    }
  }, [queryClient])
}
