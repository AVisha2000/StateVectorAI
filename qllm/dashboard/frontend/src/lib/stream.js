import { useEffect } from 'react'
import { useSyncExternalStore } from 'react'
import { useQueryClient } from '@tanstack/react-query'

// Live job updates over Server-Sent Events (GET /api/stream/jobs), replacing the
// poll interval when the stream is connected. SQLite stays authoritative on the
// backend. Each event carries a *lean* content-addressed snapshot — bounded
// lab_jobs/live_runs projections plus a `change_token` — NOT the full /jobs rows.
// So we never overwrite the jobs cache with it; a changed token just tells us to
// refetch the authoritative queries. When SSE is unavailable or drops, the live
// hooks fall back to interval polling (see useStreamActive() in hooks.js).

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

// The content-addressed token that changes whenever the projected content does.
export function readChangeToken(snapshot) {
  if (!snapshot || typeof snapshot !== 'object') return null
  return typeof snapshot.change_token === 'string' ? snapshot.change_token : null
}

// Authoritative queries to refetch when the stream reports a change.
export const STREAM_REFRESH_KEYS = Object.freeze([['jobs'], ['overview'], ['workspace']])

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
    let lastToken = null
    const onEvent = (event) => {
      let snapshot
      try {
        snapshot = JSON.parse(event.data)
      } catch (_) {
        return // heartbeats are SSE comments, never delivered as data events
      }
      const token = readChangeToken(snapshot)
      // Dedupe on the content-addressed token; only refetch on real changes.
      if (token && token === lastToken) return
      lastToken = token
      for (const key of STREAM_REFRESH_KEYS) {
        queryClient.invalidateQueries({ queryKey: key })
      }
    }
    es.onopen = () => setActive(true)
    // Backend names every event `jobs`; keep onmessage as a fallback for
    // deployments that emit the default (unnamed) event.
    es.addEventListener('jobs', onEvent)
    es.onmessage = onEvent
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
