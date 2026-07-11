import { QueryClient } from '@tanstack/react-query'

// Single app-wide client. Server state is cached and refetched on an interval
// instead of the old per-page setInterval full refetches. A live SSE/WebSocket
// stream replaces polling in a later phase (docs/UI_REDESIGN_PLAN.md §8).
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 2_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

// Poll intervals (ms) for live surfaces until the event stream lands.
export const LIVE_REFETCH_MS = 4_000
