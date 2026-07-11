import { useQuery } from '@tanstack/react-query'
import { api } from '../api.js'
import { LIVE_REFETCH_MS } from './queryClient.js'

// Thin typed-ish query hooks over the existing REST api. Live surfaces refetch
// on an interval; static ones do not.

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
  return useQuery({
    queryKey: ['jobs'],
    queryFn: api.jobs,
    refetchInterval: LIVE_REFETCH_MS,
  })
}

export function useDatasets() {
  return useQuery({ queryKey: ['datasets'], queryFn: api.datasets })
}
