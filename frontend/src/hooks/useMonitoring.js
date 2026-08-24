import { useQuery } from '@tanstack/react-query'
import { getMonitoringMetrics } from '../services/api'

export function useMonitoring() {
  return useQuery({
    queryKey: ['monitoring', 'metrics'],
    queryFn: getMonitoringMetrics,
    staleTime: 30 * 1000,
  })
}