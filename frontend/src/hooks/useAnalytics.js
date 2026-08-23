import { useQuery } from '@tanstack/react-query'
import {
  getAnalyticsCharts,
  runAnalyticsQuery,
} from '../services/api'

export function useAnalyticsQuery(payload = {}) {
  return useQuery({
    queryKey: ['analytics', payload],
    queryFn: () => runAnalyticsQuery(payload),
  })
}

export function useAnalyticsCharts(batchId = null) {
  return useQuery({
    queryKey: ['analytics-charts', batchId],
    queryFn: () => getAnalyticsCharts(batchId),
  })
}