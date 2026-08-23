import { useQuery } from '@tanstack/react-query'
import { getBatchPredictions } from '../services/api'

export function useBatchPredictions(batchId) {
  return useQuery({
    queryKey: ['batch-predictions', batchId],
    queryFn: () => getBatchPredictions(batchId),
    enabled: Boolean(batchId),
  })
}