import { useQuery } from '@tanstack/react-query'
import { getBatches } from '../services/api'

export function useBatches(limit = 50, skip = 0) {
  return useQuery({
    queryKey: ['batches', { limit, skip }],
    queryFn: () => getBatches(limit, skip),
  })
}