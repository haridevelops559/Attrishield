import { useQuery } from '@tanstack/react-query'
import { getBatches } from '../services/api'

export function useBatches(page = 1, limit = 10) {
  const skip = (page - 1) * limit

  return useQuery({
    queryKey: ['batches', { page, limit }],
    queryFn: () => getBatches(limit, skip),
    placeholderData: (previousData) => previousData,
  })
}