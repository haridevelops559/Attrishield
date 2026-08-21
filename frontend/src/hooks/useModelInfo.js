import { useQuery } from '@tanstack/react-query'
import { getModelInfo } from '../services/api'

export function useModelInfo() {
  return useQuery({
    queryKey: ['model-info'],
    queryFn: getModelInfo,
  })
}