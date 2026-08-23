import { useQuery } from '@tanstack/react-query'
import { getPrediction } from '../services/api'

export function usePrediction(predictionId) {
  return useQuery({
    queryKey: ['prediction', predictionId],
    queryFn: () => getPrediction(predictionId),
    enabled: Boolean(predictionId),
  })
}