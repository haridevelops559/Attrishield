import { useMutation } from '@tanstack/react-query'
import { predictIndividualEmployee } from '../services/api'

export function usePredictEmployee() {
  return useMutation({
    mutationFn: predictIndividualEmployee,
  })
}