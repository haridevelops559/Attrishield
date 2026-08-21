import { useMutation, useQueryClient } from '@tanstack/react-query'
import { createBatch } from '../services/api'

export function useCreateBatch() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: createBatch,

    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ['batches'],
      })
    },
  })
}