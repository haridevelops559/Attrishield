import {
  useQuery,
} from '@tanstack/react-query'

import {
  getFeatureDefinitions,
  getFeatureGroups,
  getFeatureLineage,
  getOnlineFeatures,
  getPointInTimeFeatures,
  getFeatureMaterializations,
} from '../services/api'

export function useFeatureDefinitions() {
  return useQuery({
    queryKey: ['feature-store', 'definitions'],
    queryFn: getFeatureDefinitions,
    staleTime: 60_000,
  })
}

export function useFeatureGroups() {
  return useQuery({
    queryKey: ['feature-store', 'groups'],
    queryFn: getFeatureGroups,
    staleTime: 60_000,
  })
}

export function useFeatureLineage(featureName = null) {
  return useQuery({
    queryKey: [
      'feature-store',
      'lineage',
      featureName,
    ],
    queryFn: () => getFeatureLineage(featureName),
    staleTime: 60_000,
  })
}

export function useOnlineFeatures(
  entityId,
  featureVersion = 'v3',
) {
  return useQuery({
    queryKey: [
      'feature-store',
      'online',
      entityId,
      featureVersion,
    ],
    queryFn: () =>
      getOnlineFeatures(
        entityId,
        featureVersion,
      ),
    enabled: Boolean(entityId),
  })
}

export function usePointInTimeFeatures(
  payload,
  featureVersion = 'v3',
  enabled = false,
) {
  return useQuery({
    queryKey: [
      'feature-store',
      'point-in-time',
      payload,
      featureVersion,
    ],
    queryFn: () =>
      getPointInTimeFeatures(
        payload,
        featureVersion,
      ),
    enabled,
  })
}

export function useFeatureMaterializations(
  limit = 20,
) {
  return useQuery({
    queryKey: [
      'feature-store',
      'materializations',
      limit,
    ],
    queryFn: () =>
      getFeatureMaterializations(limit),
    staleTime: 30_000,
  })
}