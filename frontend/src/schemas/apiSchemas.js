import { z } from 'zod'

export const EngineeredFeaturesSchema = z.object({
  IncomePerJobLevel: z.number(),
  PromotionStagnationRatio: z.number(),
  ManagerTenureRatio: z.number(),
  RoleTenureRatio: z.number(),
  OverTimeBinary: z.number(),
  CommuteOvertimeBurden: z.number(),
  EarlyCareerFlag: z.number(),
})

export const PredictionResponseSchema = z.object({
  prediction_id: z.string(),
  attrition_probability: z.number().min(0).max(1),
  attrition_prediction: z.number(),
  selected_threshold: z.number().min(0).max(1),
  risk_recommendation: z.string(),
  model_version: z.string(),
  feature_version: z.string(),
  latency_ms: z.number().nonnegative(),
  engineered_features: EngineeredFeaturesSchema,
})

export const ModelInfoSchema = z.object({
  // Keep this intentionally permissive until we inspect
  // the exact /model/info response contract.
}).passthrough()