import {
  ModelInfoSchema,
  PredictionResponseSchema,
} from '../schemas/apiSchemas'

const API_BASE_URL = '/api/v1'

export async function apiRequest(path, options = {}) {
  const token = getAccessToken()

  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  }

  if (token) {
    headers.Authorization = `Bearer ${token}`
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  })

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`

    try {
      const body = await response.json()

      if (body.detail) {
        message = body.detail
      }
    } catch {
      // Keep the default HTTP error message.
    }

    throw new Error(message)
  }

  return response.json()
}

function getAccessToken() {
  const stored = sessionStorage.getItem('attrishield_auth')

  if (!stored) {
    return null
  }

  try {
    const auth = JSON.parse(stored)
    return auth.accessToken ?? null
  } catch {
    return null
  }
}

export function loginRequest(email, password) {
  return apiRequest('/auth/login/json', {
    method: 'POST',
    body: JSON.stringify({
      email,
      password,
    }),
  })
}

export async function getModelInfo() {
  const response = await apiRequest('/model/info')

  const result = ModelInfoSchema.safeParse(response)

  if (!result.success) {
    console.error(
      'Invalid model info API response:',
      result.error.issues,
    )

    throw new Error(
      'The model service returned an invalid response.',
    )
  }

  return result.data
}

export async function predictIndividualEmployee(employeeData) {
  const response = await apiRequest('/inference/predict', {
    method: 'POST',
    body: JSON.stringify(employeeData),
  })

  const result = PredictionResponseSchema.safeParse(response)

  if (!result.success) {
    console.error(
      'Invalid prediction API response:',
      result.error.issues,
    )

    throw new Error(
      'The prediction service returned an invalid response.',
    )
  }

  return result.data
}

export async function getPrediction(predictionId) {
  return apiRequest(
    `/inference/predictions/${predictionId}`,
  )
}

export async function getBatches(limit = 50, skip = 0) {
  return apiRequest(`/batches?limit=${limit}&skip=${skip}`)
}

export async function getBatchStatus(batchId) {
  return apiRequest(`/batches/${batchId}`)
}

export async function createBatch(file) {
  const token = getAccessToken()

  const formData = new FormData()
  formData.append('file', file)

  const headers = {}

  if (token) {
    headers.Authorization = `Bearer ${token}`
  }

  const response = await fetch(`${API_BASE_URL}/batches`, {
    method: 'POST',
    headers,
    body: formData,
  })

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`

    try {
      const body = await response.json()

      if (body.detail) {
        message = body.detail
      }
    } catch {
      // Keep default error message.
    }

    throw new Error(message)
  }

  return response.json()
}

export async function getBatchPredictions(batchId) {
  return apiRequest(
    `/batches/${batchId}/predictions`,
  )
}
export async function runAnalyticsQuery(payload) {
  return apiRequest('/analytics/query', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function getAnalyticsCharts(batchId = null) {
  const query = batchId
    ? `?batch_id=${encodeURIComponent(batchId)}`
    : ''

  return apiRequest(`/analytics/charts${query}`)
}

export async function getMonitoringMetrics() {
  return apiRequest('/monitoring/metrics')
}

export async function generateAIInsights(payload) {
  return apiRequest('/ollama/insights', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

