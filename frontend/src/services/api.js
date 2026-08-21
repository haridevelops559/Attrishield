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
      // Keep default HTTP error message.
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

export function getModelInfo() {
  return apiRequest('/model/info')
}

export function predictIndividualEmployee(employeeData) {
  return apiRequest('/inference/predict', {
    method: 'POST',
    body: JSON.stringify(employeeData),
  })
}