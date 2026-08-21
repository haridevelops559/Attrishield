import { createContext, useContext, useMemo, useState } from 'react'

const AuthContext = createContext(null)

const STORAGE_KEY = 'attrishield_auth'

function readStoredAuth() {
  const stored = sessionStorage.getItem(STORAGE_KEY)

  if (!stored) {
    return null
  }

  try {
    return JSON.parse(stored)
  } catch {
    sessionStorage.removeItem(STORAGE_KEY)
    return null
  }
}

export function AuthProvider({ children }) {
  const [auth, setAuth] = useState(readStoredAuth)

  function login(authResponse) {
    const nextAuth = {
      accessToken: authResponse.access_token,
      tokenType: authResponse.token_type,
      userRole: authResponse.user_role,
      userEmail: authResponse.user_email,
    }

    sessionStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(nextAuth),
    )

    setAuth(nextAuth)
  }

  function logout() {
    sessionStorage.removeItem(STORAGE_KEY)
    setAuth(null)
  }

  const value = useMemo(
    () => ({
      auth,
      isAuthenticated: Boolean(auth?.accessToken),
      login,
      logout,
    }),
    [auth],
  )

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)

  if (!context) {
    throw new Error(
      'useAuth must be used inside AuthProvider',
    )
  }

  return context
}