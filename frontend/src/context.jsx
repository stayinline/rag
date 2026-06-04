import React, { createContext, useContext, useState, useEffect } from 'react'
import { login as apiLogin } from './api'

const AuthContext = createContext(null)
const shouldLogAuth =
  import.meta.env.DEV || String(import.meta.env.VITE_API_LOGGING || '').toLowerCase() === 'true'

const logAuth = (level, message, details = {}) => {
  if (!shouldLogAuth) return
  const logger = console[level] || console.log
  logger(`[auth] ${message}`, details)
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const token = localStorage.getItem('token')
    const userData = localStorage.getItem('user')
    if (token && userData) {
      try {
        const parsedUser = JSON.parse(userData)
        setUser(parsedUser)
        logAuth('debug', 'restored session', {
          userId: parsedUser.id,
          orgId: parsedUser.org_id,
          username: parsedUser.username,
        })
      } catch {
        logAuth('error', 'failed to parse stored user; clearing session')
        localStorage.removeItem('token')
        localStorage.removeItem('user')
      }
    }
    setLoading(false)
  }, [])

  const login = async (username, password) => {
    logAuth('debug', 'login start', { username })
    try {
      const data = await apiLogin(username, password)
      localStorage.setItem('token', data.access_token)
      localStorage.setItem('user', JSON.stringify(data.user))
      setUser(data.user)
      logAuth('debug', 'login complete', {
        userId: data.user?.id,
        orgId: data.user?.org_id,
        username: data.user?.username,
      })
      return data
    } catch (error) {
      logAuth('error', 'login failed', {
        username,
        status: error.response?.status,
        message: error.message,
      })
      throw error
    }
  }

  const logout = () => {
    logAuth('debug', 'logout', { userId: user?.id, orgId: user?.org_id })
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
