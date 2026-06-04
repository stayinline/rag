import React from 'react'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { AuthProvider, useAuth } from './context'
import Login from './pages/Login'
import Layout from './components/Layout'
import Chat from './pages/Chat'
import KnowledgeBases from './pages/KnowledgeBases'
import PaperHub from './pages/PaperHub'
import Analytics from './pages/Analytics'
import Evaluation from './pages/Evaluation'

function PrivateRoute({ children }) {
  const { user, loading } = useAuth()
  if (loading) return <div className="flex items-center justify-center h-screen">加载中...</div>
  return user ? children : <Navigate to="/login" />
}

function AppContent() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/*" element={
        <PrivateRoute>
          <Layout />
        </PrivateRoute>
      }>
        <Route index element={<Navigate to="/chat" replace />} />
        <Route path="chat" element={<Chat />} />
        <Route path="kbs" element={<KnowledgeBases />} />
        <Route path="papers" element={<PaperHub />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="evaluation" element={<Evaluation />} />
      </Route>
    </Routes>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  )
}
