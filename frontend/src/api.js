import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
})

const shouldLogApi =
  import.meta.env.DEV || String(import.meta.env.VITE_API_LOGGING || '').toLowerCase() === 'true'

const redact = (value) => {
  if (!value || typeof value !== 'object') return value
  const sensitive = ['password', 'token', 'access_token', 'authorization', 'secret', 'api_key']
  const copy = Array.isArray(value) ? [...value] : { ...value }
  Object.keys(copy).forEach((key) => {
    if (sensitive.some((part) => key.toLowerCase().includes(part))) {
      copy[key] = '<redacted>'
    }
  })
  return copy
}

const logApi = (level, message, details = {}) => {
  if (!shouldLogApi) return
  const logger = console[level] || console.log
  logger(`[api] ${message}`, redact(details))
}

// Add auth interceptor
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  const requestId =
    config.headers['X-Request-ID'] ||
    (globalThis.crypto?.randomUUID ? globalThis.crypto.randomUUID() : `${Date.now()}-${Math.random()}`)
  config.headers['X-Request-ID'] = requestId
  config.metadata = { startedAt: performance.now(), requestId }
  logApi('debug', 'request start', {
    requestId,
    method: config.method?.toUpperCase(),
    url: `${config.baseURL || ''}${config.url || ''}`,
    params: config.params,
    data: config.data instanceof FormData ? '<form-data>' : config.data,
  })
  return config
})

api.interceptors.response.use(
  (res) => {
    const startedAt = res.config.metadata?.startedAt
    logApi('debug', 'response complete', {
      requestId: res.headers?.['x-request-id'] || res.config.metadata?.requestId,
      method: res.config.method?.toUpperCase(),
      url: `${res.config.baseURL || ''}${res.config.url || ''}`,
      status: res.status,
      durationMs: startedAt ? Math.round(performance.now() - startedAt) : undefined,
    })
    return res
  },
  (err) => {
    const config = err.config || {}
    const startedAt = config.metadata?.startedAt
    logApi('error', 'response failed', {
      requestId: err.response?.headers?.['x-request-id'] || config.metadata?.requestId,
      method: config.method?.toUpperCase(),
      url: `${config.baseURL || ''}${config.url || ''}`,
      status: err.response?.status,
      durationMs: startedAt ? Math.round(performance.now() - startedAt) : undefined,
      response: err.response?.data,
      message: err.message,
    })
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

// Auth
export const login = (username, password) =>
  api.post('/auth/login', { username, password }).then((r) => r.data)

// Knowledge Bases
export const listKbs = () => api.get('/kbs').then((r) => r.data)
export const createKb = (data) => api.post('/kbs', data).then((r) => r.data)
export const getKb = (id) => api.get(`/kbs/${id}`).then((r) => r.data)
export const updateKb = (id, data) => api.patch(`/kbs/${id}`, data).then((r) => r.data)
export const deleteKb = (id) => api.delete(`/kbs/${id}`).then((r) => r.data)

// Documents
export const listDocuments = (kbId) => api.get(`/kbs/${kbId}/documents`).then((r) => r.data)
export const getDocument = (id) => api.get(`/documents/${id}`).then((r) => r.data)
export const uploadDocument = (kbId, formData) =>
  api.post(`/kbs/${kbId}/documents`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then((r) => r.data)
export const deleteDocument = (id) => api.delete(`/documents/${id}`).then((r) => r.data)

// Ingestion Jobs
export const getIngestionJob = (jobId) => api.get(`/ingestion-jobs/${jobId}`).then((r) => r.data)

// Chat (non-streaming)
export const chat = (query, kbIds = [], conversationId = null) =>
  api.post('/chat', { query, kb_ids: kbIds, conversation_id: conversationId, stream: false })
    .then((r) => r.data)

// Search
export const search = (query, kbIds = [], limit = 10) =>
  api.post('/search', { query, kb_ids: kbIds, limit }).then((r) => r.data)

// Papers
export const uploadPaper = (formData) =>
  api.post('/papers/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then((r) => r.data)
export const importPaperByDoi = (doi, kbId) =>
  api.post('/papers/import-doi', { doi, kb_id: kbId }).then((r) => r.data)
export const importPaperByPmid = (pmid, kbId) =>
  api.post('/papers/import-pmid', { pmid, kb_id: kbId }).then((r) => r.data)
export const getPaper = (id) => api.get(`/papers/${id}`).then((r) => r.data)
export const getPaperEvidence = (id) => api.get(`/papers/${id}/evidence`).then((r) => r.data)
export const getPaperReferences = (id) => api.get(`/papers/${id}/references`).then((r) => r.data)
export const getSimilarPapers = (id) => api.get(`/papers/${id}/similar`).then((r) => r.data)

// Feedback
export const submitFeedback = (messageId, data) =>
  api.post(`/answers/${messageId}/feedback`, data).then((r) => r.data)

// Evaluation Sets
export const listEvalSets = () => api.get('/evaluation-sets').then((r) => r.data)
export const createEvalSet = (data) => api.post('/evaluation-sets', data).then((r) => r.data)
export const getEvalSet = (id) => api.get(`/evaluation-sets/${id}`).then((r) => r.data)
export const listEvalQuestions = (setId) =>
  api.get(`/evaluation-sets/${setId}/questions`).then((r) => r.data)

// Evaluation Runs
export const runEvaluation = (evalSetId, config = {}) =>
  api.post('/evaluations/run', { eval_set_id: evalSetId, config }).then((r) => r.data)
export const getEvaluationRun = (runId) => api.get(`/evaluations/${runId}`).then((r) => r.data)

// Analytics
export const getZeroResultQueries = () => api.get('/analytics/zero-result-queries').then((r) => r.data)
export const getLowRatedAnswers = () => api.get('/analytics/low-rated-answers').then((r) => r.data)
export const getAnalyticsSummary = () => api.get('/analytics/summary').then((r) => r.data)

// Audit Logs
export const listAuditLogs = (params = {}) =>
  api.get('/audit-logs', { params }).then((r) => r.data)

export default api
