import React, { useState, useEffect } from 'react'
import {
  getAnalyticsSummary,
  getZeroResultQueries,
  getLowRatedAnswers,
  listAuditLogs,
  listRagTraces,
  getRagTrace,
} from '../api'

export default function Analytics() {
  const [activeTab, setActiveTab] = useState('summary')
  const [summary, setSummary] = useState(null)
  const [zeroQueries, setZeroQueries] = useState([])
  const [lowAnswers, setLowAnswers] = useState([])
  const [auditLogs, setAuditLogs] = useState([])
  const [ragTraces, setRagTraces] = useState([])
  const [selectedTrace, setSelectedTrace] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [auditFilter, setAuditFilter] = useState({ action: '', resource_type: '' })

  useEffect(() => {
    if (activeTab === 'summary') fetchSummary()
    if (activeTab === 'zero') fetchZeroQueries()
    if (activeTab === 'low') fetchLowAnswers()
    if (activeTab === 'traces') fetchRagTraces()
    if (activeTab === 'audit') fetchAuditLogs()
  }, [activeTab])

  const getErrorMessage = (err) => {
    const detail = err?.response?.data?.detail
    if (Array.isArray(detail)) return detail.map((item) => item.msg || item.type).filter(Boolean).join('；')
    if (typeof detail === 'string') return detail
    return err?.message || '请求失败'
  }

  const fetchSummary = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await getAnalyticsSummary()
      setSummary(data)
    } catch (err) {
      setSummary(null)
      setError(`加载总览失败：${getErrorMessage(err)}`)
    } finally {
      setLoading(false)
    }
  }

  const fetchZeroQueries = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await getZeroResultQueries()
      setZeroQueries(data.items || data)
    } catch (err) {
      setZeroQueries([])
      setError(`加载零结果查询失败：${getErrorMessage(err)}`)
    } finally {
      setLoading(false)
    }
  }

  const fetchLowAnswers = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await getLowRatedAnswers()
      setLowAnswers(data.items || data)
    } catch (err) {
      setLowAnswers([])
      setError(`加载低分回答失败：${getErrorMessage(err)}`)
    } finally {
      setLoading(false)
    }
  }

  const fetchAuditLogs = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await listAuditLogs(auditFilter)
      setAuditLogs(data.items || data)
    } catch (err) {
      setAuditLogs([])
      setError(`加载审计日志失败：${getErrorMessage(err)}`)
    } finally {
      setLoading(false)
    }
  }

  const fetchRagTraces = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await listRagTraces({ limit: 20 })
      const items = data.items || []
      setRagTraces(items)
      if (items.length > 0) {
        await fetchTraceDetail(items[0].trace_id, false)
      } else {
        setSelectedTrace(null)
      }
    } catch (err) {
      setRagTraces([])
      setSelectedTrace(null)
      setError(`加载链路追踪失败：${getErrorMessage(err)}`)
    } finally {
      setLoading(false)
    }
  }

  const fetchTraceDetail = async (traceId, showLoading = true) => {
    if (showLoading) setLoading(true)
    setError('')
    try {
      const data = await getRagTrace(traceId)
      setSelectedTrace(data)
    } catch (err) {
      setSelectedTrace(null)
      setError(`加载链路详情失败：${getErrorMessage(err)}`)
    } finally {
      if (showLoading) setLoading(false)
    }
  }

  const tabs = [
    { key: 'summary', label: '总览' },
    { key: 'traces', label: '链路追踪' },
    { key: 'zero', label: '零结果查询' },
    { key: 'low', label: '低分回答' },
    { key: 'audit', label: '审计日志' },
  ]

  return (
    <div className="p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-6">数据分析</h2>

      {/* Tabs */}
      <div className="flex gap-1 bg-gray-100 rounded-lg p-1 mb-6 w-fit">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
              activeTab === tab.key
                ? 'bg-white text-gray-800 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {loading && <div className="text-center py-8 text-gray-400">加载中...</div>}
      {!loading && error && (
        <div className="mb-4 rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Summary Tab */}
      {!loading && activeTab === 'summary' && summary && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            { label: '总查询数', value: summary.total_queries, icon: '🔍' },
            { label: '平均延迟', value: `${summary.avg_latency_ms?.toFixed(0)}ms`, icon: '⏱️' },
            { label: '平均评分', value: summary.avg_rating?.toFixed(2), icon: '⭐' },
            { label: '零结果率', value: `${((summary.zero_result_rate || 0) * 100).toFixed(1)}%`, icon: '❌' },
            { label: '低分率', value: `${((summary.low_rating_rate || 0) * 100).toFixed(1)}%`, icon: '📉' },
            { label: '平均检索数', value: summary.avg_retrieved_count?.toFixed(1), icon: '📊' },
            { label: '平均重排数', value: summary.avg_reranked_count?.toFixed(1), icon: '🎯' },
          ].map((stat) => (
            <div key={stat.label} className="bg-white border border-gray-200 rounded-xl p-4">
              <div className="flex items-center justify-between">
                <span className="text-2xl">{stat.icon}</span>
              </div>
              <div className="text-2xl font-bold text-gray-800 mt-3">{stat.value ?? '-'}</div>
              <div className="text-sm text-gray-500 mt-1">{stat.label}</div>
            </div>
          ))}
        </div>
      )}
      {!loading && activeTab === 'summary' && !summary && (
        <div className="text-center py-12 text-gray-400">暂无分析数据</div>
      )}

      {/* RAG Trace Tab */}
      {!loading && activeTab === 'traces' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
              <span className="font-medium text-gray-700">最近请求</span>
              <button onClick={fetchRagTraces} className="px-2 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700">
                刷新
              </button>
            </div>
            {ragTraces.length === 0 ? (
              <div className="text-center py-10 text-gray-400 text-sm">暂无链路数据</div>
            ) : (
              <div className="divide-y divide-gray-100 max-h-[640px] overflow-auto">
                {ragTraces.map((trace) => (
                  <button
                    key={trace.trace_id}
                    onClick={() => fetchTraceDetail(trace.trace_id)}
                    className={`w-full text-left px-4 py-3 hover:bg-gray-50 ${
                      selectedTrace?.trace_id === trace.trace_id ? 'bg-blue-50' : ''
                    }`}
                  >
                    <div className="text-sm font-medium text-gray-800 break-words">{trace.query}</div>
                    <div className="mt-1 flex flex-wrap gap-2 text-xs text-gray-500">
                      <span>{new Date(trace.created_at).toLocaleString()}</span>
                      <span>{trace.total_latency_ms}ms</span>
                      <span>{trace.steps?.length || 0} 步</span>
                    </div>
                    <div className="mt-1 text-xs text-gray-400 truncate">{trace.trace_id}</div>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="lg:col-span-2 bg-white border border-gray-200 rounded-xl p-4">
            {!selectedTrace ? (
              <div className="text-center py-12 text-gray-400">请选择一次请求</div>
            ) : (
              <div>
                <div className="border-b border-gray-100 pb-3 mb-4">
                  <div className="text-xs text-gray-400">Trace ID</div>
                  <div className="font-mono text-xs text-gray-700 break-all">{selectedTrace.trace_id}</div>
                  <div className="mt-3 text-sm text-gray-800">{selectedTrace.query}</div>
                  <div className="mt-2 flex flex-wrap gap-3 text-xs text-gray-500">
                    <span>总耗时：{selectedTrace.total_latency_ms}ms</span>
                    <span>模型：{selectedTrace.model || '-'}</span>
                    <span>答案长度：{selectedTrace.answer_length}</span>
                  </div>
                </div>

                <div className="space-y-3">
                  {(selectedTrace.steps || []).map((step, index) => (
                    <div key={`${step.step}-${index}`} className="border border-gray-200 rounded-lg p-3">
                      <div className="flex items-center justify-between gap-3">
                        <div className="font-medium text-gray-800">
                          {index + 1}. {step.step}
                        </div>
                        <div className="text-xs text-gray-500">{step.duration_ms?.toFixed?.(2) ?? step.duration_ms}ms</div>
                      </div>
                      <pre className="mt-2 bg-gray-50 rounded p-3 text-xs text-gray-700 overflow-auto max-h-72">
                        {JSON.stringify(step.details || {}, null, 2)}
                      </pre>
                    </div>
                  ))}
                </div>

                <div className="mt-4 border border-gray-200 rounded-lg p-3">
                  <div className="font-medium text-gray-800 mb-2">答案预览</div>
                  <div className="text-sm text-gray-700 whitespace-pre-wrap">{selectedTrace.answer_preview || '-'}</div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Zero Result Queries Tab */}
      {!loading && activeTab === 'zero' && (
        <div>
          <button onClick={fetchZeroQueries} className="mb-4 px-3 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">
            刷新
          </button>
          {zeroQueries.length === 0 ? (
            <div className="text-center py-12 text-gray-400">暂无零结果查询</div>
          ) : (
            <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="text-left px-4 py-2 font-medium text-gray-600">查询内容</th>
                    <th className="text-left px-4 py-2 font-medium text-gray-600">出现次数</th>
                    <th className="text-left px-4 py-2 font-medium text-gray-600">用户</th>
                  </tr>
                </thead>
                <tbody>
                  {zeroQueries.map((q, i) => (
                    <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="px-4 py-2">{q.query_text || q.query || '-'}</td>
                      <td className="px-4 py-2">
                        <span className="px-2 py-0.5 bg-red-100 text-red-700 rounded-full text-xs font-medium">
                          {q.cnt || q.count || 0}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-gray-500">{(q.user_id || '').slice(0, 8)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Low Rated Answers Tab */}
      {!loading && activeTab === 'low' && (
        <div>
          <button onClick={fetchLowAnswers} className="mb-4 px-3 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">
            刷新
          </button>
          {lowAnswers.length === 0 ? (
            <div className="text-center py-12 text-gray-400">暂无低分回答</div>
          ) : (
            <div className="space-y-3">
              {lowAnswers.map((a, i) => (
                <div key={i} className="bg-white border border-gray-200 rounded-xl p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 bg-red-100 text-red-700 rounded-full text-xs font-medium">
                        评分: {a.rating}/5
                      </span>
                      {a.reason_tags?.length > 0 && (
                        <div className="flex gap-1">
                          {a.reason_tags.map((tag, j) => (
                            <span key={j} className="px-2 py-0.5 bg-yellow-100 text-yellow-700 rounded-full text-xs">{tag}</span>
                          ))}
                        </div>
                      )}
                    </div>
                    <span className="text-xs text-gray-400">{a.created_at ? new Date(a.created_at).toLocaleString() : ''}</span>
                  </div>
                  {a.comment && <div className="text-sm text-gray-600 mt-2">{a.comment}</div>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Audit Logs Tab */}
      {!loading && activeTab === 'audit' && (
        <div>
          <div className="flex items-center gap-3 mb-4">
            <select
              value={auditFilter.action}
              onChange={(e) => setAuditFilter({ ...auditFilter, action: e.target.value })}
              className="px-3 py-1.5 border border-gray-300 rounded text-sm outline-none"
            >
              <option value="">全部操作</option>
              <option value="login">登录</option>
              <option value="search">检索</option>
              <option value="chat">问答</option>
              <option value="submit_feedback">提交反馈</option>
            </select>
            <button onClick={fetchAuditLogs} className="px-3 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">
              查询
            </button>
          </div>
          {auditLogs.length === 0 ? (
            <div className="text-center py-12 text-gray-400">暂无审计日志</div>
          ) : (
            <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="text-left px-4 py-2 font-medium text-gray-600">操作</th>
                    <th className="text-left px-4 py-2 font-medium text-gray-600">资源类型</th>
                    <th className="text-left px-4 py-2 font-medium text-gray-600">状态码</th>
                    <th className="text-left px-4 py-2 font-medium text-gray-600">IP</th>
                    <th className="text-left px-4 py-2 font-medium text-gray-600">时间</th>
                  </tr>
                </thead>
                <tbody>
                  {auditLogs.map((log) => (
                    <tr key={log.id} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="px-4 py-2 font-medium">{log.action}</td>
                      <td className="px-4 py-2 text-gray-500">{log.resource_type || '-'}</td>
                      <td className="px-4 py-2">
                        <span className={`px-1.5 py-0.5 rounded text-xs ${
                          log.status_code >= 400 ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                        }`}>
                          {log.status_code || '-'}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-gray-500">{log.ip_address || '-'}</td>
                      <td className="px-4 py-2 text-gray-400 text-xs">{log.created_at ? new Date(log.created_at).toLocaleString() : ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
