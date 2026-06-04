import React, { useEffect, useRef, useState } from 'react'
import {
  chat,
  deleteConversation,
  getConversation,
  listConversations,
  listKbs,
  search,
  submitFeedback,
} from '../api'

const NEGATIVE_REASON_TAGS = [
  { id: 'incorrect', label: '回答不准确' },
  { id: 'missing_source', label: '缺少引用来源' },
  { id: 'outdated', label: '信息过时' },
  { id: 'unsafe', label: '内容不当' },
]

function resolveAnswerId(resp) {
  const id = resp?.message_id || resp?.trace_id
  if (!id || typeof id !== 'string') return null
  return id
}

function formatConversationTime(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function normalizeHistoryMessage(message) {
  return {
    role: message.role,
    content: message.content,
    streaming: false,
    messageId: message.role === 'assistant' ? message.id : null,
    sources: message.sources || [],
  }
}

export default function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [kbs, setKbs] = useState([])
  const [selectedKbs, setSelectedKbs] = useState([])
  const [showSources, setShowSources] = useState(false)
  const [sources, setSources] = useState([])
  const [conversationId, setConversationId] = useState(null)
  const [conversations, setConversations] = useState([])
  const [conversationTotal, setConversationTotal] = useState(0)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [activeConversationLoading, setActiveConversationLoading] = useState(false)
  const [showKbSelector, setShowKbSelector] = useState(false)
  const [feedbackByMessage, setFeedbackByMessage] = useState({})
  const [expandedReasonFor, setExpandedReasonFor] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')
  const messagesEndRef = useRef(null)

  useEffect(() => {
    listKbs()
      .then((data) => {
        const active = data.items?.filter((k) => k.is_active !== false) || data
        setKbs(Array.isArray(active) ? active : [])
      })
      .catch(() => {})
    loadConversations()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const loadConversations = async () => {
    setHistoryLoading(true)
    try {
      const data = await listConversations({ limit: 50 })
      setConversations(data.items || [])
      setConversationTotal(data.total || 0)
    } catch (err) {
      setErrorMessage(err.response?.data?.detail || '加载历史会话失败')
    } finally {
      setHistoryLoading(false)
    }
  }

  const toggleKb = (id) => {
    setSelectedKbs((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    )
  }

  const handleSelectConversation = async (id) => {
    if (!id || id === conversationId || loading) return
    setActiveConversationLoading(true)
    setErrorMessage('')
    setShowKbSelector(false)
    try {
      const data = await getConversation(id)
      setConversationId(data.id)
      setMessages((data.messages || []).map(normalizeHistoryMessage))
      setSelectedKbs((data.kb_ids || []).map(String))
      const lastAssistant = [...(data.messages || [])].reverse().find((msg) => msg.role === 'assistant')
      setSources(lastAssistant?.sources || [])
      setShowSources(false)
      setFeedbackByMessage({})
      setExpandedReasonFor(null)
    } catch (err) {
      setErrorMessage(err.response?.data?.detail || '加载会话失败')
    } finally {
      setActiveConversationLoading(false)
    }
  }

  const handleSend = async () => {
    if (!input.trim() || loading || activeConversationLoading) return
    const query = input.trim()
    setInput('')
    setLoading(true)
    setErrorMessage('')

    const userMsg = { role: 'user', content: query }
    const assistantMsg = { role: 'assistant', content: '', streaming: true, messageId: null, sources: [] }
    setMessages((prev) => [...prev, userMsg, assistantMsg])
    setSources([])
    setShowSources(false)

    try {
      search(query, selectedKbs.length ? selectedKbs : undefined, 5)
        .then((data) => {
          const items = data?.items || data?.results || []
          if (items.length > 0) setSources(items.slice(0, 5))
        })
        .catch(() => {})

      const kbIds = selectedKbs.length ? selectedKbs : undefined
      const resp = await chat(query, kbIds, conversationId)
      const answer = resp?.answer || resp?.content || resp?.message || '抱歉，未能生成回答。'
      const messageId = resolveAnswerId(resp)
      const answerSources = resp?.sources || []

      setMessages((prev) => {
        const updated = [...prev]
        const lastIdx = updated.length - 1
        updated[lastIdx] = {
          ...updated[lastIdx],
          content: answer,
          streaming: false,
          messageId,
          sources: answerSources,
        }
        return updated
      })
      if (answerSources.length > 0) setSources(answerSources)
      if (resp?.conversation_id) setConversationId(resp.conversation_id)
      await loadConversations()
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          content: `请求失败: ${err.response?.data?.detail || err.message}`,
          streaming: false,
        }
        return updated
      })
    } finally {
      setLoading(false)
    }
  }

  const handleFeedback = async (messageId, rating, reasonTags = []) => {
    if (!messageId || feedbackByMessage[messageId]) return
    setFeedbackByMessage((prev) => ({ ...prev, [messageId]: 'submitting' }))
    setExpandedReasonFor(null)
    try {
      await submitFeedback(messageId, {
        message_id: messageId,
        rating,
        reason_tags: reasonTags,
        comment: '',
      })
      setFeedbackByMessage((prev) => ({
        ...prev,
        [messageId]: rating >= 4 ? 'up' : 'down',
      }))
    } catch (err) {
      setFeedbackByMessage((prev) => {
        const next = { ...prev }
        delete next[messageId]
        return next
      })
      const detail = err.response?.data?.detail
      window.alert(typeof detail === 'string' ? detail : '提交反馈失败，请稍后重试')
    }
  }

  const handleThumbsDown = (messageId) => {
    if (!messageId || feedbackByMessage[messageId]) return
    setExpandedReasonFor((prev) => (prev === messageId ? null : messageId))
  }

  const handleNewChat = () => {
    setMessages([])
    setConversationId(null)
    setSources([])
    setFeedbackByMessage({})
    setExpandedReasonFor(null)
    setErrorMessage('')
  }

  const handleDeleteConversation = async (id, event) => {
    event.stopPropagation()
    if (!id || loading) return
    const ok = window.confirm('确定删除这条会话记录吗？')
    if (!ok) return
    setErrorMessage('')
    try {
      await deleteConversation(id)
      if (id === conversationId) handleNewChat()
      await loadConversations()
    } catch (err) {
      setErrorMessage(err.response?.data?.detail || '删除会话失败')
    }
  }

  const renderFeedbackBar = (msg) => {
    if (msg.role !== 'assistant' || msg.streaming) return null

    const messageId = msg.messageId
    const state = messageId ? feedbackByMessage[messageId] : null

    if (!messageId) {
      return (
        <div className="mt-2 pt-2 border-t border-gray-100 text-xs text-gray-400">
          本次回答暂无反馈 ID，无法提交评价
        </div>
      )
    }

    if (state === 'up') {
      return (
        <div className="mt-2 pt-2 border-t border-gray-100 text-xs text-green-600">
          感谢您的好评
        </div>
      )
    }

    if (state === 'down') {
      return (
        <div className="mt-2 pt-2 border-t border-gray-100 text-xs text-gray-500">
          已记录您的反馈，我们会持续改进
        </div>
      )
    }

    const isSubmitting = state === 'submitting'
    const showReasons = expandedReasonFor === messageId

    return (
      <div className="mt-2 pt-2 border-t border-gray-100">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-gray-400">这条回答有帮助吗？</span>
          <button
            type="button"
            disabled={isSubmitting}
            onClick={() => handleFeedback(messageId, 5)}
            className="text-xs px-2 py-1 rounded-md border border-gray-200 hover:bg-green-50 hover:border-green-200 disabled:opacity-50"
            title="有帮助"
          >
            有帮助
          </button>
          <button
            type="button"
            disabled={isSubmitting}
            onClick={() => handleThumbsDown(messageId)}
            className="text-xs px-2 py-1 rounded-md border border-gray-200 hover:bg-red-50 hover:border-red-200 disabled:opacity-50"
            title="需要改进"
          >
            需要改进
          </button>
          {isSubmitting && <span className="text-xs text-gray-400">提交中...</span>}
        </div>
        {showReasons && (
          <div className="mt-2 space-y-2">
            <div className="text-xs text-gray-500">请选择原因（可选）：</div>
            <div className="flex flex-wrap gap-1.5">
              {NEGATIVE_REASON_TAGS.map((tag) => (
                <button
                  key={tag.id}
                  type="button"
                  disabled={isSubmitting}
                  onClick={() => handleFeedback(messageId, 1, [tag.id])}
                  className="text-xs px-2 py-1 rounded-full bg-gray-100 hover:bg-red-100 text-gray-600 disabled:opacity-50"
                >
                  {tag.label}
                </button>
              ))}
              <button
                type="button"
                disabled={isSubmitting}
                onClick={() => handleFeedback(messageId, 2)}
                className="text-xs px-2 py-1 rounded-full border border-gray-200 text-gray-500 hover:bg-gray-50 disabled:opacity-50"
              >
                跳过，直接提交
              </button>
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="flex h-full bg-gray-50">
      <aside className="w-72 shrink-0 border-r border-gray-200 bg-white flex flex-col">
        <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-gray-800">历史会话</h3>
            <p className="text-xs text-gray-400 mt-0.5">{conversationTotal} 条记录</p>
          </div>
          <button
            onClick={handleNewChat}
            className="text-sm px-3 py-1.5 rounded-md bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
            disabled={loading}
          >
            新对话
          </button>
        </div>

        <div className="flex-1 overflow-auto p-2">
          {historyLoading && <div className="text-xs text-gray-400 px-2 py-3">加载历史中...</div>}
          {!historyLoading && conversations.length === 0 && (
            <div className="text-xs text-gray-400 px-2 py-3">暂无历史会话</div>
          )}
          {conversations.map((item) => {
            const active = item.id === conversationId
            return (
              <div
                key={item.id}
                onClick={() => handleSelectConversation(item.id)}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') handleSelectConversation(item.id)
                }}
                className={`group w-full text-left px-3 py-2 rounded-md mb-1 border cursor-pointer ${
                  active
                    ? 'bg-blue-50 border-blue-100'
                    : 'border-transparent hover:bg-gray-50'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className={`text-sm truncate ${active ? 'text-blue-700 font-medium' : 'text-gray-700'}`}>
                      {item.title || '未命名会话'}
                    </div>
                    <div className="text-xs text-gray-400 mt-1">
                      {formatConversationTime(item.last_message_at || item.created_at)}
                      {item.message_count ? ` · ${item.message_count} 条` : ''}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={(event) => handleDeleteConversation(item.id, event)}
                    className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-600 text-xs px-1"
                    title="删除会话"
                  >
                    删除
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      </aside>

      <div className="flex flex-col flex-1 min-w-0">
        <div className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3 min-w-0">
            <h2 className="font-semibold text-gray-800 shrink-0">智能问答</h2>
            <div className="relative">
              <button
                onClick={() => setShowKbSelector(!showKbSelector)}
                className="text-sm px-3 py-1 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-600"
              >
                {selectedKbs.length ? `已选 ${selectedKbs.length} 个知识库` : '选择知识库'}
              </button>
              {showKbSelector && (
                <div className="absolute top-full left-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg p-3 w-64 z-10">
                  <div className="text-xs text-gray-500 mb-2">选择检索范围（不选则检索全部）</div>
                  {kbs.length === 0 && <div className="text-xs text-gray-400">暂无知识库</div>}
                  {kbs.map((kb) => (
                    <label key={kb.id} className="flex items-center gap-2 py-1 text-sm cursor-pointer">
                      <input
                        type="checkbox"
                        checked={selectedKbs.includes(kb.id)}
                        onChange={() => toggleKb(kb.id)}
                        className="rounded"
                      />
                      <span className="truncate">{kb.name}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>
          </div>
          {activeConversationLoading && <span className="text-sm text-gray-400">加载会话中...</span>}
        </div>

        {errorMessage && (
          <div className="px-4 py-2 bg-red-50 border-b border-red-100 text-sm text-red-600">
            {errorMessage}
          </div>
        )}

        <div className="flex-1 overflow-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full text-gray-400">
              <div className="text-center">
                <div className="text-4xl mb-2">💬</div>
                <p className="text-lg">开始提问吧</p>
                <p className="text-sm mt-1">基于您的知识库进行智能问答</p>
              </div>
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={`${msg.role}-${i}`} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-2xl rounded-xl px-4 py-3 ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white border border-gray-200 text-gray-800'
              }`}>
                <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
                {renderFeedbackBar(msg)}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {sources.length > 0 && (
          <div className="bg-gray-50 border-t border-gray-200 px-4 py-2">
            <button
              onClick={() => setShowSources(!showSources)}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              {showSources ? '收起' : '查看'}参考来源 ({sources.length})
            </button>
            {showSources && (
              <div className="mt-2 grid grid-cols-1 gap-2 max-h-40 overflow-auto">
                {sources.map((src, i) => (
                  <div key={i} className="bg-white rounded-lg p-2 text-xs border border-gray-200">
                    <div className="font-medium text-gray-700">{src.document_title || '未知文档'}</div>
                    <div className="text-gray-500 mt-1 line-clamp-2">{src.content_preview || ''}</div>
                    <div className="text-gray-400 mt-1">
                      {src.page_start ? `第 ${src.page_start} 页` : ''}
                      {src.score ? ` | 相关度 ${(src.score * 100).toFixed(1)}%` : ''}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <div className="bg-white border-t border-gray-200 p-4">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder="输入您的问题..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none text-sm"
              disabled={loading || activeConversationLoading}
            />
            <button
              onClick={handleSend}
              disabled={loading || activeConversationLoading || !input.trim()}
              className="px-5 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-medium"
            >
              {loading ? '思考中...' : '发送'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
