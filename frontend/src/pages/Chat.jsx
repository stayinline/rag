import React, { useState, useRef, useEffect } from 'react'
import { listKbs, chat, search, submitFeedback } from '../api'

export default function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [kbs, setKbs] = useState([])
  const [selectedKbs, setSelectedKbs] = useState([])
  const [showSources, setShowSources] = useState(false)
  const [sources, setSources] = useState([])
  const [conversationId, setConversationId] = useState(null)
  const [showKbSelector, setShowKbSelector] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    listKbs().then((data) => {
      const active = data.items?.filter((k) => k.is_active !== false) || data
      setKbs(Array.isArray(active) ? active : [])
    }).catch(() => {})
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const toggleKb = (id) => {
    setSelectedKbs((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    )
  }

  const handleSend = async () => {
    if (!input.trim() || loading) return
    const query = input.trim()
    setInput('')
    setLoading(true)

    const userMsg = { role: 'user', content: query }
    const assistantMsg = { role: 'assistant', content: '', streaming: true, messageId: null }
    setMessages((prev) => [...prev, userMsg, assistantMsg])
    setSources([])
    setShowSources(false)

    try {
      // First do a search to get sources
      search(query, selectedKbs.length ? selectedKbs : undefined, 5)
        .then((data) => {
          const items = data?.items || data?.results || []
          if (items.length > 0) {
            setSources(items.slice(0, 5))
          }
        })
        .catch(() => {})

      const kbIds = selectedKbs.length ? selectedKbs : undefined
      const resp = await chat(query, kbIds, conversationId)
      const answer = resp?.answer || resp?.content || resp?.message || '抱歉，未能生成回答。'

      setMessages((prev) => {
        const updated = [...prev]
        const lastIdx = updated.length - 1
        updated[lastIdx] = {
          ...updated[lastIdx],
          content: answer,
          streaming: false,
          messageId: resp?.message_id || resp?.id || null,
        }
        return updated
      })
      if (resp?.conversation_id) setConversationId(resp.conversation_id)
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

  const handleFeedback = async (messageId, rating) => {
    try {
      await submitFeedback(messageId, { message_id: messageId, rating, reason_tags: [], comment: '' })
    } catch {}
  }

  const handleNewChat = () => {
    setMessages([])
    setConversationId(null)
    setSources([])
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="font-semibold text-gray-800">智能问答</h2>
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
        {messages.length > 0 && (
          <button onClick={handleNewChat} className="text-sm px-3 py-1 rounded-lg bg-blue-50 text-blue-600 hover:bg-blue-100">
            新对话
          </button>
        )}
      </div>

      {/* Messages */}
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
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-2xl rounded-xl px-4 py-3 ${
              msg.role === 'user'
                ? 'bg-blue-600 text-white'
                : 'bg-white border border-gray-200 text-gray-800'
            }`}>
              <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
              {msg.role === 'assistant' && !msg.streaming && msg.messageId && (
                <div className="flex items-center gap-2 mt-2 pt-2 border-t border-gray-100">
                  <span className="text-xs text-gray-400">评价：</span>
                  <button onClick={() => handleFeedback(msg.messageId, 5)} className="text-xs hover:text-green-600">👍</button>
                  <button onClick={() => handleFeedback(msg.messageId, 1)} className="text-xs hover:text-red-600">👎</button>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Sources Panel */}
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
                    {src.page_start && `第${src.page_start}页`}
                    {src.score && ` | 相关度 ${(src.score * 100).toFixed(1)}%`}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Input */}
      <div className="bg-white border-t border-gray-200 p-4">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            placeholder="输入您的问题..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none text-sm"
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="px-5 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-medium"
          >
            {loading ? '思考中...' : '发送'}
          </button>
        </div>
      </div>
    </div>
  )
}
