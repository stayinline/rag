import React, { useState, useEffect } from 'react'
import { listEvalSets, createEvalSet, listEvalQuestions, runEvaluation, getEvaluationRun } from '../api'

export default function Evaluation() {
  const [evalSets, setEvalSets] = useState([])
  const [selectedSet, setSelectedSet] = useState(null)
  const [questions, setQuestions] = useState([])
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [newSetName, setNewSetName] = useState('')
  const [newSetScenario, setNewSetScenario] = useState('qa')
  const [newSetDescription, setNewSetDescription] = useState('')
  const [questionInputs, setQuestionInputs] = useState([{ question: '', category: '', difficulty: 'medium' }])
  const [runningEval, setRunningEval] = useState(null)
  const [evalResult, setEvalResult] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => { fetchSets() }, [])

  const fetchSets = async () => {
    try {
      const data = await listEvalSets()
      setEvalSets(data.items || data)
    } catch {}
  }

  const handleSelectSet = async (s) => {
    setSelectedSet(s)
    try {
      const data = await listEvalQuestions(s.id)
      setQuestions(data.items || data)
    } catch {}
  }

  const handleCreateSet = async () => {
    if (!newSetName.trim()) return
    setLoading(true)
    try {
      const questions = questionInputs
        .filter((q) => q.question.trim())
        .map((q) => ({
          question: q.question.trim(),
          category: q.category || null,
          difficulty: q.difficulty || 'medium',
        }))

      await createEvalSet({
        name: newSetName.trim(),
        scenario: newSetScenario,
        description: newSetDescription || null,
        questions,
      })
      setNewSetName('')
      setNewSetScenario('qa')
      setNewSetDescription('')
      setQuestionInputs([{ question: '', category: '', difficulty: 'medium' }])
      setShowCreateForm(false)
      fetchSets()
    } catch (err) {
      alert('创建失败: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  const addQuestion = () => {
    setQuestionInputs([...questionInputs, { question: '', category: '', difficulty: 'medium' }])
  }

  const updateQuestion = (i, field, value) => {
    setQuestionInputs((prev) => {
      const updated = [...prev]
      updated[i] = { ...updated[i], [field]: value }
      return updated
    })
  }

  const removeQuestion = (i) => {
    setQuestionInputs((prev) => prev.filter((_, idx) => idx !== i))
  }

  const handleRunEval = async () => {
    if (!selectedSet) return
    setLoading(true)
    setEvalResult(null)
    try {
      const data = await runEvaluation(selectedSet.id)
      setRunningEval(data)
      // Poll for results
      pollResult(data.id)
    } catch (err) {
      alert('启动评测失败: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  const pollResult = async (runId) => {
    const poll = async () => {
      try {
        const data = await getEvaluationRun(runId)
        setEvalResult(data)
        if (data.status === 'running' || data.status === 'pending') {
          setTimeout(poll, 3000)
        }
      } catch {}
    }
    setTimeout(poll, 3000)
  }

  const statusColor = (status) => {
    const colors = {
      pending: 'bg-gray-100 text-gray-600',
      running: 'bg-blue-100 text-blue-700',
      completed: 'bg-green-100 text-green-700',
      failed: 'bg-red-100 text-red-700',
    }
    return colors[status] || colors.pending
  }

  return (
    <div className="flex h-full">
      {/* Eval Sets Sidebar */}
      <div className="w-72 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          <h2 className="font-semibold text-gray-800">评测集</h2>
          <button
            onClick={() => setShowCreateForm(!showCreateForm)}
            className="text-sm px-2 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            + 新建
          </button>
        </div>

        {showCreateForm && (
          <div className="p-3 border-b border-gray-200 bg-gray-50 max-h-96 overflow-auto">
            <input
              type="text"
              value={newSetName}
              onChange={(e) => setNewSetName(e.target.value)}
              placeholder="评测集名称"
              className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm mb-2 outline-none focus:border-blue-500"
            />
            <select
              value={newSetScenario}
              onChange={(e) => setNewSetScenario(e.target.value)}
              className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm mb-2 outline-none"
            >
              <option value="qa">问答</option>
              <option value="search">检索</option>
              <option value="paper_extract">论文抽取</option>
            </select>
            <input
              type="text"
              value={newSetDescription}
              onChange={(e) => setNewSetDescription(e.target.value)}
              placeholder="描述（可选）"
              className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm mb-3 outline-none focus:border-blue-500"
            />

            <div className="text-xs font-medium text-gray-600 mb-2">标准问题</div>
            {questionInputs.map((q, i) => (
              <div key={i} className="mb-2 p-2 bg-white rounded border border-gray-200">
                <input
                  type="text"
                  value={q.question}
                  onChange={(e) => updateQuestion(i, 'question', e.target.value)}
                  placeholder="问题"
                  className="w-full px-2 py-1 border border-gray-200 rounded text-xs mb-1 outline-none"
                />
                <div className="flex gap-1">
                  <input
                    type="text"
                    value={q.category}
                    onChange={(e) => updateQuestion(i, 'category', e.target.value)}
                    placeholder="分类"
                    className="flex-1 px-2 py-1 border border-gray-200 rounded text-xs outline-none"
                  />
                  <select
                    value={q.difficulty}
                    onChange={(e) => updateQuestion(i, 'difficulty', e.target.value)}
                    className="px-2 py-1 border border-gray-200 rounded text-xs outline-none"
                  >
                    <option value="easy">简单</option>
                    <option value="medium">中等</option>
                    <option value="hard">困难</option>
                  </select>
                  {questionInputs.length > 1 && (
                    <button onClick={() => removeQuestion(i)} className="text-red-400 hover:text-red-600 px-1">×</button>
                  )}
                </div>
              </div>
            ))}
            <button onClick={addQuestion} className="text-xs text-blue-600 hover:text-blue-700 mb-3">
              + 添加问题
            </button>

            <button
              onClick={handleCreateSet}
              disabled={loading || !newSetName.trim()}
              className="w-full px-3 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
            >
              创建评测集
            </button>
          </div>
        )}

        <div className="flex-1 overflow-auto">
          {evalSets.map((s) => (
            <div
              key={s.id}
              onClick={() => handleSelectSet(s)}
              className={`p-3 border-b border-gray-100 cursor-pointer hover:bg-gray-50 ${
                selectedSet?.id === s.id ? 'bg-blue-50 border-l-2 border-l-blue-500' : ''
              }`}
            >
              <div className="font-medium text-sm text-gray-800">{s.name}</div>
              <div className="text-xs text-gray-400 mt-1 flex items-center gap-2">
                <span className={`px-1.5 py-0.5 rounded ${statusColor(s.status)}`}>{s.status}</span>
                <span>{s.scenario}</span>
                <span>· {s.question_count || 0} 题</span>
              </div>
            </div>
          ))}
          {evalSets.length === 0 && (
            <div className="p-6 text-center text-gray-400 text-sm">暂无评测集</div>
          )}
        </div>
      </div>

      {/* Detail Panel */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {selectedSet ? (
          <>
            <div className="p-4 border-b border-gray-200 bg-white flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-gray-800">{selectedSet.name}</h3>
                {selectedSet.description && <p className="text-sm text-gray-500 mt-1">{selectedSet.description}</p>}
              </div>
              <button
                onClick={handleRunEval}
                disabled={loading || questions.length === 0}
                className="px-4 py-1.5 bg-green-600 text-white rounded text-sm hover:bg-green-700 disabled:opacity-50"
              >
                运行评测
              </button>
            </div>

            {/* Eval Result */}
            {evalResult && (
              <div className="p-4 bg-green-50 border-b border-green-200">
                <div className="flex items-center gap-3">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${statusColor(evalResult.status)}`}>
                    {evalResult.status}
                  </span>
                  {evalResult.metrics && Object.keys(evalResult.metrics).length > 0 && (
                    <div className="flex gap-4 text-sm">
                      {Object.entries(evalResult.metrics).map(([k, v]) => (
                        <span key={k} className="text-gray-600">
                          {k}: <strong>{typeof v === 'number' ? v.toFixed(3) : v}</strong>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Questions List */}
            <div className="flex-1 overflow-auto p-4">
              <h4 className="text-sm font-medium text-gray-700 mb-3">问题列表 ({questions.length})</h4>
              <div className="space-y-2">
                {questions.map((q) => (
                  <div key={q.id} className="bg-white border border-gray-200 rounded-lg p-3">
                    <div className="font-medium text-sm text-gray-800">{q.question}</div>
                    <div className="flex gap-2 mt-2">
                      {q.category && <span className="px-2 py-0.5 bg-blue-50 text-blue-600 rounded text-xs">{q.category}</span>}
                      {q.difficulty && (
                        <span className={`px-2 py-0.5 rounded text-xs ${
                          q.difficulty === 'hard' ? 'bg-red-50 text-red-600' :
                          q.difficulty === 'medium' ? 'bg-yellow-50 text-yellow-600' :
                          'bg-green-50 text-green-600'
                        }`}>
                          {q.difficulty === 'easy' ? '简单' : q.difficulty === 'medium' ? '中等' : '困难'}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
                {questions.length === 0 && (
                  <div className="text-center text-gray-400 text-sm py-8">暂无问题</div>
                )}
              </div>
            </div>
          </>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <div className="text-4xl mb-2">🎯</div>
              <p>请选择或创建一个评测集</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
