import React, { useEffect, useState } from 'react'
import { listKbs, uploadPaper, importPaperByDoi, importPaperByPmid } from '../api'

const getErrorMessage = (err) => {
  const detail = err?.response?.data?.detail

  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      if (typeof item === 'string') return item
      if (!item || typeof item !== 'object') return String(item)

      const location = Array.isArray(item.loc) ? item.loc.join('.') : item.loc
      const message = item.msg || item.message || item.type || JSON.stringify(item)
      return location ? `${location}: ${message}` : message
    }).filter(Boolean)

    return messages.join('；') || err?.message || '请求失败'
  }

  if (typeof detail === 'string') return detail
  if (detail && typeof detail === 'object') return detail.message || detail.msg || JSON.stringify(detail)

  return err?.message || '请求失败'
}

export default function PaperHub() {
  const [activeTab, setActiveTab] = useState('upload') // upload | doi | pmid
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [kbs, setKbs] = useState([])
  const [kbsLoading, setKbsLoading] = useState(false)
  const [kbsError, setKbsError] = useState('')
  const [selectedKbId, setSelectedKbId] = useState('')

  // Upload state
  const [uploadFile, setUploadFile] = useState(null)

  // DOI/PMID state
  const [doiInput, setDoiInput] = useState('')
  const [pmidInput, setPmidInput] = useState('')

  useEffect(() => {
    fetchKbs()
  }, [])

  const fetchKbs = async () => {
    setKbsLoading(true)
    setKbsError('')
    try {
      const data = await listKbs()
      const items = data.items || data || []
      setKbs(items)
      setSelectedKbId((current) => {
        if (current && items.some((kb) => kb.id === current)) return current
        return items[0]?.id || ''
      })
    } catch (err) {
      setKbs([])
      setSelectedKbId('')
      setKbsError(getErrorMessage(err))
    } finally {
      setKbsLoading(false)
    }
  }

  const requireSelectedKb = () => {
    if (selectedKbId) return true
    setError('请先选择知识库')
    return false
  }

  const handleUpload = async () => {
    if (!uploadFile) return
    if (!requireSelectedKb()) return
    setLoading(true)
    setError('')
    setResult(null)
    const formData = new FormData()
    formData.append('file', uploadFile)
    formData.append('kb_id', selectedKbId)
    try {
      const data = await uploadPaper(formData)
      setResult(data)
      setUploadFile(null)
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  const handleDoiImport = async () => {
    if (!doiInput.trim()) return
    if (!requireSelectedKb()) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const data = await importPaperByDoi(doiInput.trim(), selectedKbId)
      setResult(data)
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  const handlePmidImport = async () => {
    if (!pmidInput.trim()) return
    if (!requireSelectedKb()) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const data = await importPaperByPmid(pmidInput.trim(), selectedKbId)
      setResult(data)
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h2 className="text-xl font-semibold text-gray-800 mb-6">论文中心</h2>

      <div className="mb-6 flex flex-wrap items-center gap-3">
        <label className="text-sm font-medium text-gray-700" htmlFor="paper-kb-select">目标知识库</label>
        <select
          id="paper-kb-select"
          value={selectedKbId}
          onChange={(e) => setSelectedKbId(e.target.value)}
          disabled={kbsLoading || kbs.length === 0}
          className="min-w-64 px-3 py-2 border border-gray-300 rounded-lg text-sm outline-none focus:border-blue-500 disabled:bg-gray-100 disabled:text-gray-400"
        >
          {kbs.length === 0 ? (
            <option value="">{kbsLoading ? '加载中...' : '暂无知识库'}</option>
          ) : (
            kbs.map((kb) => (
              <option key={kb.id} value={kb.id}>{kb.name}</option>
            ))
          )}
        </select>
        <button
          onClick={fetchKbs}
          disabled={kbsLoading}
          className="px-3 py-2 bg-gray-100 text-gray-600 rounded-lg text-sm hover:bg-gray-200 disabled:opacity-50"
        >
          刷新
        </button>
      </div>

      {kbsError && (
        <div className="mb-6 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-600">
          知识库加载失败：{kbsError}
        </div>
      )}

      {!kbsLoading && !kbsError && kbs.length === 0 && (
        <div className="mb-6 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-sm text-yellow-700">
          请先在知识库管理中创建知识库
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 bg-gray-100 rounded-lg p-1 mb-6 w-fit">
        {[
          { key: 'upload', label: '上传 PDF' },
          { key: 'doi', label: 'DOI 导入' },
          { key: 'pmid', label: 'PMID 导入' },
        ].map((tab) => (
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

      {/* Content */}
      <div className="bg-white border border-gray-200 rounded-xl p-6">
        {activeTab === 'upload' && (
          <div>
            <h3 className="font-medium text-gray-700 mb-3">上传 SCI 论文 PDF</h3>
            <p className="text-sm text-gray-500 mb-4">系统将自动解析论文结构、提取元数据和引用关系</p>
            <div className="flex items-center gap-3">
              <input
                type="file"
                accept=".pdf"
                onChange={(e) => setUploadFile(e.target.files[0])}
                className="text-sm"
              />
              <button
                onClick={handleUpload}
                disabled={!uploadFile || !selectedKbId || loading}
                className="px-4 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
              >
                {loading ? '解析中...' : '上传并解析'}
              </button>
            </div>
          </div>
        )}

        {activeTab === 'doi' && (
          <div>
            <h3 className="font-medium text-gray-700 mb-3">通过 DOI 导入</h3>
            <p className="text-sm text-gray-500 mb-4">输入论文 DOI 号从 CrossRef 获取元数据</p>
            <div className="flex items-center gap-3">
              <input
                type="text"
                value={doiInput}
                onChange={(e) => setDoiInput(e.target.value)}
                placeholder="例如: 10.1038/s41586-023-06789-9"
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm outline-none focus:border-blue-500"
              />
              <button
                onClick={handleDoiImport}
                disabled={!doiInput.trim() || !selectedKbId || loading}
                className="px-4 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
              >
                {loading ? '导入中...' : '导入'}
              </button>
            </div>
          </div>
        )}

        {activeTab === 'pmid' && (
          <div>
            <h3 className="font-medium text-gray-700 mb-3">通过 PMID 导入</h3>
            <p className="text-sm text-gray-500 mb-4">输入 PubMed ID 从 PubMed 获取元数据</p>
            <div className="flex items-center gap-3">
              <input
                type="text"
                value={pmidInput}
                onChange={(e) => setPmidInput(e.target.value)}
                placeholder="例如: 12345678"
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm outline-none focus:border-blue-500"
              />
              <button
                onClick={handlePmidImport}
                disabled={!pmidInput.trim() || !selectedKbId || loading}
                className="px-4 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
              >
                {loading ? '导入中...' : '导入'}
              </button>
            </div>
          </div>
        )}

        {/* Result / Error */}
        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-600">
            {error}
          </div>
        )}

        {result && (
          <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
            <div className="font-medium text-green-800 text-sm mb-2">操作成功</div>
            <div className="text-xs text-gray-600 space-y-1">
              {result.title && <div><span className="font-medium">标题：</span>{result.title}</div>}
              {result.doi && <div><span className="font-medium">DOI：</span>{result.doi}</div>}
              {result.paper_id && <div><span className="font-medium">论文 ID：</span>{result.paper_id}</div>}
              {result.document_id && <div><span className="font-medium">文档 ID：</span>{result.document_id}</div>}
              {result.id && <div><span className="font-medium">ID：</span>{result.id}</div>}
              <pre className="mt-2 text-xs text-gray-500 overflow-auto max-h-40">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>

      {/* Info */}
      <div className="mt-6 bg-blue-50 border border-blue-200 rounded-xl p-4">
        <h4 className="font-medium text-blue-800 text-sm mb-2">论文智能处理</h4>
        <ul className="text-xs text-blue-700 space-y-1">
          <li>自动提取论文元数据（标题、作者、期刊、摘要等）</li>
          <li>结构化解析章节内容（摘要、方法、结果、讨论）</li>
          <li>构建引用关系和相似论文推荐</li>
          <li>支持 PICO 证据摘要提取</li>
        </ul>
      </div>
    </div>
  )
}
