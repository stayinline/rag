import React, { useState, useEffect } from 'react'
import { listKbs, createKb, deleteKb, listDocuments, uploadDocument, deleteDocument, getIngestionJob } from '../api'

export default function KnowledgeBases() {
  const [kbs, setKbs] = useState([])
  const [selectedKb, setSelectedKb] = useState(null)
  const [documents, setDocuments] = useState([])
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [newKb, setNewKb] = useState({ name: '', description: '' })
  const [uploadFile, setUploadFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('list') // list | docs

  useEffect(() => { fetchKbs() }, [])

  const fetchKbs = async () => {
    try {
      const data = await listKbs()
      setKbs(data.items || data)
    } catch {}
  }

  const handleCreateKb = async () => {
    if (!newKb.name.trim()) return
    setLoading(true)
    try {
      await createKb(newKb)
      setNewKb({ name: '', description: '' })
      setShowCreateForm(false)
      fetchKbs()
    } catch (err) {
      alert('创建失败: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteKb = async (id) => {
    if (!confirm('确定删除此知识库吗？')) return
    try {
      await deleteKb(id)
      if (selectedKb?.id === id) {
        setSelectedKb(null)
        setDocuments([])
      }
      fetchKbs()
    } catch {}
  }

  const handleSelectKb = async (kb) => {
    setSelectedKb(kb)
    setActiveTab('docs')
    try {
      const data = await listDocuments(kb.id)
      setDocuments(data.items || data)
    } catch {}
  }

  const handleUpload = async () => {
    if (!uploadFile || !selectedKb) return
    setLoading(true)
    const formData = new FormData()
    formData.append('file', uploadFile)
    try {
      await uploadDocument(selectedKb.id, formData)
      setUploadFile(null)
      const data = await listDocuments(selectedKb.id)
      setDocuments(data.items || data)
    } catch (err) {
      alert('上传失败: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteDoc = async (id) => {
    if (!confirm('确定删除此文档吗？')) return
    try {
      await deleteDocument(id)
      const data = await listDocuments(selectedKb.id)
      setDocuments(data.items || data)
    } catch {}
  }

  const statusBadge = (status) => {
    const colors = {
      ready: 'bg-green-100 text-green-700',
      parsing: 'bg-blue-100 text-blue-700',
      embedding: 'bg-yellow-100 text-yellow-700',
      failed: 'bg-red-100 text-red-700',
      draft: 'bg-gray-100 text-gray-600',
      deleted: 'bg-gray-100 text-gray-400 line-through',
    }
    return <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colors[status] || colors.draft}`}>{status}</span>
  }

  return (
    <div className="flex h-full">
      {/* KB List Sidebar */}
      <div className="w-72 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          <h2 className="font-semibold text-gray-800">知识库</h2>
          <button
            onClick={() => setShowCreateForm(!showCreateForm)}
            className="text-sm px-2 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            + 新建
          </button>
        </div>

        {showCreateForm && (
          <div className="p-3 border-b border-gray-200 bg-gray-50">
            <input
              type="text"
              value={newKb.name}
              onChange={(e) => setNewKb({ ...newKb, name: e.target.value })}
              placeholder="知识库名称"
              className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm mb-2 outline-none focus:border-blue-500"
            />
            <textarea
              value={newKb.description}
              onChange={(e) => setNewKb({ ...newKb, description: e.target.value })}
              placeholder="描述（可选）"
              rows={2}
              className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm mb-2 outline-none focus:border-blue-500 resize-none"
            />
            <div className="flex gap-2">
              <button onClick={handleCreateKb} disabled={loading || !newKb.name.trim()}
                className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50">
                创建
              </button>
              <button onClick={() => setShowCreateForm(false)} className="px-3 py-1 bg-gray-200 text-gray-600 rounded text-sm hover:bg-gray-300">
                取消
              </button>
            </div>
          </div>
        )}

        <div className="flex-1 overflow-auto">
          {kbs.map((kb) => (
            <div
              key={kb.id}
              onClick={() => handleSelectKb(kb)}
              className={`p-3 border-b border-gray-100 cursor-pointer hover:bg-gray-50 ${
                selectedKb?.id === kb.id ? 'bg-blue-50 border-l-2 border-l-blue-500' : ''
              } ${kb.is_active === false ? 'opacity-50' : ''}`}
            >
              <div className="font-medium text-sm text-gray-800">{kb.name}</div>
              {kb.description && <div className="text-xs text-gray-500 mt-1 line-clamp-2">{kb.description}</div>}
              <div className="flex items-center justify-between mt-2">
                <span className={`px-1.5 py-0.5 rounded text-xs ${kb.is_active !== false ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-400'}`}>
                  {kb.is_active !== false ? '活跃' : '已停用'}
                </span>
                <button
                  onClick={(e) => { e.stopPropagation(); handleDeleteKb(kb.id) }}
                  className="text-xs text-red-400 hover:text-red-600"
                >
                  删除
                </button>
              </div>
            </div>
          ))}
          {kbs.length === 0 && (
            <div className="p-6 text-center text-gray-400 text-sm">暂无知识库</div>
          )}
        </div>
      </div>

      {/* Document Panel */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {selectedKb ? (
          <>
            <div className="p-4 border-b border-gray-200 bg-white flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-gray-800">{selectedKb.name}</h3>
                {selectedKb.description && <p className="text-sm text-gray-500 mt-1">{selectedKb.description}</p>}
              </div>
            </div>

            {/* Upload Area */}
            <div className="p-4 border-b border-gray-200 bg-gray-50">
              <div className="flex items-center gap-3">
                <input
                  type="file"
                  onChange={(e) => setUploadFile(e.target.files[0])}
                  className="text-sm"
                  accept=".pdf,.docx,.doc,.txt,.md,.html,.csv,.xlsx"
                />
                <button
                  onClick={handleUpload}
                  disabled={!uploadFile || loading}
                  className="px-4 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
                >
                  {loading ? '上传中...' : '上传文档'}
                </button>
                <span className="text-xs text-gray-400">支持 PDF, DOCX, TXT, MD, HTML 等格式</span>
              </div>
            </div>

            {/* Document List */}
            <div className="flex-1 overflow-auto p-4">
              <h4 className="text-sm font-medium text-gray-700 mb-3">文档列表 ({documents.length})</h4>
              <div className="space-y-2">
                {documents.map((doc) => (
                  <div key={doc.id} className="bg-white border border-gray-200 rounded-lg p-3 flex items-center justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-sm text-gray-800 truncate">{doc.title}</div>
                      <div className="text-xs text-gray-400 mt-1 flex items-center gap-3">
                        {doc.file_type && <span>{doc.file_type}</span>}
                        <span>{new Date(doc.created_at).toLocaleDateString()}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 ml-3">
                      {statusBadge(doc.status)}
                      <button
                        onClick={() => handleDeleteDoc(doc.id)}
                        className="text-xs text-red-400 hover:text-red-600"
                      >
                        删除
                      </button>
                    </div>
                  </div>
                ))}
                {documents.length === 0 && (
                  <div className="text-center text-gray-400 text-sm py-8">暂无文档，请上传文件</div>
                )}
              </div>
            </div>
          </>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <div className="text-4xl mb-2">📚</div>
              <p>请选择或创建一个知识库</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
