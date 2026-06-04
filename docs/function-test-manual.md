# RAG Knowledge Base 功能测试手册

生成日期：2026-06-04  
适用项目路径：`D:\code\rag`

## 1. 测试结论摘要

### 已确认事实

- 后端应用可以启动，`GET /health` 返回 `{"status":"ok"}`。
- 后端 OpenAPI 当前暴露 31 个 `/api/v1` 接口路径/方法组合。
- 当前项目虚拟环境中执行 `python -m pytest`，结果为 `284 passed, 1 warning`。
- 前端在 `frontend/` 下执行 `npm run build`，构建成功。
- 当前配置下，PostgreSQL、Redis、ClickHouse、Weaviate 的只读连通性检查通过。
- 数据库核心表已存在，默认 `admin` 用户存在且启用。
- `POST /api/v1/auth/login` 使用 `admin/admin123` 可以获取 bearer token。

### 待验证假设

- 假设 DashScope 模型密钥和模型名称仍然可用。当前仅从代码配置看到调用方式，没有在本次测试中执行真实生成和真实 embedding。
- 假设 Celery worker 能正常消费 Redis 队列。本次只验证 Redis 可连通，没有启动 worker 执行完整异步任务。
- 假设上传文件解析、论文解析、评测运行在真实 worker 中可完成。当前自动化测试大量使用 mock，不能替代真实异步链路验证。

### 当前完成度评估

项目没有提供正式验收清单，因此无法给出绝对客观的完成百分比。基于当前代码、自动化测试、前后端构建和只读连通性验证，建议按以下方式理解完成度：

- 后端 API 框架与核心模块：约 75% 到 85%。
- 前端页面与接口接入：约 55% 到 65%。
- 真实端到端 RAG 闭环：约 45% 到 60%。
- 生产可用性：约 35% 到 45%。

结论：可以开始功能测试和试用基础模块，但不建议直接按生产可用系统验收。测试时应重点验证文档入库到检索问答、论文中心、评测与分析这几条链路。

## 2. 当前已知限制

### 普通文档入库链路不完整

`POST /api/v1/kbs/{kb_id}/documents` 会保存文件、创建 `documents`、`document_versions`、`ingestion_jobs` 记录，并触发 `parse_document` Celery 任务。  
但当前代码中未看到普通文档上传后自动串联执行 `chunk_and_embed` 和 `publish_document` 的逻辑。因此普通文档上传后可能长期停留在 `draft`，不能进入 Weaviate 的 `ready` 检索状态。

测试影响：知识库创建、文档上传、文档列表可以测试；上传后立即检索或问答可能没有结果。

### 论文中心前端缺少 kb_id

后端 `POST /api/v1/papers/upload` 要求表单字段 `kb_id`。后端 DOI/PMID 导入请求也要求 `kb_id`。  
当前 `frontend/src/pages/PaperHub.jsx` 上传和导入时没有让用户选择知识库，也没有提交有效 `kb_id`。

测试影响：通过前端论文中心上传 PDF、DOI 导入、PMID 导入预计会失败；应优先用 Swagger 或 curl 带上 `kb_id` 测试后端接口。

### 前端搜索 top_k 参数不一致

前端 `search(query, kbIds, limit)` 提交的是 `limit` 字段，后端 `SearchRequest` 接收的是 `top_k`。  
后端默认会使用 `top_k=10`，因此前端传入的 `limit=5` 不会生效。

测试影响：聊天页参考来源数量可能不完全符合前端调用意图，但不会阻断请求。

### 聊天反馈前端难以触发

后端聊天接口返回 `answer`、`trace_id`、`conversation_id`、`sources`、`model`、`prompt_version`，未返回 `message_id`。  
前端只有在 `messageId` 存在时才显示点赞/点踩按钮。

测试影响：通过聊天页提交反馈预计无法直接验证；可用后端接口构造 UUID 单独测试反馈提交。

### `/health` 不代表外部依赖全可用

应用启动时会尝试初始化 Weaviate collection，但异常只打印 warning，不会导致 `/health` 失败。  
因此 `/health=ok` 只能证明 FastAPI 进程可用，不能证明 RAG 全链路可用。

## 3. 测试环境准备

### 3.1 安装依赖

在项目根目录执行：

```powershell
cd D:\code\rag
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
cd frontend
npm install
cd ..
```

如果没有 `.venv`，先创建：

```powershell
cd D:\code\rag
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

### 3.2 初始化数据库和 Weaviate

如果是新环境，执行：

```powershell
cd D:\code\rag
.\.venv\Scripts\python.exe scripts\init_db.py
.\.venv\Scripts\python.exe scripts\seed_user.py
.\.venv\Scripts\python.exe scripts\init_weaviate.py
```

当前代码默认用户来自 `scripts/seed_user.py`：

```text
username: admin
password: admin123
```

正式环境必须修改默认密码和 JWT secret。

### 3.3 启动服务

后端 API：

```powershell
cd D:\code\rag
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Celery worker：

```powershell
cd D:\code\rag
.\.venv\Scripts\celery.exe -A app.workers.celery_app worker --loglevel=info
```

前端：

```powershell
cd D:\code\rag\frontend
npm run dev
```

访问地址：

- 后端健康检查：`http://127.0.0.1:8000/health`
- Swagger：`http://127.0.0.1:8000/docs`
- 前端：`http://127.0.0.1:3000`

## 4. 基础验证

### FT-001 后端健康检查

步骤：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

预期结果：

```json
{"status":"ok"}
```

通过标准：返回 `status=ok`。

失败排查：

- 检查 uvicorn 是否启动。
- 检查端口 8000 是否被占用。
- 检查 Python 依赖是否安装完成。

### FT-002 OpenAPI 页面

步骤：

浏览器打开：

```text
http://127.0.0.1:8000/docs
```

预期结果：

- Swagger 页面可以打开。
- 页面中可见 `auth`、`knowledge-bases`、`documents`、`chat`、`search`、`papers`、`analytics` 等接口分组。

### FT-003 前端页面启动

步骤：

浏览器打开：

```text
http://127.0.0.1:3000
```

预期结果：

- 未登录时进入登录页。
- 页面显示用户名、密码输入框。

## 5. 登录与鉴权测试

### FT-101 登录成功

接口：

```text
POST /api/v1/auth/login
```

请求：

```powershell
$login = Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/auth/login `
  -Method Post `
  -ContentType 'application/json' `
  -Body '{"username":"admin","password":"admin123"}'

$token = $login.access_token
$headers = @{ Authorization = "Bearer $token" }
$login
```

预期结果：

- 返回 `access_token`。
- 返回 `token_type` 为 `bearer`。
- 返回用户 `username=admin`。

前端验证：

- 打开 `http://127.0.0.1:3000/login`。
- 输入 `admin/admin123`。
- 登录成功后跳转到聊天页。

### FT-102 登录失败

请求：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/auth/login `
  -Method Post `
  -ContentType 'application/json' `
  -Body '{"username":"admin","password":"wrong"}'
```

预期结果：

- HTTP 401。
- 错误信息为用户名或密码错误。

### FT-103 未带 token 访问受保护接口

请求：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/kbs
```

预期结果：

- HTTP 401。

## 6. 知识库管理测试

### FT-201 创建知识库

请求：

```powershell
$kb = Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/kbs `
  -Method Post `
  -Headers $headers `
  -ContentType 'application/json' `
  -Body '{"name":"功能测试知识库","description":"用于功能测试"}'

$kbId = $kb.id
$kb
```

预期结果：

- HTTP 201。
- 返回 `id`。
- `name=功能测试知识库`。
- `is_active=true`。

前端验证：

- 进入“知识库”页面。
- 点击“新建”。
- 输入名称和描述。
- 创建后左侧列表出现新知识库。

### FT-202 查询知识库列表

请求：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/kbs `
  -Method Get `
  -Headers $headers
```

预期结果：

- 返回 `items` 和 `total`。
- `items` 中包含刚创建的知识库。

### FT-203 查询单个知识库

请求：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/kbs/$kbId" `
  -Method Get `
  -Headers $headers
```

预期结果：

- 返回指定知识库详情。

### FT-204 更新知识库

请求：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/kbs/$kbId" `
  -Method Patch `
  -Headers $headers `
  -ContentType 'application/json' `
  -Body '{"description":"已更新的功能测试描述"}'
```

预期结果：

- 返回更新后的 `description`。

### FT-205 删除知识库

注意：如果后续还要验证文档、检索和问答，不要立即删除当前 `$kbId`。

请求：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/kbs/$kbId" `
  -Method Delete `
  -Headers $headers
```

预期结果：

- HTTP 204。
- 再次列表查询时该知识库不应出现在未删除列表中。

## 7. 普通文档管理测试

### FT-301 上传普通文档

准备测试文件：

```powershell
$sample = "D:\code\rag\data\files\manual-test.txt"
New-Item -ItemType Directory -Force -Path (Split-Path $sample)
Set-Content -Path $sample -Encoding UTF8 -Value "这是功能测试文档。产品A用于测试知识库检索。"
```

上传：

```powershell
$doc = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/kbs/$kbId/documents" `
  -Method Post `
  -Headers $headers `
  -Form @{ file = Get-Item $sample }

$docId = $doc.id
$doc
```

预期结果：

- HTTP 201。
- 返回文档 `id`。
- `status` 初始通常为 `draft`。
- 文档列表中可以看到该文档。

前端验证：

- 进入“知识库”页面。
- 选择测试知识库。
- 上传 `.txt`、`.pdf` 或 `.docx` 文件。
- 文档列表出现新文件。

### FT-302 查询文档列表

请求：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/kbs/$kbId/documents" `
  -Method Get `
  -Headers $headers
```

预期结果：

- 返回 `items` 和 `total`。
- 包含刚上传的文档。

### FT-303 查询单个文档

请求：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/documents/$docId" `
  -Method Get `
  -Headers $headers
```

预期结果：

- 返回文档详情。

### FT-304 查询 ingestion job

当前上传接口返回的是 `DocumentResponse`，不直接返回 `ingestion_job_id`。  
如需查询 job，需要从数据库或 Swagger 中结合实际记录获取 job id。

建议验证方法：

```powershell
@"
import asyncio
from sqlalchemy import select
from app.database import async_session
from app.models.task import IngestionJob

async def main():
    async with async_session() as session:
        result = await session.execute(select(IngestionJob).order_by(IngestionJob.created_at.desc()).limit(5))
        for job in result.scalars():
            print(job.id, job.document_id, job.job_type, job.status)

asyncio.run(main())
"@ | .\.venv\Scripts\python.exe -
```

预期结果：

- 可以看到最近创建的 `parse` job。
- 当前代码创建 job 时默认状态为 `pending`。

### FT-305 删除文档

注意：如果后续还要验证检索和问答，不要立即删除当前 `$docId`。

请求：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/documents/$docId" `
  -Method Delete `
  -Headers $headers
```

预期结果：

- HTTP 204。
- 文档状态应被标记为 `deleted` 或不再出现在知识库文档列表中。

## 8. 检索测试

### FT-401 空库检索

请求：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/search `
  -Method Post `
  -Headers $headers `
  -ContentType 'application/json' `
  -Body (@{
    query = "产品A"
    kb_ids = @($kbId)
    top_k = 5
  } | ConvertTo-Json)
```

预期结果：

- 如果知识库中没有 `ready` 状态的 Weaviate chunk，返回 `total=0`。
- 如果已有 ready chunk，返回 `results`。

当前风险：

- 普通文档上传不会自动完成 chunk、embedding、publish，因此刚上传文档后检索可能仍然为 0。

### FT-402 检索鉴权

请求中去掉 `$headers`。

预期结果：

- HTTP 401。

## 9. 聊天问答测试

### FT-501 空库或无检索结果问答

请求：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/chat `
  -Method Post `
  -Headers $headers `
  -ContentType 'application/json' `
  -Body (@{
    query = "产品A是什么？"
    kb_ids = @($kbId)
    stream = $false
  } | ConvertTo-Json)
```

预期结果：

- 如果检索不到上下文，返回回答应包含“未找到相关的参考资料”。
- 返回字段包含 `answer`、`trace_id`、`sources`、`model`、`prompt_version`。

### FT-502 有资料问答

前置条件：

- Weaviate 中存在当前组织、当前知识库、`status=ready` 的 chunk。
- DashScope LLM 配置有效。

请求同 FT-501。

预期结果：

- `answer` 不为空。
- `sources` 不为空。
- 回答中应按系统提示引用来源编号。

当前风险：

- 如果普通文档未完成发布到 ready，问答仍会进入无资料分支。
- 如果模型密钥不可用，生成阶段可能失败。

前端验证：

- 登录后进入“智能问答”。
- 选择测试知识库。
- 输入问题并发送。
- 检查回答和参考来源。

## 10. 论文中心测试

### FT-601 DOI 元数据导入

接口：

```text
POST /api/v1/papers/import-doi
```

请求：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/papers/import-doi `
  -Method Post `
  -Headers $headers `
  -ContentType 'application/json' `
  -Body (@{
    doi = "10.1038/s41586-023-06789-9"
    kb_id = $kbId
  } | ConvertTo-Json)
```

预期结果：

- 成功时返回 DOI、标题、作者、期刊、摘要等元数据。
- 找不到 DOI 时返回 HTTP 404。

当前风险：

- 依赖 CrossRef 网络访问。
- 前端论文中心当前没有提交 `kb_id`，前端直接测预计失败。

### FT-602 PMID 元数据导入

请求：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/papers/import-pmid `
  -Method Post `
  -Headers $headers `
  -ContentType 'application/json' `
  -Body (@{
    pmid = "12345678"
    kb_id = $kbId
  } | ConvertTo-Json)
```

预期结果：

- 成功时返回 PMID、标题、作者、期刊、摘要、MeSH terms。
- 找不到 PMID 时返回 HTTP 404。

当前风险：

- 依赖 PubMed 网络访问。
- 前端论文中心当前没有提交 `kb_id`。

### FT-603 上传论文 PDF

准备：

- 准备一份本地 PDF。
- 确保 Celery worker 已启动。
- 确保 DashScope embedding 配置可用。
- 确保 Weaviate 可写。

请求：

```powershell
$paperPdf = "D:\path\to\paper.pdf"

$paper = Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/papers/upload `
  -Method Post `
  -Headers $headers `
  -Form @{
    file = Get-Item $paperPdf
    kb_id = $kbId
  }

$paperId = $paper.paper_id
$paper
```

预期结果：

- 返回 `paper_id`、`document_id`、`ingestion_job_id`。
- 初始 `status` 可能为 `draft`。
- worker 处理完成后，论文和文档状态应变为 `ready`。

当前风险：

- 前端没有提交 `kb_id`，应优先使用 Swagger 或 curl/PowerShell 测试。
- 真实解析、embedding、Weaviate 写入依赖外部服务。

### FT-604 查询论文详情

请求：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/papers/$paperId" `
  -Method Get `
  -Headers $headers
```

预期结果：

- 返回论文标题、DOI/PMID、摘要、实体、证据字段等。

### FT-605 查询论文证据摘要

请求：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/papers/$paperId/evidence" `
  -Method Get `
  -Headers $headers
```

预期结果：

- 返回 `study_type`、`sample_size`、`pico`、`evidence_level` 等字段。
- 如果解析未提取到相关信息，字段可能为空。

### FT-606 查询参考文献

请求：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/papers/$paperId/references" `
  -Method Get `
  -Headers $headers
```

预期结果：

- 返回 `references` 和 `total`。

### FT-607 查询相似论文

请求：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/papers/$paperId/similar" `
  -Method Get `
  -Headers $headers
```

预期结果：

- 如果同组织内有状态为 `ready` 且 MeSH terms 有交集的论文，返回相似论文列表。
- 否则返回空列表。

## 11. 反馈、评测与分析测试

### FT-701 提交回答反馈

请求：

```powershell
$messageId = [guid]::NewGuid().ToString()

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/answers/$messageId/feedback" `
  -Method Post `
  -Headers $headers `
  -ContentType 'application/json' `
  -Body (@{
    message_id = $messageId
    rating = 5
    reason_tags = @()
    comment = "功能测试反馈"
  } | ConvertTo-Json)
```

预期结果：

- 返回反馈 `id`。
- `rating=5`。

当前风险：

- 前端聊天页无法稳定拿到 `message_id`，因此前端反馈按钮可能不显示。

### FT-702 创建评测集

请求：

```powershell
$evalSet = Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/evaluation-sets `
  -Method Post `
  -Headers $headers `
  -ContentType 'application/json' `
  -Body (@{
    name = "功能测试评测集"
    scenario = "qa"
    description = "用于手工功能验证"
    questions = @(
      @{
        question = "产品A是什么？"
        expected_kb_ids = @($kbId)
        expected_doc_ids = @()
        expected_answer = $null
        category = "general"
        difficulty = "easy"
      }
    )
  } | ConvertTo-Json -Depth 5)

$evalSetId = $evalSet.id
$evalSet
```

预期结果：

- 返回评测集 `id`。
- `status=active`。

前端验证：

- 进入“评测”页面。
- 新建评测集。
- 添加至少一个问题。
- 保存后左侧列表出现评测集。

### FT-703 查询评测集和问题

请求：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/evaluation-sets `
  -Method Get `
  -Headers $headers

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/evaluation-sets/$evalSetId/questions" `
  -Method Get `
  -Headers $headers
```

预期结果：

- 能查询到刚创建的评测集和问题。

### FT-704 运行评测

前置条件：

- Celery worker 已启动。
- 检索链路可用。

请求：

```powershell
$run = Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/evaluations/run `
  -Method Post `
  -Headers $headers `
  -ContentType 'application/json' `
  -Body (@{
    eval_set_id = $evalSetId
    config = @{}
  } | ConvertTo-Json)

$runId = $run.id
$run
```

查询结果：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/evaluations/$runId" `
  -Method Get `
  -Headers $headers
```

预期结果：

- 初始状态为 `pending`。
- worker 执行后应变为 `running`，最终为 `completed` 或 `failed`。
- `completed` 时 `metrics` 中包含 `total_questions`、`answered`、`zero_result`、`recall_at_10`、`zero_result_rate` 等。

### FT-705 分析总览

请求：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/analytics/summary `
  -Method Get `
  -Headers $headers
```

预期结果：

- 返回总查询数、平均延迟、零结果率、平均评分等字段。

当前风险：

- 如果 ClickHouse 中没有 `rag_trace_events` 表或没有数据，可能返回空结果或后端响应校验失败。

### FT-706 零结果查询

请求：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/analytics/zero-result-queries `
  -Method Get `
  -Headers $headers
```

预期结果：

- 返回 `items` 和 `total`。
- 如果没有数据，`items=[]`。

### FT-707 低分回答

请求：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/analytics/low-rated-answers `
  -Method Get `
  -Headers $headers
```

预期结果：

- 返回 `items` 和 `total`。
- 如果没有低分记录，`items=[]`。

### FT-708 审计日志

请求：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/audit-logs `
  -Method Get `
  -Headers $headers
```

预期结果：

- 返回 `items` 和 `total`。
- 提交反馈、运行评测后应能看到对应审计日志。

## 12. 前端回归测试清单

### FT-801 登录页

- 输入正确账号密码，应进入系统。
- 输入错误密码，应显示登录失败。
- 刷新页面后，如果 token 仍在 localStorage，应保持登录状态。

### FT-802 知识库页

- 创建知识库。
- 选择知识库。
- 上传文档。
- 查看文档列表。
- 删除文档。
- 删除知识库。

通过标准：

- 页面无白屏。
- 浏览器控制台无阻断性异常。
- 与后端数据一致。

### FT-803 聊天页

- 进入聊天页。
- 选择知识库。
- 输入问题。
- 检查回答区域是否显示结果或错误信息。
- 检查参考来源是否可展开。

当前注意：

- 如果没有 ready chunk，回答“未找到相关的参考资料”属于预期。
- 前端传给搜索接口的 `limit` 当前不会被后端识别。

### FT-804 论文中心

当前前端直接验证预计失败，因为没有提交 `kb_id`。  
前端修复前，只建议验证页面是否能打开、表单是否能输入、失败提示是否能显示。

### FT-805 分析页

- 总览页可打开。
- 零结果查询页可刷新。
- 低分回答页可刷新。
- 审计日志页可查询。

当前注意：

- 分析数据依赖 ClickHouse 表和 RAG trace 写入。

### FT-806 评测页

- 创建评测集。
- 添加问题。
- 查看问题列表。
- 点击运行评测。
- 观察状态轮询结果。

当前注意：

- 运行评测依赖 Celery worker 和检索链路。

## 13. 自动化验证命令

### 后端测试

```powershell
cd D:\code\rag
.\.venv\Scripts\python.exe -m pytest
```

本次验证结果：

```text
284 passed, 1 warning
```

### 前端构建

```powershell
cd D:\code\rag\frontend
npm run build
```

本次验证结果：

```text
vite build 成功，生成 dist/
```

### 依赖连通性检查

PostgreSQL：

```powershell
@"
import asyncio
from sqlalchemy import text
from app.database import engine

async def main():
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        print(result.scalar())

asyncio.run(main())
"@ | .\.venv\Scripts\python.exe -
```

Redis：

```powershell
@"
from app.config import settings
import redis
r = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=5, socket_timeout=5)
print(r.ping())
"@ | .\.venv\Scripts\python.exe -
```

ClickHouse：

```powershell
@"
import asyncio
import httpx
from app.services.clickhouse import clickhouse_client

async def main():
    async with httpx.AsyncClient(timeout=5, auth=clickhouse_client._auth) as client:
        resp = await client.post(clickhouse_client.url, params=clickhouse_client._params("SELECT 1 FORMAT JSON"))
        print(resp.status_code, resp.text[:100])

asyncio.run(main())
"@ | .\.venv\Scripts\python.exe -
```

Weaviate：

```powershell
@"
from app.services.weaviate_client import get_client, COLLECTION_NAME
client = get_client()
client.connect()
try:
    print(client.collections.exists(COLLECTION_NAME))
finally:
    client.close()
"@ | .\.venv\Scripts\python.exe -
```

## 14. 当前接口清单

| 方法 | 路径 |
|---|---|
| POST | `/api/v1/auth/login` |
| POST | `/api/v1/kbs` |
| GET | `/api/v1/kbs` |
| GET | `/api/v1/kbs/{kb_id}` |
| PATCH | `/api/v1/kbs/{kb_id}` |
| DELETE | `/api/v1/kbs/{kb_id}` |
| POST | `/api/v1/kbs/{kb_id}/documents` |
| GET | `/api/v1/documents/{document_id}` |
| GET | `/api/v1/kbs/{kb_id}/documents` |
| DELETE | `/api/v1/documents/{document_id}` |
| GET | `/api/v1/ingestion-jobs/{job_id}` |
| POST | `/api/v1/chat` |
| POST | `/api/v1/search` |
| POST | `/api/v1/papers/upload` |
| POST | `/api/v1/papers/import-doi` |
| POST | `/api/v1/papers/import-pmid` |
| GET | `/api/v1/papers/{paper_id}` |
| GET | `/api/v1/papers/{paper_id}/evidence` |
| GET | `/api/v1/papers/{paper_id}/references` |
| GET | `/api/v1/papers/{paper_id}/similar` |
| POST | `/api/v1/answers/{message_id}/feedback` |
| POST | `/api/v1/evaluation-sets` |
| GET | `/api/v1/evaluation-sets` |
| GET | `/api/v1/evaluation-sets/{eval_set_id}` |
| GET | `/api/v1/evaluation-sets/{eval_set_id}/questions` |
| POST | `/api/v1/evaluations/run` |
| GET | `/api/v1/evaluations/{run_id}` |
| GET | `/api/v1/analytics/zero-result-queries` |
| GET | `/api/v1/analytics/low-rated-answers` |
| GET | `/api/v1/analytics/summary` |
| GET | `/api/v1/audit-logs` |

## 15. 验收记录模板

| 编号 | 功能 | 结果 | 实际现象 | 截图/日志 | 备注 |
|---|---|---|---|---|---|
| FT-001 | 后端健康检查 | 未测 |  |  |  |
| FT-101 | 登录成功 | 未测 |  |  |  |
| FT-201 | 创建知识库 | 未测 |  |  |  |
| FT-301 | 上传文档 | 未测 |  |  |  |
| FT-401 | 检索 | 未测 |  |  |  |
| FT-501 | 问答 | 未测 |  |  |  |
| FT-601 | DOI 导入 | 未测 |  |  |  |
| FT-603 | 论文上传 | 未测 |  |  |  |
| FT-702 | 创建评测集 | 未测 |  |  |  |
| FT-704 | 运行评测 | 未测 |  |  |  |
| FT-705 | 分析总览 | 未测 |  |  |  |
| FT-708 | 审计日志 | 未测 |  |  |  |

## 16. 本回答可信度

可信度：中到高。

原因：后端测试、前端构建、API 启动、登录、核心依赖只读连通性均有当前环境实测结果支撑；但真实文件入库、Celery 异步任务、模型调用、Weaviate 写入、ClickHouse 业务表数据读取尚未逐项执行完整端到端验证，因此这些链路仍需按本手册继续验证。
