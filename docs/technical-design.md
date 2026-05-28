# 企业级生命健康领域知识库 — 技术方案设计文档

## 1. 项目概述

构建一个面向生命健康领域的企业通用知识库系统，支持多源知识摄入（文档、PDF论文、结构化数据等），基于 RAG（Retrieval-Augmented Generation）架构实现智能检索与问答。系统需满足企业级的高可用、安全可控、权限隔离、可观测性等要求。

### 1.1 核心目标

- 支持企业内部文档（Word、PDF、Markdown、Wiki 等）的知识摄入与检索
- 支持 SCI PDF 学术论文的解析、结构化提取与知识关联
- 面向生命健康领域的专业语义检索与问答
- 企业级多租户、RBAC 权限控制、审计日志
- 可扩展的架构，支持未来模型与向量数据库的替换

### 1.2 技术栈选型

| 层级 | 组件 | 选型 | 说明 |
|------|------|------|------|
| 关系存储 | RDBMS | PostgreSQL | 用户、权限、文档元数据、会话记录、审核流 |
| 列式分析 | OLAP | ClickHouse | 检索日志、用户行为分析、系统监控指标、热点分析 |
| 向量检索 | Vector DB | Weaviate | 语义向量存储与检索，支持混合检索（BM25 + Vector） |
| 大模型 | LLM | 千问（Qwen）系列 | Qwen-Max / Qwen-Plus 用于生成，Qwen-Embedding 用于向量化 |
| 应用框架 | Backend | FastAPI (Python) | 高并发异步、生态丰富、RAG 工具链完善 |
| 任务队列 | Queue | Celery + Redis | 文档解析、向量化等异步后台任务 |
| 全文检索 | Search | Weaviate 内置 BM25 | Weaviate 原生混合检索，减少组件依赖 |
| 缓存 | Cache | Redis | 查询缓存、会话缓存、限流 |
| 前端 | Frontend | Next.js + React | SSR 渲染、企业级后台管理 |

---

## 2. 系统架构

### 2.1 总体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│   Web Portal (Next.js)  │  API SDK  │  Admin Console  │  Webhook │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTPS
┌────────────────────────▼────────────────────────────────────────┐
│                     API Gateway (Nginx)                          │
│         TLS 终止 │ 限流 │ 路由 │ 负载均衡 │ WAF                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   Application Layer (FastAPI)                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐│
│  │ 知识管理  │ │ 检索问答  │ │ 用户权限  │ │ 论文管理  │ │ 系统管理││
│  │ Service  │ │ Service  │ │ Service  │ │ Service  │ │Service ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────┘│
└──┬──────────┬──────────┬──────────┬──────────┬──────────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│PostgreSQL││Weaviate││ Redis  ││ClickHouse│
│元数据/权限│ │向量/混合 │ │缓存/队列│ │分析/日志 │
└────────┘ └────────┘ └────────┘ └────────┘
                              │
                   ┌──────────▼──────────┐
                   │   Celery Workers    │
                   │ 文档解析 │ 向量化    │
                   │ PDF 提取 │ 质量校验  │
                   └─────────────────────┘
```

### 2.2 核心服务拆分

| 服务 | 职责 | 端口 |
|------|------|------|
| `api-gateway` | 反向代理、认证拦截、限流 | 443 |
| `app-server` | 业务 API、RAG 编排 | 8000 |
| `worker-parsing` | 文档解析、PDF 结构化提取 | - |
| `worker-embedding` | 文本向量化、索引构建 | - |
| `admin-server` | 后台管理、用户管理、知识库管理 | 8001 |

---

## 3. 数据模型设计

### 3.1 PostgreSQL 核心表

#### 3.1.1 多租户与组织

```sql
-- 组织/租户
CREATE TABLE organizations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        VARCHAR(255) NOT NULL,
    slug        VARCHAR(100) UNIQUE NOT NULL,
    status      VARCHAR(20) DEFAULT 'active',
    config      JSONB DEFAULT '{}',          -- 租户级配置
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 部门（组织架构树）
CREATE TABLE departments (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      UUID REFERENCES organizations(id),
    parent_id   UUID REFERENCES departments(id),
    name        VARCHAR(255) NOT NULL,
    path        VARCHAR(500),                -- 物化路径
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

#### 3.1.2 用户与权限

```sql
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      UUID REFERENCES organizations(id),
    dept_id     UUID REFERENCES departments(id),
    email       VARCHAR(255) UNIQUE NOT NULL,
    name        VARCHAR(100) NOT NULL,
    role        VARCHAR(50) DEFAULT 'viewer', -- admin/editor/viewer
    status      VARCHAR(20) DEFAULT 'active',
    password_hash VARCHAR(255),
    last_login  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 知识库（Knowledge Base）
CREATE TABLE knowledge_bases (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      UUID REFERENCES organizations(id),
    name        VARCHAR(255) NOT NULL,
    description TEXT,
    type        VARCHAR(50) DEFAULT 'general', -- general / paper / clinical / drug
    domain_tags JSONB DEFAULT '[]',            -- 领域标签：["oncology","cardiology"]
    is_public   BOOLEAN DEFAULT FALSE,
    created_by  UUID REFERENCES users(id),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- KB 访问权限
CREATE TABLE kb_permissions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kb_id       UUID REFERENCES knowledge_bases(id),
    user_id     UUID REFERENCES users(id),
    permission  VARCHAR(50) NOT NULL,  -- read / write / admin
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(kb_id, user_id)
);
```

#### 3.1.3 文档与文档版本

```sql
CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kb_id           UUID REFERENCES knowledge_bases(id),
    title           VARCHAR(500) NOT NULL,
    file_name       VARCHAR(500),
    file_type       VARCHAR(50),            -- pdf / docx / md / html / txt / sci_paper
    file_size       BIGINT,
    storage_path    VARCHAR(1000),          -- 对象存储路径
    content_hash    VARCHAR(64),            -- 文件内容hash，用于去重
    parse_status    VARCHAR(50) DEFAULT 'pending', -- pending/parsing/ready/failed
    parse_error     TEXT,
    metadata        JSONB DEFAULT '{}',     -- 领域特定元数据
    created_by      UUID REFERENCES users(id),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 文档版本（支持更新替换，保留历史）
CREATE TABLE document_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID REFERENCES documents(id),
    version         INT NOT NULL,
    content_hash    VARCHAR(64),
    storage_path    VARCHAR(1000),
    chunk_count     INT DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 文档分块（仅存元数据，实际向量在 Weaviate）
CREATE TABLE document_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID REFERENCES documents(id),
    chunk_index     INT NOT NULL,
    weaviate_id     VARCHAR(100),           -- Weaviate 中的对象ID
    content_preview TEXT,                   -- 前200字符预览
    token_count     INT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

#### 3.1.4 SCI 论文专用表

```sql
CREATE TABLE papers (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID REFERENCES documents(id),
    doi             VARCHAR(100) UNIQUE,
    pmid            VARCHAR(50),
    title           TEXT NOT NULL,
    abstract        TEXT,
    authors         JSONB,                  -- [{name, affiliation, orcid}]
    journal_name    VARCHAR(500),
    journal_issn    VARCHAR(20),
    publication_date DATE,
    impact_factor   DECIMAL(5,3),
    keywords        JSONB,                  -- MeSH terms + 自定义关键词
    medical_subject_headings JSONB DEFAULT '[]',  -- MeSH 主题词
    citation_count  INT DEFAULT 0,
    paper_type      VARCHAR(50),            -- review / clinical_trial / meta_analysis / basic_research
    fulltext_path   VARCHAR(1000),          -- 解析后的全文结构化JSON
    parse_method    VARCHAR(50),            -- grobid / pymupdf / custom
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 论文引用关系
CREATE TABLE paper_citations (
    citing_paper_id UUID REFERENCES papers(id),
    cited_paper_id  UUID REFERENCES papers(id),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (citing_paper_id, cited_paper_id)
);
```

#### 3.1.5 会话与反馈

```sql
CREATE TABLE conversations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID REFERENCES organizations(id),
    user_id         UUID REFERENCES users(id),
    kb_ids          UUID[],                 -- 关联的知识库
    title           VARCHAR(500),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id),
    role            VARCHAR(20),            -- user / assistant / system
    content         TEXT,
    sources         JSONB DEFAULT '[]',     -- 引用的文档片段 [{doc_id, chunk_id, score}]
    model_used      VARCHAR(50),
    token_usage     JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 用户反馈（RLHF 数据积累）
CREATE TABLE feedback (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id      UUID REFERENCES messages(id),
    rating          INT,                    -- 1-5
    comment         TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

### 3.2 Weaviate Schema 设计

```yaml
# Weaviate Collection: knowledge_chunks
classes:
  - name: KnowledgeChunk
    vectorizer: none  # 使用外部 Qwen-Embedding
    properties:
      - name: document_id
        dataType: [string]
        indexFilterable: true
      - name: kb_id
        dataType: [string]
        indexFilterable: true
      - name: org_id
        dataType: [string]
        indexFilterable: true
      - name: content
        dataType: [text]
        indexFullText: true   # 支持 BM25
      - name: chunk_index
        dataType: [int]
      - name: token_count
        dataType: [int]
      - name: document_type
        dataType: [string]    # paper / doc / wiki / guideline
      - name: domain_tags
        dataType: [string[]]  # 领域标签
      - name: title
        dataType: [text]
        indexFullText: true
      - name: metadata
        dataType: [object]    # 扩展元数据（论文作者、期刊等）
      - name: created_at
        dataType: [date]
    moduleConfig:
      text2vec-transformers:
        vectorizeClassName: false
        vectorizePropertyName: false
```

---

## 4. RAG Pipeline 设计

### 4.1 文档摄入流程

```
上传/导入 ──► 去重检查 ──► 格式识别 ──► 异步解析 ──► 文本分块 ──► 向量化 ──► 入库
              (hash)              │                      │
                                  ▼                      ▼
                            普通文档                SCI PDF 特殊处理
                            (docx/md/html)          (GROBID 结构化提取)
```

### 4.2 文档解析策略

| 文件类型 | 解析工具 | 特殊处理 |
|----------|----------|----------|
| PDF (普通) | PyMuPDF + Layout Parser | 保留表格、图片OCR |
| PDF (SCI论文) | GROBID + 自定义后处理 | 提取标题、摘要、章节、参考文献、MeSH词 |
| DOCX | python-docx | 保留标题层级、表格 |
| Markdown | 原生解析 | 直接解析 |
| HTML | BeautifulSoup | 去除噪声、提取正文 |
| Excel/CSV | pandas | 结构化数据特殊处理 |
| TXT | 原生读取 | 编码自动检测 |

### 4.3 SCI PDF 论文解析 Pipeline

```
PDF 文件 ──► GROBID 解析 ──► 结构化 TEI XML ──► 字段提取
    │                                                │
    ├── 标题、作者、摘要、期刊信息                     │
    ├── 章节结构 (Introduction/Methods/Results...)    │
    ├── 参考文献列表 ──► DOI 解析 ──► 引用关系构建       │
    ├── 图表数据 ──► OCR/表格提取                      │
    └── 全文文本 ──► 智能分块 ──► 向量化               │

# 论文级元数据增强
├── CrossRef API 补充元数据
├── PubMed API 匹配 MeSH 主题词
├── 影响因子数据更新
└── 相似文献关联（向量相似度）
```

### 4.4 智能分块策略

针对生命健康领域的特殊性，采用分层分块策略：

```python
# 分块策略配置
CHUNK_CONFIG = {
    "general": {
        "chunk_size": 512,       # token 数
        "chunk_overlap": 50,     # 重叠
        "strategy": "semantic",  # 语义边界分块
    },
    "paper": {
        "strategy": "section_based",  # 按论文章节分块
        "max_chunk_size": 1024,
        "preserve_structure": True,   # 保留章节层级
        "abstract_independent": True, # 摘要独立成块
        "table_independent": True,    # 表格独立成块
    },
    "clinical_guideline": {
        "strategy": "hierarchical",   # 按指南层级分块
        "preserve_recommendations": True,  # 推荐意见独立
    }
}
```

### 4.5 检索策略

```
用户查询 ──► 查询重写 ──► 混合检索 ──► 重排序 ──► 上下文组装 ──► LLM 生成
                │            │            │
                ▼            ▼            ▼
         拼写纠正        Weaviate      Cross-Encoder
         领域扩展         BM25 +       重排序模型
         同义扩展         Vector       (BGE-Reranker)
         多语言归一化      过滤:KB/权限/领域
```

#### 4.5.1 检索流程详细设计

```python
class RagRetrievalService:
    async def retrieve(self, query: str, kb_ids: list, user_id: str, top_k: int = 20):
        # Step 1: 查询增强
        enhanced_queries = await self.query_rewrite(query)

        # Step 2: 混合检索（多查询并行）
        results = await asyncio.gather(*[
            self.hybrid_search(q, kb_ids, user_id, top_k)
            for q in enhanced_queries
        ])

        # Step 3: 去重合并
        merged = self.deduplicate_and_merge(results)

        # Step 4: Cross-Encoder 重排序
        reranked = await self.rerank(query, merged, top_k=10)

        # Step 5: 权限过滤
        filtered = await self.apply_permission_filter(reranked, user_id)

        return filtered[:top_k]
```

#### 4.5.2 领域特定优化

- **医学术语标准化**: 查询词映射到 MeSH / SNOMED CT / ICD-10 标准术语
- **缩写扩展**: 如 "MI" → "Myocardial Infarction / 心肌梗死"
- **药物名称归一化**: 商品名 ↔ 通用名 ↔ 化学名
- **多语言支持**: 中文查询 ↔ 英文文献的跨语言检索

### 4.6 生成策略

```python
SYSTEM_PROMPT_TEMPLATE = """
你是一个生命健康领域的专业知识助手。请基于以下参考资料回答问题。

规则：
1. 优先引用提供的参考资料，并在回答中标注来源编号 [1][2]...
2. 如果参考资料不足以回答问题，明确告知用户
3. 涉及医疗建议时，必须提醒用户咨询专业医疗人员
4. 涉及药物信息时，注明信息仅供参考，需以最新药品说明书为准
5. 区分事实性陈述与推理性建议

参考资料：
{context}

历史对话：
{history}
"""
```

---

## 5. 企业级特性

### 5.1 多租户架构

- **数据隔离**: 所有数据表通过 `org_id` 实现租户级隔离
- **Weaviate 过滤**: 所有向量检索强制附加 `org_id` 过滤条件
- **资源配额**: 租户级文档数量、存储空间、API 调用量限制

### 5.2 RBAC 权限模型

```
角色层级：
  Organization Admin ── 管理租户全部设置
      │
      ├── KB Admin ── 管理特定知识库（成员、设置）
      │
      ├── Editor ── 上传/编辑/删除文档
      │
      └── Viewer ── 检索与问答（只读）

权限校验流程：
  每次请求 ──► JWT 解析 ──► org_id + user_id + role
           ──► 检查目标资源的权限矩阵
           ──► 通过/拒绝（记录审计日志）
```

### 5.3 审计日志

```sql
CREATE TABLE audit_logs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      UUID,
    user_id     UUID,
    action      VARCHAR(100),       -- doc.upload / doc.delete / kb.query / login
    resource_type VARCHAR(50),
    resource_id UUID,
    ip_address  INET,
    user_agent  TEXT,
    detail      JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
-- 审计日志写入 ClickHouse 用于分析，PostgreSQL 保留最近90天热数据
```

### 5.4 内容安全与合规

- **敏感信息检测**: 上传文档中的个人信息（PII）自动检测与脱敏
- **内容审核**: 基于 LLM 的内容合规性检查
- **数据导出控制**: 限制批量导出，记录所有导出操作
- **HIPAA 兼容考虑**: 如涉及临床患者数据，需符合 HIPAA 要求
- **数据留存策略**: 可配置的数据保留与自动清理策略

---

## 6. ClickHouse 分析设计

### 6.1 分析场景

| 场景 | 表 | 说明 |
|------|-----|------|
| 检索分析 | `search_events` | 查询词、结果数量、点击率、零结果查询 |
| 用户行为 | `user_activities` | 登录频次、活跃KB、使用时长 |
| 文档热度 | `document_access` | 被引用最多的文档、过期文档识别 |
| 系统性能 | `system_metrics` | API 延迟、Token 消耗、向量化耗时 |
| 质量反馈 | `feedback_analytics` | 回答满意度、低分问答归因 |

### 6.2 核心分析表

```sql
-- 检索事件表
CREATE TABLE search_events (
    event_time      DateTime,
    org_id          UUID,
    user_id         UUID,
    query_text      String,
    kb_ids          Array(UUID),
    result_count    UInt16,
    latency_ms      UInt32,
    has_click       Bool,
    clicked_doc_id  Nullable(UUID),
    model           String
) ENGINE = MergeTree()
ORDER BY (org_id, event_time);

-- 问答事件表
CREATE TABLE qa_events (
    event_time      DateTime,
    org_id          UUID,
    user_id         UUID,
    query_text      String,
    response_length UInt32,
    token_usage     UInt32,
    latency_ms      UInt32,
    source_count    UInt8,
    rating          Nullable(UInt8),
    model           String
) ENGINE = MergeTree()
ORDER BY (org_id, event_time);
```

### 6.3 典型分析查询

```sql
-- 热门查询词（用于优化索引和预计算）
SELECT query_text, count() as cnt, avg(result_count) as avg_results
FROM search_events
WHERE org_id = ? AND event_time > now() - INTERVAL 7 DAY
GROUP BY query_text
ORDER BY cnt DESC LIMIT 50;

-- 零结果查询（需要补充知识的内容缺口）
SELECT query_text, count() as cnt
FROM search_events
WHERE result_count = 0 AND event_time > now() - INTERVAL 7 DAY
GROUP BY query_text
ORDER BY cnt DESC LIMIT 20;

-- Token 消耗分析（成本控制）
SELECT toDate(event_time) as day, sum(token_usage) as total_tokens,
       avg(latency_ms) as avg_latency
FROM qa_events
WHERE org_id = ?
GROUP BY day
ORDER BY day DESC LIMIT 30;
```

---

## 7. 项目目录结构

```
rag-knowledge-base/
├── docs/                          # 项目文档
│   └── technical-design.md        # 本文件
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI 入口
│   │   ├── config.py              # 配置管理
│   │   ├── dependencies.py        # 依赖注入
│   │   ├── api/                   # API 路由
│   │   │   ├── __init__.py
│   │   │   ├── v1/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── auth.py        # 认证/登录
│   │   │   │   ├── users.py       # 用户管理
│   │   │   │   ├── organizations.py
│   │   │   │   ├── knowledge_bases.py
│   │   │   │   ├── documents.py    # 文档CRUD
│   │   │   │   ├── papers.py       # 论文管理
│   │   │   │   ├── search.py       # 检索/问答
│   │   │   │   ├── conversations.py
│   │   │   │   ├── analytics.py    # 数据分析
│   │   │   │   └── admin.py        # 系统管理
│   │   │   └── health.py           # 健康检查
│   │   ├── core/                  # 核心组件
│   │   │   ├── security.py        # JWT/权限校验
│   │   │   ├── exceptions.py      # 全局异常处理
│   │   │   └── middleware.py       # 中间件
│   │   ├── services/              # 业务服务层
│   │   │   ├── rag_service.py     # RAG 核心编排
│   │   │   ├── retrieval_service.py
│   │   │   ├── generation_service.py
│   │   │   ├── document_service.py
│   │   │   ├── paper_service.py
│   │   │   ├── embedding_service.py
│   │   │   ├── query_rewrite_service.py
│   │   │   └── analytics_service.py
│   │   ├── parsers/               # 文档解析器
│   │   │   ├── base.py            # 解析器基类
│   │   │   ├── pdf_parser.py      # PDF 解析
│   │   │   ├── grobid_parser.py   # GROBID SCI论文解析
│   │   │   ├── docx_parser.py
│   │   │   ├── html_parser.py
│   │   │   └── table_extractor.py # 表格提取
│   │   ├── chunkers/              # 分块策略
│   │   │   ├── base.py
│   │   │   ├── semantic_chunker.py
│   │   │   ├── section_chunker.py
│   │   │   └── paper_chunker.py   # 论文专用分块
│   │   ├── models/                # SQLAlchemy 数据模型
│   │   │   ├── __init__.py
│   │   │   ├── organization.py
│   │   │   ├── user.py
│   │   │   ├── knowledge_base.py
│   │   │   ├── document.py
│   │   │   ├── paper.py
│   │   │   ├── conversation.py
│   │   │   └── audit.py
│   │   ├── schemas/               # Pydantic 数据校验
│   │   │   ├── user.py
│   │   │   ├── document.py
│   │   │   ├── paper.py
│   │   │   ├── search.py
│   │   │   └── conversation.py
│   │   └── clients/               # 外部服务客户端
│   │       ├── weaviate_client.py
│   │       ├── clickhouse_client.py
│   │       ├── qwen_client.py     # 千问 API 封装
│   │       ├── crossref_client.py # CrossRef API
│   │       └── pubmed_client.py   # PubMed API
│   ├── workers/                   # Celery 任务
│   │   ├── celery_app.py
│   │   ├── parsing_tasks.py       # 文档解析任务
│   │   ├── embedding_tasks.py     # 向量化任务
│   │   └── maintenance_tasks.py   # 定时维护任务
│   ├── migrations/                # Alembic 数据库迁移
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── fixtures/
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── requirements.txt
├── frontend/
│   ├── app/                       # Next.js App Router
│   ├── components/
│   ├── lib/
│   └── package.json
├── deploy/
│   ├── docker-compose.yml         # 本地开发
│   ├── docker-compose.prod.yml    # 生产部署
│   ├── k8s/                       # Kubernetes 部署
│   │   ├── namespace.yaml
│   │   ├── api-deployment.yaml
│   │   ├── worker-deployment.yaml
│   │   ├── postgres-statefulset.yaml
│   │   ├── weaviate-statefulset.yaml
│   │   ├── clickhouse-statefulset.yaml
│   │   ├── redis-statefulset.yaml
│   │   └── ingress.yaml
│   └── nginx/
│       └── nginx.conf
└── scripts/
    ├── init_db.sql                # 数据库初始化
    ├── seed_data.py               # 测试数据填充
    └── backup.sh                  # 备份脚本
```

---

## 8. API 设计（核心端点）

### 8.1 认证

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/auth/login` | 登录（返回 JWT） |
| POST | `/api/v1/auth/refresh` | 刷新 Token |
| POST | `/api/v1/auth/logout` | 登出 |

### 8.2 知识库管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/knowledge-bases` | 列出知识库 |
| POST | `/api/v1/knowledge-bases` | 创建知识库 |
| GET | `/api/v1/knowledge-bases/{id}` | 获取详情 |
| PUT | `/api/v1/knowledge-bases/{id}` | 更新 |
| DELETE | `/api/v1/knowledge-bases/{id}` | 删除 |

### 8.3 文档管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/knowledge-bases/{kb_id}/documents` | 上传文档 |
| GET | `/api/v1/knowledge-bases/{kb_id}/documents` | 列出文档 |
| GET | `/api/v1/documents/{id}` | 获取详情 |
| DELETE | `/api/v1/documents/{id}` | 删除 |
| GET | `/api/v1/documents/{id}/status` | 解析状态轮询 |

### 8.4 论文管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/papers/import-doi` | 通过 DOI 导入论文 |
| POST | `/api/v1/papers/import-pmid` | 通过 PMID 导入论文 |
| POST | `/api/v1/papers/upload` | 上传 SCI PDF |
| GET | `/api/v1/papers/{id}` | 获取论文详情（含结构化数据） |
| GET | `/api/v1/papers/{id}/references` | 获取参考文献 |
| GET | `/api/v1/papers/{id}/citations` | 获取被引情况 |

### 8.5 检索与问答

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/search` | 语义检索（返回文档片段） |
| POST | `/api/v1/chat` | 对话式问答（RAG 生成） |
| POST | `/api/v1/chat/{conversation_id}` | 继续对话 |
| GET | `/api/v1/conversations` | 历史会话列表 |
| POST | `/api/v1/feedback` | 回答反馈 |

---

## 9. 部署架构

### 9.1 开发环境

```yaml
# docker-compose.yml 服务清单
services:
  postgres:      # PostgreSQL 16
  weaviate:      # Weaviate 最新版
  clickhouse:    # ClickHouse
  redis:         # Redis 7
  app:           # FastAPI 应用
  worker:        # Celery Worker
  grobid:        # GROBID 服务 (Docker)
  frontend:      # Next.js
```

### 9.2 生产环境（Kubernetes）

```
┌───────────────────────────────────────────────────┐
│                    Kubernetes Cluster               │
│                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │  App Pod ×3  │  │ Worker Pod×2│  │ Frontend×2 │ │
│  │  (HPA)       │  │ (HPA)       │  │            │ │
│  └──────┬───────┘  └──────┬──────┘  └─────┬──────┘ │
│         │                 │               │        │
│  ┌──────▼─────────────────▼───────────────▼──────┐ │
│  │              Ingress (Nginx)                   │ │
│  └────────────────────┬──────────────────────────┘ │
│                       │                             │
│  ┌────────────────────▼──────────────────────────┐ │
│  │           StatefulSets (持久化)                │ │
│  │  PostgreSQL × 2 (主从)                         │ │
│  │  Weaviate × 3 (集群)                           │ │
│  │  ClickHouse × 3 (分片+副本)                    │ │
│  │  Redis × 3 (Sentinel)                          │ │
│  └───────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────┘
```

### 9.3 存储规划

| 组件 | 存储类型 | 预估容量 | 说明 |
|------|----------|----------|------|
| PostgreSQL | SSD (gp3) | 500GB 起 | 元数据，增长较慢 |
| Weaviate | SSD (gp3) | 1TB 起 | 向量索引，与文档量正比 |
| ClickHouse | HDD (st1) | 2TB 起 | 分析数据，可压缩 |
| Redis | SSD | 16GB | 纯缓存 |
| 对象存储(OSS) | 标准存储 | 按需 | 原始文档与解析结果 |

---

## 10. 性能与可扩展性

### 10.1 性能目标

| 指标 | 目标 |
|------|------|
| 检索 P99 延迟 | < 500ms |
| 问答 P99 延迟 | < 5s（不含流式首字时间） |
| 首 Token 响应 | < 2s |
| 文档解析吞吐量 | > 100 文档/分钟 |
| 向量化吞吐量 | > 500 chunks/秒 |
| 并发问答用户 | > 500 |
| 系统可用性 | 99.9% |

### 10.2 扩展策略

- **水平扩展**: API 服务与 Worker 无状态，可直接 HPA
- **向量库扩展**: Weaviate 支持水平分片扩展
- **ClickHouse**: 支持分片与副本扩展
- **PostgreSQL**: 读写分离，大表分区
- **Embedding 优化**: 批量向量化、向量缓存、增量更新

---

## 11. 可观测性

### 11.1 监控体系

| 层级 | 工具 | 监控内容 |
|------|------|----------|
| 基础设施 | Prometheus + Grafana | CPU/内存/磁盘/网络 |
| 应用 | OpenTelemetry | 请求链路、延迟、错误率 |
| 业务 | ClickHouse Dashboard | 检索量、活跃度、Token消耗 |
| 日志 | Loki / ELK | 结构化日志 |
| 告警 | AlertManager | 延迟告警、错误率、资源阈值 |

### 11.2 关键业务指标

- 日活跃用户数 / 日检索次数
- 零结果查询率（知识缺口指标）
- 回答满意度评分
- 平均 Token 消耗成本
- 文档解析成功率
- 向量索引构建延迟

---

## 12. 实施路线图

### Phase 1: 基础设施与核心链路（4-6周）

- [ ] 搭建 PostgreSQL / Weaviate / Redis 基础环境
- [ ] 实现文档上传 → 解析 → 分块 → 向量化 → 入库完整链路
- [ ] 实现基础语义检索 API
- [ ] 实现基础 RAG 问答 API（接入千问）
- [ ] 用户认证与 RBAC 基础

### Phase 2: 论文支持与管理后台（3-4周）

- [ ] 集成 GROBID，实现 SCI PDF 结构化解析
- [ ] 论文元数据管理（DOI/PMID/CrossRef/PubMed）
- [ ] 论文引用关系构建
- [ ] 论文专用分块与检索优化
- [ ] 管理后台前端

### Phase 3: 企业级特性（3-4周）

- [ ] 多租户完善
- [ ] 审计日志
- [ ] ClickHouse 分析体系
- [ ] 查询重写与领域优化（MeSH/术语）
- [ ] Cross-Encoder 重排序
- [ ] 内容安全与合规

### Phase 4: 性能优化与生产就绪（2-3周）

- [ ] 性能压测与优化
- [ ] Kubernetes 部署
- [ ] 监控告警体系
- [ ] 备份与容灾
- [ ] 文档与培训
