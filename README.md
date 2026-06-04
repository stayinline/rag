# RAG Knowledge Base

企业级 RAG 知识库项目。当前仓库包含 FastAPI 后端、Celery 异步任务、React + Vite 前端，并通过项目根目录的 `config.py` 读取 PostgreSQL、Redis、Weaviate、ClickHouse 等依赖组件连接信息。

## 项目功能

当前代码中可以确认的主要模块：

- 知识库管理：创建、查询、更新、删除知识库。
- 文档管理：向知识库上传文档、查询文档、删除文档，并创建解析任务。
- RAG 检索与问答：提供 `/api/v1/search` 和 `/api/v1/chat` 接口。
- 论文处理：支持上传论文 PDF，以及通过 DOI、PMID 获取论文元数据。
- 质量与分析：包含反馈、评测集、评测运行、分析摘要、审计日志相关接口。
- 前端页面：包含聊天、知识库、论文、分析、评测、登录等页面入口。

## 技术栈

- 后端：Python 3.11+、FastAPI、SQLAlchemy、Alembic
- 异步任务：Celery、Redis
- 数据库：PostgreSQL
- 向量检索：Weaviate
- 分析存储：ClickHouse
- 前端：React、Vite、Tailwind CSS、Axios
- 模型服务：代码配置中使用 DashScope/Qwen 相关模型参数

## 目录结构

```text
.
├── app/                 # FastAPI 应用、API 路由、模型、服务、Celery worker
├── frontend/            # React + Vite 前端
├── scripts/             # 初始化数据库、初始化 Weaviate 的脚本
├── migrations/          # Alembic 迁移配置
├── tests/               # 测试用例
├── data/                # 本地文件存储目录
├── docker-compose.yml   # 历史本地依赖编排配置，当前启动说明不再依赖它
├── pyproject.toml       # Python 项目配置
├── Makefile             # 常用开发命令
└── .env.example         # 环境变量示例
```

## 启动前准备

需要先准备：

- Python 3.11 或更高版本
- Docker 与 Docker Compose
- Node.js 与 npm

依赖组件连接信息从根目录 `config.py` 读取。当前启动方式不要求先通过 `docker compose up -d` 启动本地 PostgreSQL、Redis、Weaviate、ClickHouse。

如果需要用环境变量覆盖 `config.py` 中的配置，可以再创建 `.env`；否则不要从 `.env.example` 复制本地 Docker 默认连接，以免覆盖项目级配置。

## 启动后端

1. 安装 Python 依赖：

```powershell
pip install -e ".[dev]"
```

也可以使用 Makefile：

```powershell
make install
```

2. 初始化数据库表：

```powershell
python scripts/init_db.py
```

或：

```powershell
make init-db
```

3. 初始化 Weaviate Collection：

```powershell
python scripts/init_weaviate.py
```

或：

```powershell
make init-weaviate
```

4. 启动 API 服务：

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

或：

```powershell
make start
```

启动后可访问：

- 健康检查：`http://localhost:8000/health`
- Swagger 文档：`http://localhost:8000/docs`
- ReDoc 文档：`http://localhost:8000/redoc`

## 启动异步任务 Worker

文档解析、论文解析、评测等任务依赖 Celery worker：

```powershell
celery -A app.workers.celery_app worker --loglevel=info
```

或：

```powershell
make worker
```

## 启动前端

```powershell
cd frontend
npm install
npm run dev
```

前端开发服务默认运行在：

```text
http://localhost:3000
```

Vite 配置会把 `/api` 请求代理到：

```text
http://localhost:8000
```

## 基本使用方式

后端接口统一使用 `/api/v1` 前缀。常见流程如下：

1. 创建或选择知识库。
2. 上传文档到知识库。
3. 启动 Celery worker 后，等待文档解析与入库任务执行。
4. 通过搜索接口检索知识片段。
5. 通过聊天接口进行 RAG 问答。
6. 在分析与评测模块中查看反馈、低分回答、零结果查询、评测运行等数据。

可通过 Swagger 查看当前后端实际暴露的接口：

```text
http://localhost:8000/docs
```

## 当前需要注意的限制

- 当前代码中前端会调用 `/api/v1/auth/login`，但后端路由注册文件中没有看到对应登录接口；因此前端登录流程可能无法直接完成。
- 仓库中没有看到可确认的初始化用户或种子账号脚本；需要补充用户初始化方式后，受鉴权保护的接口才能正常使用。
- 文档上传接口只创建 `parse_document` 任务；当前代码中没有确认普通文档上传后会自动串联执行 `chunk_and_embed` 与 `publish_document`。
- `config.py` 里的连接信息是当前依赖组件来源；如果 `.env` 中存在同名变量，Pydantic Settings 会按环境变量覆盖项目级默认值。
- `docs/technical-design*.md` 文件在当前读取结果中存在编码异常，本 README 未基于这些文件补充未能验证的说明。

## 常用开发命令

```powershell
make install          # 安装 Python 开发依赖
make init-db          # 创建数据库表
make init-weaviate    # 初始化 Weaviate Collection
make start            # 启动 FastAPI
make worker           # 启动 Celery worker
make frontend-install # 安装前端依赖
make frontend-dev     # 启动前端开发服务
make frontend-build   # 构建前端
```

## 运行测试

```powershell
pytest
```
