# FinInsRAG-V3

> 面向金融保险领域的检索增强生成（RAG）智能问答系统

基于 RAG 技术栈构建的全栈智能问答应用，支持多格式文档上传、混合检索、Rerank 精排、大模型流式生成，提供完整的 Web 交互界面。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| **后端框架** | FastAPI + Uvicorn |
| **RAG 引擎** | 自研模块化流水线（解析→分块→向量化→存储→检索→生成） |
| **向量检索** | 内存向量存储 + HNSW 思路近似检索，支持 BM25+向量混合检索 |
| **Embedding** | 阿里云 text-embedding-v3（1024 维） |
| **Rerank** | 阿里云 gte-rerank（cross-encoder 精排） |
| **大模型** | DeepSeek-R1（流式生成，支持思考过程） |
| **文档解析** | pdfplumber / python-docx / openpyxl（PDF/DOCX/XLSX/TXT/MD） |
| **前端** | React 18 + TypeScript + Vite + Ant Design 5 |
| **状态管理** | Valtio |
| **认证** | JWT + bcrypt |
| **流式传输** | SSE (Server-Sent Events) |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (React + TS)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  登录注册  │  │  知识库管理 │  │  对话聊天  │  │  引用溯源  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘  │
└───────┼──────────────┼──────────────┼────────────────────────┘
        │              │              │ SSE 流式
        ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                    后端 (FastAPI)                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    API 路由层                              │ │
│  │  /auth/*   /documents/*   /chat/*   /sessions/*        │ │
│  └────────────────────────┬────────────────────────────────┘ │
│                           │                                    │
│  ┌────────────────────────▼────────────────────────────────┐ │
│  │                 RAG 流水线 (Pipeline)                     │ │
│  │                                                          │ │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐           │ │
│  │  │ 文档解析  │──▶│ 文本分块  │──▶│ 向量化   │           │ │
│  │  │ Parser   │   │ Chunker  │   │ Embedding│           │ │
│  │  └──────────┘   └──────────┘   └────┬─────┘           │ │
│  │                                      │                    │ │
│  │  ┌──────────┐   ┌──────────┐   ┌───▼──────┐           │ │
│  │  │ LLM 生成  │◀──│ Prompt构建│◀──│ 向量存储  │           │ │
│  │  │ Generator│   │ Builder  │   │ VectorStore│          │ │
│  │  └────┬─────┘   └──────────┘   └───▲──────┘           │ │
│  │       │                              │                    │ │
│  │       │                    ┌─────────┴──────────┐        │ │
│  │       │                    │   检索器 Retriever    │        │ │
│  │       │                    │  ┌────────────────┐  │        │ │
│  │       │                    │  │ 混合检索(BM25+向量)│ │        │ │
│  │       │                    │  │ Rerank 精排      │  │        │ │
│  │       │                    │  │ 降级重试策略      │  │        │ │
│  │       │                    │  └────────────────┘  │        │ │
│  │       │                    └──────────────────────┘        │ │
│  └───────┼───────────────────────────────────────────────────┘ │
└──────────┼──────────────────────────────────────────────────────┘
           │
           ▼
   ┌───────────────┐
   │  大模型 API     │
   │  Embedding / Rerank / LLM
   └───────────────┘
```

---

## 核心亮点

### 1. 完整的 RAG 流水线引擎
自研六模块 RAG 流水线，而非直接调用 LangChain 等框架的封装，每个环节可独立调优：
- **文档解析**：支持 PDF/DOCX/XLSX/TXT/MD，Excel 自动转 Markdown 表格
- **智能分块**：naive_merge 算法，段落优先、句子兜底、相邻块带重叠，避免语义断裂
- **向量化**：批量调用 text-embedding-v3，1024 维语义向量
- **混合检索**：BM25 关键词检索 + 向量语义检索，0.4:0.6 加权融合，兼顾精确术语和语义理解
- **Rerank 精排**：三级检索架构（粗排→精排→综合打分），cross-encoder 模型提升排序精度
- **降级策略**：检索结果不足时自动放宽权重重试

### 2. 流式生成 + 思考过程展示
基于 SSE 的流式输出，支持 DeepSeek-R1 的思考过程（reasoning_content）和正式回答分开展示，用户可实时看到 AI 的"思考"过程。

### 3. 引用溯源
回答中每块内容标注引用编号（`##1$$`），前端可点击跳转到对应原文片段，支持答案可追溯、可验证。

### 4. 工程化全栈实现
- 前后端分离，RESTful API 设计
- JWT 认证 + bcrypt 密码哈希
- 会话管理 + 消息历史持久化
- Docker 一键部署
- 完整的项目文档和示例数据

---

## 项目结构

```
FinInsRAG-V3/
├── backend/                    # 后端服务
│   ├── app/
│   │   ├── api/                # API 路由层
│   │   │   ├── auth.py         # 认证接口（注册/登录）
│   │   │   ├── chat.py         # 对话接口（SSE 流式/会话管理）
│   │   │   └── documents.py    # 文档接口（上传/列表/删除）
│   │   ├── core/               # 核心配置
│   │   │   ├── config.py       # 环境变量配置
│   │   │   └── security.py     # JWT + bcrypt 安全
│   │   ├── rag/                # RAG 引擎核心
│   │   │   ├── parser.py       # 文档解析（PDF/DOCX/XLSX/TXT）
│   │   │   ├── chunker.py      # 文本分块（naive_merge 算法）
│   │   │   ├── embedding.py    # 向量化（text-embedding-v3）
│   │   │   ├── vectorstore.py  # 向量存储（混合检索）
│   │   │   ├── retriever.py    # 检索器（三级检索+Rerank）
│   │   │   ├── generator.py    # LLM 生成（Prompt构建+流式）
│   │   │   └── pipeline.py     # RAG 流水线编排
│   │   ├── models/             # Pydantic 数据模型
│   │   └── main.py             # 应用入口
│   ├── tests/                  # 测试
│   └── requirements.txt        # Python 依赖
├── frontend/                   # 前端应用
│   ├── src/
│   │   ├── api/                # API 封装
│   │   ├── pages/              # 页面（登录/聊天/知识库）
│   │   ├── components/         # 通用组件
│   │   ├── store/              # 状态管理（Valtio）
│   │   └── router/             # 路由
│   └── package.json
├── data/
│   └── examples/               # 示例文档
├── docs/                       # 项目文档
├── .env.example                # 环境变量模板
├── docker-compose.yml          # Docker 编排
└── README.md
```

---

## 快速开始

### 环境要求
- Python 3.10+
- Node.js 18+
- 阿里云 DashScope API Key（[获取地址](https://bailian.console.aliyun.com)）

### 1. 克隆项目
```bash
git clone <repo-url>
cd FinInsRAG-V3
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 3. 启动后端
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
后端 API 文档：http://localhost:8000/docs

### 4. 启动前端
```bash
cd frontend
npm install
npm run dev
```
前端访问：http://localhost:5173

### 5. Docker 一键启动（可选）
```bash
docker-compose up -d
```

---

## 使用流程

1. **注册账号** → 登录系统
2. **上传文档** → 进入知识库页面，上传 PDF/Word/Excel/TXT 文档
3. **开始对话** → 创建会话，基于上传的文档提问
4. **查看引用** → 点击回答中的引用编号，查看原文来源

---

## RAG 核心流程说明

### 离线：文档摄入
```
文档上传 → 格式解析 → 文本清洗 → naive_merge 分块 → jieba 分词
     → text-embedding-v3 向量化 → 存入向量存储（带元数据）
```

### 在线：问答生成
```
用户提问 → 查询预处理（去停用词/分词/同义词扩展）
     → 向量化 → 混合检索（BM25 + 向量，加权融合）
     → Rerank 精排（cross-encoder）→ Top-K 筛选
     → Prompt 构建（系统指令 + 参考文档 + 用户问题）
     → DeepSeek-R1 流式生成 → SSE 推送到前端
     → 生成推荐问题 → 保存对话记录
```

---

## API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/register` | 用户注册 |
| POST | `/api/login` | 用户登录（返回 JWT） |
| GET | `/api/me` | 当前用户信息 |
| GET | `/api/get_files` | 知识库文档列表 |
| POST | `/api/upload_files` | 上传并解析文档 |
| DELETE | `/api/delete_file/{name}` | 删除文档 |
| GET | `/api/get_sessions` | 会话列表 |
| POST | `/api/create_session` | 创建会话 |
| GET | `/api/get_messages` | 会话消息记录 |
| POST | `/api/chat_on_docs` | 基于文档的流式对话（SSE） |
| POST | `/api/quick_parse` | 快速解析文件 |

---

## 许可证

MIT License
