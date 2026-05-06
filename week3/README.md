# Week 3: RAG 基础与应用（增强版）

## 学习目标

1. 理解向量数据库原理（内存实现 + FAISS）
2. 掌握 Embedding API 的使用
3. 实现 RAG 检索增强生成
4. 理解文档分割（Chunking）对检索质量的影响
5. 构建支持文件上传的完整知识库问答系统

## 文件说明

```
week3/
├── 01_simple_vector_store.py     # Step 1: 理解向量存储原理（内存版）
├── 02_embedding_api.py           # Step 2: 使用 Embedding API
├── 03_rag_demo.py                # Step 3: 完整 RAG 演示（教学版）
├── 04_faiss_vector_store.py      # Step 4: FAISS 向量数据库
├── 05_document_chunking.py       # Step 5: 文档分割（Chunking）
├── main.py                        # 基础版: 知识库问答 API 服务
├── main_enhanced.py               # 增强版: FAISS + 文档分割 + 文件上传
├── test_api.py                    # 基础版 API 测试脚本
├── test_api_enhanced.py           # 增强版 API 测试脚本
└── README.md                      # 本文件
```

## 学习步骤

### Step 1: 理解向量存储原理

```bash
python week3/01_simple_vector_store.py
```

学习内容：
- 什么是向量（Embedding）
- 相似度计算原理
- 简单向量数据库实现

### Step 2: 使用 Embedding API

```bash
python week3/02_embedding_api.py
```

学习内容：
- 调用智谱 AI Embedding API
- 实现语义搜索
- 理解向量维度含义（768维）

### Step 3: 完整 RAG 演示

```bash
python week3/03_rag_demo.py
```

学习内容：
- RAG 完整流程
- 文档检索 + 大模型生成
- 实战：产品知识库问答

### Step 4: FAISS 向量数据库

```bash
python week3/04_faiss_vector_store.py
```

学习内容：
- FAISS 索引创建与检索
- 向量归一化与余弦相似度
- 索引持久化（保存/加载）
- 与暴力搜索的性能对比

### Step 5: 文档分割（Chunking）

```bash
python week3/05_document_chunking.py
```

学习内容：
- 为什么需要文档分割
- CharacterTextSplitter vs RecursiveCharacterTextSplitter
- chunk_size 和 chunk_overlap 的影响
- 分割策略对检索效果的影响

### Step 6: 增强版知识库问答 API

```bash
# 启动增强版服务
python week3/main_enhanced.py

# 测试 API（另开终端）
python week3/test_api_enhanced.py
```

增强功能：
- **FAISS 向量检索**：支持大规模数据高效检索
- **自动文档分割**：RecursiveCharacterTextSplitter，chunk_size=500, overlap=100
- **文件上传**：支持 PDF、TXT 文件直接上传解析
- **向量索引持久化**：重启服务自动加载已有索引
- **健康检查接口**：GET /health

## API 使用说明（增强版）

### 1. 健康检查

```bash
curl http://localhost:8002/health
```

### 2. 添加文本文档

```bash
curl -X POST http://localhost:8002/documents \
  -H "Content-Type: application/json" \
  -d '[
    {"content": "产品A的价格是100元", "metadata": {"category": "价格"}},
    {"content": "产品B的价格是200元", "metadata": {"category": "价格"}}
  ]'
```

### 3. 上传文件

```bash
# 上传 PDF
curl -X POST http://localhost:8002/upload \
  -F "file=@document.pdf"

# 上传 TXT
curl -X POST http://localhost:8002/upload \
  -F "file=@notes.txt"
```

### 4. 问答

```bash
curl -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -d '{"question": "产品A多少钱？", "top_k": 3}'
```

### 5. 查看知识库状态

```bash
curl http://localhost:8002/documents
```

### 6. 清空知识库

```bash
curl -X DELETE http://localhost:8002/documents
```

## 架构对比：基础版 vs 增强版

| 特性 | 基础版 (main.py) | 增强版 (main_enhanced.py) |
|------|-----------------|-------------------------|
| 向量存储 | 内存 + numpy | FAISS |
| 检索算法 | cosine_similarity | FAISS IndexFlatIP |
| 文档分割 | 无（整篇文档） | RecursiveCharacterTextSplitter |
| 文件上传 | 不支持 | PDF / TXT |
| 持久化 | JSON 文件 | FAISS 二进制 + JSON |
| 适用规模 | 小数据量 | 中大规模 |
| 性能 | 一般 | 高效（万级以上优势显现） |

## RAG 核心概念

### 什么是 RAG？

RAG（Retrieval-Augmented Generation）= 检索增强生成

```
用户问题 → 文档分割 → Embedding → 向量检索 → 找到相关 chunks → 大模型生成回答
```

### 为什么需要 RAG？

| 问题 | 传统大模型 | RAG |
|------|----------|-----|
| 知识过时 | ❌ 知识截止于训练时间 | ✅ 实时检索最新信息 |
| 私有数据 | ❌ 无法访问企业数据 | ✅ 可接入任意知识库 |
| 幻觉问题 | ❌ 可能编造信息 | ✅ 回答有据可查 |
| 成本 | ❌ 微调成本高 | ✅ 无需训练模型 |

### RAG vs 微调

| 特性 | RAG | 微调 |
|------|-----|------|
| 适用场景 | 知识更新频繁 | 特定任务优化 |
| 成本 | 低 | 高 |
| 实时性 | ✅ 实时 | ❌ 需重新训练 |
| 可解释性 | ✅ 可追溯来源 | ❌ 黑盒 |

## 文档分割最佳实践

### 为什么分割很重要？

1. **语义聚焦**：每个 chunk 聚焦一个主题，检索更精准
2. **上下文限制**：大模型有 token 限制，需要控制输入长度
3. **避免稀释**：长文档的 Embedding 会稀释关键信息

### 参数选择建议

| 场景 | chunk_size | chunk_overlap | 说明 |
|------|-----------|---------------|------|
| 通用文档 | 300-500 | 50-100 | 平衡精度与上下文 |
| 代码文档 | 300-800 | 100-200 | 保持函数/类完整 |
| 短问答对 | 100-200 | 20-50 | 精准检索 |
| 长文章 | 500-1000 | 100-200 | 保持段落连贯 |

### 推荐策略

- **RecursiveCharacterTextSplitter**：优先按段落、句子分割，保持语义完整
- **chunk_overlap**：设置为 chunk_size 的 10%-25%
- **中文文档**：注意按标点符号分割（。！？）

## FAISS 索引类型选择

| 索引类型 | 特点 | 适用场景 |
|---------|------|---------|
| IndexFlatIP/IndexFlatL2 | 精确搜索 | 数据量 < 10万 |
| IndexIVFFlat | 近似搜索，速度快 | 数据量 10万-100万 |
| IndexHNSW | 图搜索，精度高 | 数据量 > 100万 |
| IndexPQ | 乘积量化，内存小 | 超大规模，内存受限 |

当前项目使用 IndexFlatIP（精确内积搜索），因为数据量通常不大，精确搜索更可靠。

## 实践建议

1. **文档质量**：知识库文档要结构清晰、信息准确
2. **分块策略**：长文档要合理分割，保持语义完整
3. **检索优化**：根据效果调整 top_k 和 chunk_size 参数
4. **Prompt 设计**：引导模型正确使用检索结果，明确限制信息来源
5. **元数据管理**：为文档添加分类、来源等元数据，便于后续过滤
6. **索引持久化**：定期保存向量索引，避免重复计算 Embedding

## 依赖安装

```bash
pip install faiss-cpu pypdf langchain-text-splitters
```

> 注：ChromaDB 目前与 Python 3.14 存在兼容性问题，本项目使用 FAISS 作为替代方案。FAISS 是 Facebook 开源的高性能向量检索库，在企业级应用中广泛使用。

## 下一步

Week 4: AI Agent 入门
- Function Calling
- ReAct 模式
- 工具调用
