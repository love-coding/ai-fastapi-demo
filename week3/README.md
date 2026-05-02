# Week 3: RAG 基础与应用

## 学习目标

1. 理解向量数据库原理
2. 掌握 Embedding API 的使用
3. 实现 RAG 检索增强生成
4. 构建完整的知识库问答系统

## 文件说明

```
week3/
├── 01_simple_vector_store.py   # Step 1: 理解向量存储原理
├── 02_embedding_api.py         # Step 2: 使用 Embedding API
├── 03_rag_demo.py              # Step 3: 完整 RAG 演示
├── main.py                     # 实战: 知识库问答 API 服务
└── test_api.py                 # API 测试脚本
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
- 理解向量维度含义

### Step 3: 完整 RAG 演示

```bash
python week3/03_rag_demo.py
```

学习内容：
- RAG 完整流程
- 文档检索 + 大模型生成
- 实战：产品知识库问答

### Step 4: 知识库问答 API 服务

```bash
# 启动服务
python week3/main.py

# 测试 API
python week3/test_api.py
```

## API 使用说明

### 1. 添加文档

```bash
curl -X POST http://localhost:8002/documents \
  -H "Content-Type: application/json" \
  -d '[
    {"content": "产品A的价格是100元"},
    {"content": "产品B的价格是200元"}
  ]'
```

### 2. 问答

```bash
curl -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -d '{"question": "产品A多少钱？", "top_k": 3}'
```

### 3. 查看文档列表

```bash
curl http://localhost:8002/documents
```

## RAG 核心概念

### 什么是 RAG？

RAG（Retrieval-Augmented Generation）= 检索增强生成

```
用户问题 → 向量检索 → 找到相关文档 → 大模型生成回答
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

## 实践建议

1. **文档质量**：知识库文档要结构清晰、信息准确
2. **分块策略**：长文档要合理分割，保持语义完整
3. **检索优化**：根据效果调整 top_k 参数
4. **Prompt 设计**：引导模型正确使用检索结果

## 下一步

Week 4: AI Agent 入门
- Function Calling
- ReAct 模式
- 工具调用
