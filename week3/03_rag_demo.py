"""
Week 3 - Step 3: 实现 RAG（检索增强生成）
==========================================
功能：完整的 RAG 系统，结合向量检索与大模型
作者：AI学习项目
日期：2026-05-02

学习目标：
1. 理解 RAG 的完整流程
2. 实现文档的加载与分割
3. 实现检索 + 生成的完整链路
"""

import os
import numpy as np
from dotenv import load_dotenv
from zhipuai import ZhipuAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

print("=" * 60)
print("Week 3 - Step 3: RAG 系统实现")
print("=" * 60)

# ====================================
# RAG 核心组件
# ====================================

class SimpleRAG:
    """
    简单的 RAG 系统

    流程：
    1. 文档 -> Embedding -> 向量数据库
    2. 问题 -> Embedding -> 相似度搜索
    3. 问题 + 检索结果 -> 大模型 -> 回答
    """

    def __init__(self):
        self.documents = []      # 原始文档
        self.embeddings = []     # 文档向量

    def get_embedding(self, text: str) -> list:
        """获取文本的 Embedding"""
        response = client.embeddings.create(
            model="embedding-3",
            input=text
        )
        return response.data[0].embedding

    def add_documents(self, documents: List[str]):
        """
        添加文档到知识库

        Args:
            documents: 文档列表
        """
        print(f"\n正在添加 {len(documents)} 个文档...")
        for i, doc in enumerate(documents):
            print(f"  [{i+1}/{len(documents)}] 添加中...")
            embedding = self.get_embedding(doc)
            self.documents.append(doc)
            self.embeddings.append(embedding)
        print("✓ 文档添加完成！")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回最相似的 k 个文档

        Returns:
            相关文档列表
        """
        # 获取查询向量
        query_embedding = self.get_embedding(query)

        # 计算相似度
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # 获取 top_k 文档
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "content": self.documents[idx],
                "similarity": similarities[idx]
            })
        return results

    def generate(self, query: str, context: List[str]) -> str:
        """
        基于上下文生成回答

        Args:
            query: 用户问题
            context: 检索到的相关文档

        Returns:
            大模型生成的回答
        """
        # 构建 Prompt
        context_text = "\n\n".join([f"参考文档 {i+1}:\n{doc}"
                                    for i, doc in enumerate(context)])

        prompt = f"""你是一个智能助手。请根据以下参考文档回答用户问题。

{context_text}

用户问题：{query}

请基于参考文档回答问题，如果参考文档中没有相关信息，请明确告知。
回答："""

        # 调用大模型
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        RAG 完整流程：检索 + 生成

        Args:
            question: 用户问题
            top_k: 检索文档数量

        Returns:
            包含回答和相关文档的字典
        """
        print(f"\n问题: {question}")

        # 1. 检索相关文档
        print("正在检索相关文档...")
        retrieved = self.retrieve(question, top_k)

        # 2. 提取文档内容
        context = [doc["content"] for doc in retrieved]

        # 3. 生成回答
        print("正在生成回答...")
        answer = self.generate(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved
        }

# ====================================
# 实战演示
# ====================================

print("\n" + "=" * 60)
print("实战演示：构建知识库问答系统")
print("=" * 60)

# 创建 RAG 系统
rag = SimpleRAG()

# 准备知识库文档（模拟产品文档）
knowledge_base = [
    """
产品名称：智能客服助手 Pro
功能：自动回答客户咨询，支持多轮对话
特点：
- 支持 24 小时在线服务
- 可接入微信、钉钉等平台
- 支持自定义知识库
- 响应时间 < 2 秒
价格：基础版 999 元/月，专业版 2999 元/月
""",
    """
产品名称：文档智能分析系统
功能：自动解析 PDF、Word 等文档，提取关键信息
特点：
- 支持 20+ 种文档格式
- 自动提取表格、图表数据
- 支持批量处理
- 准确率 > 95%
价格：按处理页数计费，0.1 元/页
""",
    """
产品名称：智能写作助手
功能：辅助写作，支持多种文体
特点：
- 支持新闻、公文、营销文案等多种文体
- 提供智能改写建议
- 支持团队协作
- 内置语法检查
价格：个人版 49 元/月，团队版 199 元/月
""",
    """
技术支持政策：
1. 所有产品提供 7 天免费试用
2. 工作日 9:00-18:00 提供技术支持
3. 专业版客户享有专属客服
4. 提供在线培训课程
5. API 文档开放，支持二次开发
""",
    """
常见问题解答：
Q: 如何开通试用？
A: 在官网注册账号后，选择产品即可开通 7 天免费试用。

Q: 支持哪些支付方式？
A: 支持支付宝、微信支付、对公转账。

Q: 数据安全如何保障？
A: 所有数据加密存储，通过 ISO27001 认证。

Q: 可以私有化部署吗？
A: 专业版支持私有化部署，请联系销售团队。
""",
]

# 添加文档到知识库
rag.add_documents(knowledge_base)

# 进行问答测试
print("\n" + "=" * 60)
print("问答测试")
print("=" * 60)

# 测试问题 1
result = rag.query("智能客服助手的价格是多少？")
print("\n" + "-" * 60)
print(f"回答: {result['answer']}")
print("\n参考来源:")
for i, source in enumerate(result['sources'], 1):
    print(f"  [{i}] 相似度: {source['similarity']:.4f}")
    print(f"      {source['content'][:50]}...")

# 测试问题 2
result = rag.query("我想试用一下产品，怎么操作？")
print("\n" + "-" * 60)
print(f"回答: {result['answer']}")
print("\n参考来源:")
for i, source in enumerate(result['sources'], 1):
    print(f"  [{i}] 相似度: {source['similarity']:.4f}")
    print(f"      {source['content'][:50]}...")

# 测试问题 3
result = rag.query("文档分析系统支持哪些格式？")
print("\n" + "-" * 60)
print(f"回答: {result['answer']}")
print("\n参考来源:")
for i, source in enumerate(result['sources'], 1):
    print(f"  [{i}] 相似度: {source['similarity']:.4f}")
    print(f"      {source['content'][:50]}...")

# ====================================
# 总结
# ====================================
print("\n" + "=" * 60)
print("总结：RAG 核心流程")
print("=" * 60)
print("""
┌─────────────┐
│   用户问题   │
└──────┬──────┘
       ↓
┌─────────────┐
│ 问题→向量   │  Embedding API
└──────┬──────┘
       ↓
┌─────────────┐
│  向量检索   │  找到最相似的文档
└──────┬──────┘
       ↓
┌─────────────┐
│ 问题+文档   │
│   → 大模型  │  生成回答
└──────┬──────┘
       ↓
┌─────────────┐
│   回答用户   │
└─────────────┘

RAG 的优势：
1. 让大模型"知道"私有数据
2. 减少幻觉，回答有据可查
3. 无需重新训练模型

下一步：构建完整的知识库问答 API 服务！
""")
