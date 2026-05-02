"""
Week 3 - Step 2: 使用 Embedding API
====================================
功能：使用智谱 AI 的 Embedding API 将文本转换为向量
作者：AI学习项目
日期：2026-05-02

学习目标：
1. 学会调用 Embedding API
2. 理解 Embedding 的维度和含义
3. 实现真实的语义搜索
"""

import os
import numpy as np
from dotenv import load_dotenv
from zhipuai import ZhipuAI
from sklearn.metrics.pairwise import cosine_similarity

# 加载环境变量
load_dotenv()

# 初始化客户端
client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

print("=" * 50)
print("Step 2: 使用 Embedding API")
print("=" * 50)

# ====================================
# 第一部分：调用 Embedding API
# ====================================

def get_embedding(text: str) -> list:
    """
    调用智谱 AI Embedding API 获取文本向量

    Args:
        text: 要转换的文本

    Returns:
        向量列表（768 维）
    """
    response = client.embeddings.create(
        model="embedding-3",  # 智谱的 Embedding 模型
        input=text
    )
    return response.data[0].embedding

# 测试 Embedding API
print("\n测试 Embedding API...")
test_text = "Python 是一种编程语言"
embedding = get_embedding(test_text)
print(f"文本: {test_text}")
print(f"向量维度: {len(embedding)}")
print(f"向量前 5 个值: {embedding[:5]}")

# ====================================
# 第二部分：实现语义搜索
# ====================================

class SemanticSearch:
    """
    基于语义的文档搜索
    使用真实的 Embedding API
    """

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_document(self, text: str):
        """添加文档并自动生成 Embedding"""
        print(f"正在添加文档: {text[:30]}...")
        embedding = get_embedding(text)
        self.documents.append(text)
        self.embeddings.append(embedding)
        print(f"  完成！向量维度: {len(embedding)}")

    def search(self, query: str, top_k: int = 2) -> list:
        """
        语义搜索

        Args:
            query: 查询文本
            top_k: 返回最相似的 k 个文档

        Returns:
            包含 (文本, 相似度) 的列表
        """
        print(f"\n正在搜索: {query}")

        # 获取查询的 Embedding
        query_embedding = get_embedding(query)

        # 计算相似度
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # 获取最相似的文档
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.documents[idx],
                "similarity": similarities[idx]
            })
        return results

# ====================================
# 第三部分：实战演示
# ====================================

print("\n" + "=" * 50)
print("实战演示：语义搜索")
print("=" * 50)

# 创建搜索引擎
search_engine = SemanticSearch()

# 添加文档
print("\n添加文档到知识库...")
docs = [
    "Python 是一种高级编程语言，以简洁易读著称。",
    "FastAPI 是一个现代、快速的 Python Web 框架。",
    "向量数据库用于存储和检索高维向量数据。",
    "RAG 是检索增强生成的缩写，用于提升大模型的回答质量。",
    "机器学习是人工智能的一个分支，让计算机从数据中学习。",
]

for doc in docs:
    search_engine.add_document(doc)

# 进行搜索
print("\n" + "=" * 50)
print("搜索测试")
print("=" * 50)

results = search_engine.search("我想学习 Python 开发", top_k=2)
print("\n搜索结果：")
for i, result in enumerate(results, 1):
    print(f"  {i}. {result['text']}")
    print(f"     相似度: {result['similarity']:.4f}")

results = search_engine.search("什么是 RAG 技术？", top_k=2)
print("\n搜索结果：")
for i, result in enumerate(results, 1):
    print(f"  {i}. {result['text']}")
    print(f"     相似度: {result['similarity']:.4f}")

# ====================================
# 总结
# ====================================
print("\n" + "=" * 50)
print("总结")
print("=" * 50)
print("""
1. Embedding API 将文本转换为高维向量（768 维）
2. 相似的内容在向量空间中距离更近
3. 可以通过向量相似度实现语义搜索

下一步：将向量搜索与大模型结合，实现 RAG！
""")
