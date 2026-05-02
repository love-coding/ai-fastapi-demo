"""
Week 3 - Step 1: 简单向量数据库实现
====================================
功能：从零实现一个简单的向量数据库，理解 RAG 的核心原理
作者：AI学习项目
日期：2026-05-02

学习目标：
1. 理解什么是向量（Embedding）
2. 理解什么是向量数据库
3. 理解相似度搜索的原理
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ====================================
# 第一部分：什么是向量（Embedding）？
# ====================================
"""
向量（Embedding）是将文本转换为数字数组的过程。

例如：
"苹果" -> [0.1, 0.5, 0.3, ...]
"香蕉" -> [0.1, 0.6, 0.2, ...]
"汽车" -> [0.8, 0.1, 0.9, ...]

相似的词语，向量也更相似！
"""

print("=" * 50)
print("第一部分：理解向量")
print("=" * 50)

# 模拟一些文本的向量表示（实际项目中会用 Embedding API 生成）
# 这里我们手动创建简单的向量来演示原理
documents = [
    {"text": "Python 是一种编程语言", "vector": np.array([0.9, 0.1, 0.1])},
    {"text": "Java 是一种编程语言", "vector": np.array([0.8, 0.2, 0.1])},
    {"text": "苹果是一种水果", "vector": np.array([0.1, 0.9, 0.2])},
    {"text": "香蕉是一种水果", "vector": np.array([0.1, 0.8, 0.3])},
    {"text": "汽车是一种交通工具", "vector": np.array([0.3, 0.1, 0.9])},
]

print("\n文档库：")
for i, doc in enumerate(documents):
    print(f"  {i+1}. {doc['text']}")
    print(f"     向量: {doc['vector']}")

# ====================================
# 第二部分：什么是相似度？
# ====================================
"""
相似度计算方法：
1. 余弦相似度（Cosine Similarity）- 最常用
2. 欧几里得距离（Euclidean Distance）
3. 点积（Dot Product）

余弦相似度：计算两个向量之间的夹角
- 值范围：-1 到 1
- 1 表示完全相同
- 0 表示无关
- -1 表示完全相反
"""

print("\n" + "=" * 50)
print("第二部分：相似度计算")
print("=" * 50)

# 计算两个向量的相似度
vec1 = np.array([0.9, 0.1, 0.1])  # Python
vec2 = np.array([0.8, 0.2, 0.1])  # Java
vec3 = np.array([0.1, 0.9, 0.2])  # 苹果

# 使用 scikit-learn 计算余弦相似度
sim_1_2 = cosine_similarity([vec1], [vec2])[0][0]
sim_1_3 = cosine_similarity([vec1], [vec3])[0][0]

print(f"\n'Python' 和 'Java' 的相似度: {sim_1_2:.4f}")
print(f"'Python' 和 '苹果' 的相似度: {sim_1_3:.4f}")
print("\n结论：编程语言之间的相似度更高！")

# ====================================
# 第三部分：实现简单的向量数据库
# ====================================

class SimpleVectorStore:
    """
    简单的向量数据库实现

    功能：
    1. 添加文档（文本 + 向量）
    2. 相似度搜索
    """

    def __init__(self):
        self.documents = []
        self.vectors = []

    def add_document(self, text: str, vector: np.ndarray):
        """添加文档到向量数据库"""
        self.documents.append(text)
        self.vectors.append(vector)

    def search(self, query_vector: np.ndarray, top_k: int = 2) -> list:
        """
        搜索最相似的文档

        Args:
            query_vector: 查询向量
            top_k: 返回最相似的 k 个文档

        Returns:
            包含 (文本, 相似度) 的列表
        """
        if len(self.vectors) == 0:
            return []

        # 计算查询向量与所有文档的相似度
        similarities = cosine_similarity([query_vector], self.vectors)[0]

        # 获取最相似的 k 个文档的索引
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # 返回结果
        results = []
        for idx in top_indices:
            results.append({
                "text": self.documents[idx],
                "similarity": similarities[idx]
            })
        return results

print("\n" + "=" * 50)
print("第三部分：向量数据库搜索演示")
print("=" * 50)

# 创建向量数据库
vector_store = SimpleVectorStore()

# 添加文档
for doc in documents:
    vector_store.add_document(doc["text"], doc["vector"])

print("\n已添加 5 个文档到向量数据库")

# 模拟查询
print("\n--- 查询：'我想学习编程' ---")
query_vector = np.array([0.85, 0.15, 0.1])  # 模拟"编程"相关的向量
results = vector_store.search(query_vector, top_k=2)

print("\n最相似的 2 个文档：")
for i, result in enumerate(results, 1):
    print(f"  {i}. {result['text']}")
    print(f"     相似度: {result['similarity']:.4f}")

print("\n--- 查询：'我想吃水果' ---")
query_vector = np.array([0.1, 0.85, 0.15])  # 模拟"水果"相关的向量
results = vector_store.search(query_vector, top_k=2)

print("\n最相似的 2 个文档：")
for i, result in enumerate(results, 1):
    print(f"  {i}. {result['text']}")
    print(f"     相似度: {result['similarity']:.4f}")

# ====================================
# 总结
# ====================================
print("\n" + "=" * 50)
print("总结：向量数据库的核心原理")
print("=" * 50)
print("""
1. 文本 -> 向量（Embedding）
   将文本转换为数字数组，保留语义信息

2. 相似度计算
   使用余弦相似度衡量两个向量的相似程度

3. 向量搜索
   找到与查询向量最相似的文档向量

这就是 RAG（检索增强生成）的基础！
下一步：使用真实的大模型 Embedding API 来生成向量。
""")
