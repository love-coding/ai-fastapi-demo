"""
Week 3 - Step 4: FAISS 向量数据库
====================================
功能：使用 Facebook 的 FAISS 实现高效的向量检索
学习目标：
1. 了解 FAISS 的基本使用
2. 对比内存检索与 FAISS 的性能差异
3. 掌握向量持久化存储
"""

import os
import time
import numpy as np
from dotenv import load_dotenv
from zhipuai import ZhipuAI
import faiss

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

print("=" * 60)
print("Step 4: FAISS 向量数据库")
print("=" * 60)


def get_embedding(text: str) -> list:
    """获取文本的 Embedding（智谱 embedding-3，768维）"""
    response = client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return response.data[0].embedding


# ====================================
# 准备测试数据
# ====================================

documents = [
    "Python 是一种高级编程语言，以简洁易读著称，适合 Web 开发、数据分析等场景。",
    "FastAPI 是一个现代、快速的 Python Web 框架，基于 Starlette 和 Pydantic。",
    "Django 是 Python 的全栈 Web 框架，内置 ORM、管理后台、认证系统。",
    "Flask 是一个轻量级的 Python Web 框架，灵活简单，适合小型项目。",
    "向量数据库用于存储和检索高维向量数据，支持语义搜索。",
    "ChromaDB 是一个开源的向量数据库，简单易用，适合快速原型。",
    "FAISS 是 Facebook 开源的向量检索库，支持海量数据高效检索。",
    "Milvus 是一个分布式向量数据库，适合企业级大规模部署。",
    "RAG 是检索增强生成的缩写，结合检索系统和大模型提升回答质量。",
    "大语言模型（LLM）通过海量文本训练，具备强大的语言理解和生成能力。",
    "Prompt Engineering 是通过设计输入提示来引导大模型输出预期结果的技术。",
    "Embedding 是将文本、图像等转换为数值向量的技术，保留语义信息。",
    "机器学习是人工智能的一个分支，让计算机从数据中学习规律。",
    "深度学习是机器学习的子领域，使用神经网络处理复杂任务。",
    "神经网络由多层神经元组成，能够模拟人脑的信息处理方式。",
]

print(f"\n准备 {len(documents)} 个文档...")

# 获取所有文档的 Embedding
print("正在生成 Embedding（需要一些时间）...")
embeddings = []
for i, doc in enumerate(documents):
    emb = get_embedding(doc)
    embeddings.append(emb)
    print(f"  [{i+1}/{len(documents)}] 完成")

embeddings_np = np.array(embeddings, dtype=np.float32)
dimension = embeddings_np.shape[1]  # 768

print(f"\n向量维度: {dimension}")
print(f"向量矩阵形状: {embeddings_np.shape}")

# ====================================
# FAISS 索引创建
# ====================================

print("\n" + "=" * 60)
print("FAISS 索引创建")
print("=" * 60)

# 创建 FAISS 索引
# IndexFlatIP: 精确的内积搜索（需要归一化后等同于余弦相似度）
# IndexFlatL2: 精确的欧氏距离搜索

# 为了使用余弦相似度，先对向量进行 L2 归一化
faiss.normalize_L2(embeddings_np)

index = faiss.IndexFlatIP(dimension)  # IP = Inner Product
print(f"索引类型: {type(index).__name__}")
print(f"是否训练: {index.is_trained}")

# 添加向量到索引
index.add(embeddings_np)
print(f"索引中向量数量: {index.ntotal}")

# ====================================
# 检索测试
# ====================================

print("\n" + "=" * 60)
print("检索测试")
print("=" * 60)

queries = [
    "Python 的 Web 框架有哪些？",
    "什么是向量数据库？",
    "RAG 技术如何工作？",
    "深度学习和机器学习有什么关系？",
]

for query in queries:
    print(f"\n问题: {query}")

    # 获取查询向量
    query_embedding = np.array([get_embedding(query)], dtype=np.float32)
    faiss.normalize_L2(query_embedding)

    # FAISS 检索
    top_k = 3
    start_time = time.time()
    scores, indices = index.search(query_embedding, top_k)
    faiss_time = time.time() - start_time

    print(f"  FAISS 检索耗时: {faiss_time*1000:.2f} ms")
    print(f"  最相似的 {top_k} 个文档:")
    for i, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        print(f"    [{i}] 相似度: {score:.4f} | {documents[idx]}")

# ====================================
# FAISS 索引保存与加载
# ====================================

print("\n" + "=" * 60)
print("索引持久化")
print("=" * 60)

index_file = "faiss_index.bin"

# 保存索引
faiss.write_index(index, index_file)
print(f"索引已保存到: {index_file}")
print(f"文件大小: {os.path.getsize(index_file) / 1024:.2f} KB")

# 加载索引
loaded_index = faiss.read_index(index_file)
print(f"加载后向量数量: {loaded_index.ntotal}")

# 验证加载后的索引是否正常工作
query_embedding = np.array([get_embedding("Python Web 开发")], dtype=np.float32)
faiss.normalize_L2(query_embedding)
scores, indices = loaded_index.search(query_embedding, 2)
print(f"\n验证检索: {documents[indices[0][0]]}")

# 清理
os.remove(index_file)

# ====================================
# 性能对比：暴力搜索 vs FAISS
# ====================================

print("\n" + "=" * 60)
print("性能对比：暴力搜索 vs FAISS")
print("=" * 60)

# 暴力搜索（使用 sklearn cosine_similarity）
from sklearn.metrics.pairwise import cosine_similarity

query_embedding = np.array([get_embedding("向量数据库的特点")], dtype=np.float32)

start = time.time()
sklearn_similarities = cosine_similarity(query_embedding, embeddings_np)[0]
top_indices_sklearn = np.argsort(sklearn_similarities)[::-1][:3]
sklearn_time = time.time() - start

faiss.normalize_L2(query_embedding)
start = time.time()
scores, indices = index.search(query_embedding, 3)
faiss_time = time.time() - start

print(f"暴力搜索耗时: {sklearn_time*1000:.2f} ms")
print(f"FAISS 搜索耗时: {faiss_time*1000:.2f} ms")
print(f"加速比: {sklearn_time/faiss_time:.1f}x")

print("\n注意：当数据量小时差异不明显，FAISS 的优势在万级以上数据量时才会显现。")
print("      此外，FAISS 还支持近似搜索（IndexIVFFlat、IndexHNSW），可进一步提升速度。")

# ====================================
# 总结
# ====================================
print("\n" + "=" * 60)
print("总结：FAISS 的优势")
print("=" * 60)
print("""
1. 高性能：C++ 实现，支持 GPU 加速
2. 大规模：可处理十亿级向量
3. 多种索引：Flat、IVF、HNSW、PQ 等，权衡速度与精度
4. 持久化：支持索引保存和加载
5. 批处理：支持一次性检索多个查询

与 ChromaDB 的区别：
- FAISS：纯向量检索库，需要自行管理文档和元数据
- ChromaDB：完整的向量数据库，自带存储、过滤、元数据管理

下一步：将 FAISS 与文档分割结合，构建更强大的 RAG 系统！
""")
