"""
Week 3 - 实战项目：知识库问答系统
===================================
功能：基于 RAG 的知识库问答 API 服务
作者：AI学习项目
日期：2026-05-02

API 接口：
- POST /documents: 添加文档到知识库
- POST /query: 问答接口
- GET /documents: 查看知识库文档列表
"""

import os
import json
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from zhipuai import ZhipuAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ====================================
# 初始化
# ====================================
load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
app = FastAPI(title="知识库问答系统", version="1.0.0")

# ====================================
# 数据模型
# ====================================

class Document(BaseModel):
    """文档模型"""
    content: str
    metadata: Optional[dict] = None

class QueryRequest(BaseModel):
    """问答请求"""
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    """问答响应"""
    question: str
    answer: str
    sources: List[dict]

# ====================================
# RAG 引擎
# ====================================

class RAGEngine:
    """RAG 引擎"""

    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []

    def get_embedding(self, text: str) -> list:
        """获取文本向量"""
        response = client.embeddings.create(
            model="embedding-3",
            input=text
        )
        return response.data[0].embedding

    def add_document(self, content: str, metadata: dict = None):
        """添加单个文档"""
        embedding = self.get_embedding(content)
        self.documents.append(content)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})

    def add_documents(self, documents: List[Document]):
        """批量添加文档"""
        for doc in documents:
            self.add_document(doc.content, doc.metadata)

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        """检索相关文档"""
        if len(self.embeddings) == 0:
            return []

        query_embedding = self.get_embedding(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "content": self.documents[idx][:200] + "...",  # 截断显示
                "full_content": self.documents[idx],
                "similarity": float(similarities[idx]),
                "metadata": self.metadata[idx]
            })
        return results

    def generate(self, query: str, context: List[str]) -> str:
        """生成回答"""
        context_text = "\n\n".join([f"文档 {i+1}:\n{doc}"
                                    for i, doc in enumerate(context)])

        prompt = f"""你是一个专业的客服助手。请根据以下知识库内容回答用户问题。

知识库内容：
{context_text}

用户问题：{query}

要求：
1. 只使用知识库中的信息回答
2. 如果知识库没有相关信息，请说"抱歉，我在知识库中没有找到相关信息"
3. 回答要简洁清晰
4. 如果涉及价格等具体信息，要准确引用

回答："""

        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def query(self, question: str, top_k: int = 3) -> dict:
        """完整 RAG 流程"""
        # 检索
        retrieved = self.retrieve(question, top_k)

        if not retrieved:
            return {
                "question": question,
                "answer": "抱歉，知识库中暂无文档，请先添加文档。",
                "sources": []
            }

        # 生成
        context = [doc["full_content"] for doc in retrieved]
        answer = self.generate(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": [{
                "content": doc["content"],
                "similarity": doc["similarity"]
            } for doc in retrieved]
        }

    def clear(self):
        """清空知识库"""
        self.documents = []
        self.embeddings = []
        self.metadata = []

    def save(self, filepath: str):
        """保存知识库到文件"""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "metadata": self.metadata
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """从文件加载知识库"""
        if not os.path.exists(filepath):
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.documents = data["documents"]
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]

# 全局 RAG 引擎实例
rag_engine = RAGEngine()

# 启动时加载已有知识库
KNOWLEDGE_BASE_FILE = "knowledge_base.json"

@app.on_event("startup")
async def startup():
    """启动时加载知识库"""
    if os.path.exists(KNOWLEDGE_BASE_FILE):
        rag_engine.load(KNOWLEDGE_BASE_FILE)
        print(f"已加载 {len(rag_engine.documents)} 个文档")

# ====================================
# API 接口
# ====================================

@app.post("/documents")
async def add_documents(documents: List[Document]):
    """
    添加文档到知识库

    示例请求：
    ```json
    [
        {"content": "产品A的价格是100元"},
        {"content": "产品B的价格是200元"}
    ]
    ```
    """
    try:
        rag_engine.add_documents(documents)
        # 保存到文件
        rag_engine.save(KNOWLEDGE_BASE_FILE)
        return {
            "message": f"成功添加 {len(documents)} 个文档",
            "total": len(rag_engine.documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    问答接口

    示例请求：
    ```json
    {
        "question": "产品A的价格是多少？",
        "top_k": 3
    }
    ```
    """
    try:
        result = rag_engine.query(request.question, request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """查看知识库文档列表"""
    return {
        "total": len(rag_engine.documents),
        "documents": [
            {
                "id": i,
                "content": doc[:100] + "...",
                "metadata": rag_engine.metadata[i]
            }
            for i, doc in enumerate(rag_engine.documents)
        ]
    }

@app.delete("/documents")
async def clear_documents():
    """清空知识库"""
    rag_engine.clear()
    if os.path.exists(KNOWLEDGE_BASE_FILE):
        os.remove(KNOWLEDGE_BASE_FILE)
    return {"message": "知识库已清空"}

# ====================================
# 启动服务
# ====================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("知识库问答系统 API")
    print("=" * 60)
    print("\nAPI 文档: http://localhost:8002/docs")
    print("添加文档: POST http://localhost:8002/documents")
    print("问答接口: POST http://localhost:8002/query")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8002)
