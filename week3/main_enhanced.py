"""
Week 3 - 增强版知识库问答系统
=============================
功能：基于 FAISS + 文档分割 + 文件上传的 RAG 知识库问答 API
改进点：
1. 使用 FAISS 替代内存检索，支持大规模数据
2. 增加文档自动分割，提升检索精度
3. 支持 PDF、TXT 文件上传
4. 支持向量索引持久化

API 接口：
- POST /documents: 添加文本文档到知识库
- POST /upload: 上传文件（PDF/TXT）到知识库
- POST /query: 问答接口
- GET /documents: 查看知识库文档列表
- DELETE /documents: 清空知识库
"""

import os
import json
import tempfile
import shutil
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from zhipuai import ZhipuAI
import numpy as np
import faiss
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ====================================
# 初始化
# ====================================
load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
app = FastAPI(title="增强版知识库问答系统", version="2.0.0")

# 持久化文件路径
INDEX_FILE = "week3/faiss_index.bin"
META_FILE = "week3/faiss_metadata.json"

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
# FAISS RAG 引擎
# ====================================

class FAISSRAGEngine:
    """
    基于 FAISS 的 RAG 引擎

    核心流程：
    1. 文档分割（RecursiveCharacterTextSplitter）
    2. Embedding 生成
    3. FAISS 索引存储
    4. 向量检索 + 大模型生成
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []      # chunk 文本列表
        self.metadata = []    # 元数据列表
        self.doc_count = 0    # 文档计数

        # 文档分割器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
            length_function=len,
        )

        self._init_index()
        self._load()

    def _init_index(self):
        """初始化 FAISS 索引"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)

    def get_embedding(self, text: str) -> np.ndarray:
        """获取文本的 Embedding 向量"""
        response = client.embeddings.create(
            model="embedding-3",
            input=text[:8000]  # 限制长度，避免超出限制
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        # L2 归一化，使内积等价于余弦相似度
        faiss.normalize_L2(embedding.reshape(1, -1))
        return embedding

    def split_document(self, text: str) -> List[str]:
        """将文档分割为 chunks"""
        return self.splitter.split_text(text)

    def add_text(self, content: str, metadata: dict = None):
        """
        添加文本到知识库

        流程：
        1. 文档分割 -> chunks
        2. 每个 chunk 生成 Embedding
        3. 添加到 FAISS 索引
        """
        if not content or not content.strip():
            return 0

        # 1. 分割文档
        chunks = self.split_document(content)

        if not chunks:
            return 0

        # 2. 生成 Embeddings
        embeddings = []
        for chunk in chunks:
            emb = self.get_embedding(chunk)
            embeddings.append(emb)

        embeddings_np = np.vstack(embeddings)

        # 3. 添加到 FAISS 索引
        self.index.add(embeddings_np)

        # 4. 保存元数据
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            meta = metadata or {}
            meta["chunk_index"] = i
            meta["total_chunks"] = len(chunks)
            self.metadata.append(meta)

        self.doc_count += 1
        return len(chunks)

    def add_documents(self, documents: List[Document]):
        """批量添加文档"""
        total_chunks = 0
        for doc in documents:
            count = self.add_text(doc.content, doc.metadata)
            total_chunks += count
        return total_chunks

    def add_pdf(self, file_path: str, original_filename: str = None):
        """添加 PDF 文件到知识库"""
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

        full_text = "\n".join(text_parts)
        metadata = {
            "source_type": "pdf",
            "filename": original_filename or os.path.basename(file_path),
            "total_pages": len(reader.pages),
        }
        return self.add_text(full_text, metadata)

    def add_txt(self, file_path: str, original_filename: str = None):
        """添加 TXT 文件到知识库"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        metadata = {
            "source_type": "txt",
            "filename": original_filename or os.path.basename(file_path),
        }
        return self.add_text(content, metadata)

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        """检索相关文档 chunks"""
        if self.index.ntotal == 0:
            return []

        # 获取查询向量
        query_embedding = self.get_embedding(query)

        # FAISS 检索
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append({
                "content": self.chunks[idx][:200] + "..." if len(self.chunks[idx]) > 200 else self.chunks[idx],
                "full_content": self.chunks[idx],
                "similarity": float(score),
                "metadata": self.metadata[idx],
            })
        return results

    def generate(self, query: str, context: List[str]) -> str:
        """基于上下文生成回答"""
        context_text = "\n\n".join([f"参考文档 {i+1}:\n{doc}" for i, doc in enumerate(context)])

        prompt = f"""你是一个专业的知识库助手。请根据以下参考文档回答用户问题。

参考文档：
{context_text}

用户问题：{query}

要求：
1. 只使用参考文档中的信息回答
2. 如果参考文档没有相关信息，请明确告知"我在知识库中没有找到相关信息"
3. 回答要简洁、准确、有条理
4. 如果涉及具体数据，请准确引用

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
                "similarity": doc["similarity"],
                "metadata": doc["metadata"]
            } for doc in retrieved]
        }

    def clear(self):
        """清空知识库"""
        self._init_index()
        self.chunks = []
        self.metadata = []
        self.doc_count = 0

    def get_stats(self) -> dict:
        """获取知识库统计信息"""
        return {
            "total_chunks": len(self.chunks),
            "total_documents": self.doc_count,
            "vector_dimension": self.embedding_dim,
        }

    def save(self):
        """保存索引和元数据"""
        if self.index.ntotal > 0:
            faiss.write_index(self.index, INDEX_FILE)
            with open(META_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "chunks": self.chunks,
                    "metadata": self.metadata,
                    "doc_count": self.doc_count,
                }, f, ensure_ascii=False, indent=2)

    def _load(self):
        """加载索引和元数据"""
        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]
            self.doc_count = data.get("doc_count", 0)
            print(f"已加载 {len(self.chunks)} 个 chunks")


# 全局引擎实例
rag_engine = FAISSRAGEngine()

# ====================================
# API 接口
# ====================================

@app.post("/documents")
async def add_documents(documents: List[Document]):
    """
    添加文本文档到知识库

    示例请求：
    ```json
    [
        {"content": "产品A的价格是100元", "metadata": {"category": "价格"}},
        {"content": "产品B的价格是200元", "metadata": {"category": "价格"}}
    ]
    ```
    """
    try:
        total_chunks = rag_engine.add_documents(documents)
        rag_engine.save()
        return {
            "message": f"成功添加 {len(documents)} 个文档，生成 {total_chunks} 个 chunks",
            "stats": rag_engine.get_stats()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传文件到知识库

    支持格式：.pdf, .txt
    """
    # 检查文件类型
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in [".pdf", ".txt"]:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}，仅支持 .pdf 和 .txt")

    # 保存临时文件
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)

    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 根据类型处理
        if ext == ".pdf":
            chunk_count = rag_engine.add_pdf(temp_path, filename)
        else:
            chunk_count = rag_engine.add_txt(temp_path, filename)

        rag_engine.save()

        return {
            "message": f"文件 '{filename}' 上传成功",
            "chunks_added": chunk_count,
            "stats": rag_engine.get_stats()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")

    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)


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
    """查看知识库统计信息和文档列表"""
    stats = rag_engine.get_stats()
    recent_chunks = []
    for i in range(max(0, len(rag_engine.chunks) - 5), len(rag_engine.chunks)):
        recent_chunks.append({
            "id": i,
            "content": rag_engine.chunks[i][:100] + "...",
            "metadata": rag_engine.metadata[i]
        })

    return {
        "stats": stats,
        "recent_chunks": recent_chunks
    }


@app.delete("/documents")
async def clear_documents():
    """清空知识库"""
    rag_engine.clear()
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    if os.path.exists(META_FILE):
        os.remove(META_FILE)
    return {"message": "知识库已清空"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "version": "2.0.0",
        "vector_db": "faiss",
        "embedding_model": "embedding-3",
        "llm_model": "glm-4-flash",
        "chunking": "recursive",
    }


# ====================================
# 启动服务
# ====================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("增强版知识库问答系统 API")
    print("=" * 60)
    print("\nAPI 文档: http://localhost:8002/docs")
    print("添加文档: POST http://localhost:8002/documents")
    print("上传文件: POST http://localhost:8002/upload")
    print("问答接口: POST http://localhost:8002/query")
    print("健康检查: GET  http://localhost:8002/health")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8002)
