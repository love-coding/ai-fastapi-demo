"""
Week 3 - Step 5: 文档分割（Chunking）
======================================
功能：学习如何合理分割长文档，提升 RAG 检索精度
学习目标：
1. 理解为什么需要文档分割
2. 掌握不同的分割策略
3. 了解 chunk_size 和 chunk_overlap 的影响
"""

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

print("=" * 60)
print("Step 5: 文档分割（Chunking）")
print("=" * 60)

# ====================================
# 为什么需要文档分割？
# ====================================

print("""
为什么需要文档分割？
-------------------
1. 大模型有上下文长度限制（通常 4K-128K tokens）
2. 长文档直接 Embedding 会稀释语义，降低检索精度
3. 分割后每个 chunk 聚焦一个主题，检索更精准
4. 可以控制每次传入大模型的上下文量

分割策略：
- 按字符数分割（固定长度）
- 递归分割（优先按段落、句子分割，保持语义完整）
- 按语义分割（根据内容语义边界分割）
- 按 Token 分割（根据模型 tokenizer）
""")

# ====================================
# 准备测试文档
# ====================================

long_document = """
人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支，致力于创造能够模拟人类智能的系统。

机器学习是人工智能的核心技术之一。它使计算机能够从数据中学习规律，而无需被明确编程。机器学习分为监督学习、无监督学习和强化学习三大类。监督学习使用带标签的数据进行训练，无监督学习发现数据中的隐藏模式，强化学习通过与环境交互来学习最优策略。

深度学习是机器学习的一个子领域，基于人工神经网络。深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展。卷积神经网络（CNN）擅长处理图像数据，循环神经网络（RNN）适合序列数据，而 Transformer 架构则是当前大语言模型的基础。

大语言模型（LLM）是通过海量文本数据训练的深度学习模型。GPT 系列、BERT、T5 等都是著名的大语言模型。这些模型能够理解和生成自然语言，应用于机器翻译、文本摘要、问答系统等多种任务。大语言模型的训练需要大量的计算资源，通常使用数千个 GPU 进行分布式训练。

RAG（检索增强生成）是一种将检索系统与大语言模型结合的技术。RAG 通过从外部知识库检索相关信息，增强大模型的回答能力。这种方法可以有效减少大模型的幻觉问题，并让模型具备访问私有知识的能力。RAG 系统通常包括文档索引、向量检索和答案生成三个主要组件。

向量数据库是 RAG 系统的核心基础设施。它将文本转换为高维向量进行存储和检索。常用的向量数据库包括 ChromaDB、Milvus、Pinecone 等。向量数据库支持相似度搜索，能够根据语义找到最相关的文档片段。在企业应用中，向量数据库需要支持大规模数据、高并发查询和数据持久化等特性。

Python 是人工智能领域最常用的编程语言。它拥有丰富的科学计算库，如 NumPy、Pandas、Scikit-learn 等。PyTorch 和 TensorFlow 是两个主流的深度学习框架。FastAPI 是一个现代、快速的 Web 框架，常用于构建 AI 服务的后端 API。在部署方面，Docker 和 Kubernetes 是常用的容器化技术，云服务商如 AWS、Azure、阿里云也提供了专门的 AI 平台。
""".strip()

print(f"\n原始文档长度: {len(long_document)} 字符")
print(f"原始文档段落数: {len([p for p in long_document.split(chr(10)+chr(10)) if p.strip()])}")

# ====================================
# 策略 1：按字符数分割（固定长度）
# ====================================

print("\n" + "=" * 60)
print("策略 1：按字符数分割（固定长度）")
print("=" * 60)

char_splitter = CharacterTextSplitter(
    separator="\n",           # 分隔符
    chunk_size=200,           # 每个 chunk 最大字符数
    chunk_overlap=50,         # 相邻 chunk 重叠字符数
    length_function=len,      # 长度计算函数
)

char_chunks = char_splitter.split_text(long_document)

print(f"chunk_size=200, chunk_overlap=50")
print(f"分割结果: {len(char_chunks)} 个 chunk")
for i, chunk in enumerate(char_chunks[:3], 1):
    print(f"\n--- Chunk {i} ({len(chunk)} 字符) ---")
    print(chunk[:150] + "..." if len(chunk) > 150 else chunk)

if len(char_chunks) > 3:
    print(f"\n... 还有 {len(char_chunks) - 3} 个 chunk")

# ====================================
# 策略 2：递归分割（推荐）
# ====================================

print("\n" + "=" * 60)
print("策略 2：递归分割（推荐）")
print("=" * 60)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    length_function=len,
)

recursive_chunks = recursive_splitter.split_text(long_document)

print(f"chunk_size=200, chunk_overlap=50")
print(f"分隔符优先级: 段落 > 换行 > 句号 > 感叹号 > 问号 > 空格 > 字符")
print(f"分割结果: {len(recursive_chunks)} 个 chunk")

for i, chunk in enumerate(recursive_chunks[:3], 1):
    print(f"\n--- Chunk {i} ({len(chunk)} 字符) ---")
    print(chunk[:150] + "..." if len(chunk) > 150 else chunk)

if len(recursive_chunks) > 3:
    print(f"\n... 还有 {len(recursive_chunks) - 3} 个 chunk")

# ====================================
# 对比不同参数的效果
# ====================================

print("\n" + "=" * 60)
print("参数对比：chunk_size 的影响")
print("=" * 60)

sizes = [100, 200, 500]
for size in sizes:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=min(50, size // 4),
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(long_document)
    print(f"chunk_size={size:3d}, overlap={min(50, size//4):2d} -> {len(chunks):2d} 个 chunk")

print("""
分析：
- chunk_size 越小：chunk 数量越多，检索越精细，但上下文信息越少
- chunk_size 越大：chunk 数量越少，上下文越完整，但可能包含无关信息
- 实际项目中通常选择 200-500 tokens（注意这里是字符数，tokens 通常更少）
""")

# ====================================
# chunk_overlap 的作用
# ====================================

print("\n" + "=" * 60)
print("参数对比：chunk_overlap 的作用")
print("=" * 60)

overlaps = [0, 50, 100]
for overlap in overlaps:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(long_document)
    print(f"chunk_overlap={overlap:3d} -> {len(chunks)} 个 chunk")

print("""
分析：
- chunk_overlap=0：chunk 之间没有重叠，可能切断连贯的语义
- chunk_overlap=50：保持一定连续性，推荐设置
- chunk_overlap=100：重叠更多，chunk 数量增加，成本上升
- overlap 通常设置为 chunk_size 的 10%-25%
""")

# ====================================
# 实战：模拟检索效果对比
# ====================================

print("\n" + "=" * 60)
print("实战：不同分割策略的检索效果对比")
print("=" * 60)

# 模拟一个查询
query = "RAG 技术如何减少幻觉？"

# 不分割（整篇文档作为 1 个 chunk）
no_split = [long_document]

# 递归分割
splitter_200 = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    length_function=len,
)
chunks_200 = splitter_200.split_text(long_document)

splitter_500 = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    length_function=len,
)
chunks_500 = splitter_500.split_text(long_document)

print(f"查询: {query}")
print(f"\n策略对比:")
print(f"  不分割: 1 个 chunk, 长度 {len(no_split[0])} 字符")
print(f"  chunk=200: {len(chunks_200)} 个 chunk")
print(f"  chunk=500: {len(chunks_500)} 个 chunk")

print("\n检索结果模拟（假设基于语义相似度）:")
print(f"  不分割: 检索到整篇文档，包含大量无关信息")
print(f"  chunk=200: 精准检索到 RAG 相关段落，上下文略少")
print(f"  chunk=500: 检索到包含 RAG 段落的 chunk，上下文适中")

# ====================================
# 从文件加载并分割
# ====================================

print("\n" + "=" * 60)
print("从文件加载并分割")
print("=" * 60)

# 创建一个示例文件
sample_file = "sample_doc.txt"
with open(sample_file, "w", encoding="utf-8") as f:
    f.write(long_document)

# 读取并分割
with open(sample_file, "r", encoding="utf-8") as f:
    file_content = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    length_function=len,
)
file_chunks = splitter.split_text(file_content)

print(f"文件: {sample_file}")
print(f"文件大小: {os.path.getsize(sample_file)} 字节")
print(f"分割后: {len(file_chunks)} 个 chunk")

# 清理
os.remove(sample_file)

# ====================================
# 总结
# ====================================

print("\n" + "=" * 60)
print("总结：文档分割最佳实践")
print("=" * 60)
print("""
1. 推荐使用 RecursiveCharacterTextSplitter
   - 优先保持语义边界（段落、句子）
   - 避免在单词中间切断

2. chunk_size 选择
   - 通用场景: 200-500 tokens
   - 代码文档: 300-800 tokens
   - 短问答对: 100-200 tokens

3. chunk_overlap 设置
   - 推荐: chunk_size 的 10%-25%
   - 保证上下文连续性
   - 避免信息丢失

4. 实际项目流程
   文件加载 → 文档分割 → 生成 Embedding → 存入向量数据库

下一步：将文档分割与 FAISS 结合，构建完整的文档处理流水线！
""")
