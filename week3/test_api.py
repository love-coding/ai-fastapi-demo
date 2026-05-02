"""
Week 3 - API 测试脚本
=====================
功能：测试知识库问答 API
"""

import requests
import json

BASE_URL = "http://localhost:8002"

def print_response(response):
    """美化打印响应"""
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))

def test_rag_api():
    """测试 RAG API"""

    print("=" * 60)
    print("测试知识库问答 API")
    print("=" * 60)

    # 1. 添加文档
    print("\n[1] 添加文档到知识库...")
    documents = [
        {
            "content": """
产品名称：智能客服机器人
功能：7x24 小时自动回答客户咨询
特点：
- 支持多轮对话
- 自动学习常见问题
- 可接入微信、网站等渠道
- 支持人工转接
价格：基础版 999 元/月，专业版 2999 元/月
联系方式：400-123-4567
            """.strip()
        },
        {
            "content": """
产品名称：智能文档处理系统
功能：自动识别和处理各类文档
特点：
- 支持 PDF、Word、Excel 等格式
- 自动提取关键信息
- 支持批量处理
- 准确率 > 95%
价格：按处理量计费，0.1 元/页
            """.strip()
        },
        {
            "content": """
售后服务政策：
1. 所有产品提供 7 天免费试用
2. 工作日 9:00-18:00 提供技术支持
3. 专业版客户享有专属客服
4. 提供 API 接口，支持二次开发
5. 数据安全保障：所有数据加密存储
            """.strip()
        }
    ]

    response = requests.post(f"{BASE_URL}/documents", json=documents)
    print_response(response)

    # 2. 查看文档列表
    print("\n[2] 查看知识库文档...")
    response = requests.get(f"{BASE_URL}/documents")
    print_response(response)

    # 3. 问答测试
    questions = [
        "智能客服机器人的价格是多少？",
        "我可以试用吗？",
        "支持哪些文档格式？",
        "如何联系你们？"
    ]

    for question in questions:
        print(f"\n[问答] 问题: {question}")
        response = requests.post(
            f"{BASE_URL}/query",
            json={"question": question, "top_k": 3}
        )
        result = response.json()
        print(f"\n回答: {result['answer']}")
        print("\n参考来源:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  [{i}] 相似度: {source['similarity']:.4f}")
        print("-" * 60)

    print("\n测试完成！")

if __name__ == "__main__":
    test_rag_api()
