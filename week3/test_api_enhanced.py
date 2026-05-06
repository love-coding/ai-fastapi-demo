"""
Week 3 - 增强版 API 测试脚本
===========================
功能：测试 FAISS + 文档分割 + 文件上传的增强版 API
"""

import os
import json
import requests

BASE_URL = "http://localhost:8002"


def print_response(response):
    """美化打印响应"""
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


def test_health():
    """测试健康检查"""
    print("=" * 60)
    print("测试健康检查")
    print("=" * 60)
    response = requests.get(f"{BASE_URL}/health")
    print_response(response)


def test_add_documents():
    """测试添加文档"""
    print("\n" + "=" * 60)
    print("测试添加文档")
    print("=" * 60)

    documents = [
        {
            "content": """
产品名称：智能客服机器人 Pro
功能：7x24 小时自动回答客户咨询，支持多轮对话
特点：
- 支持微信、网站、APP 等多渠道接入
- 自动学习常见问题，持续优化回答质量
- 支持人工无缝转接
- 响应时间 < 1 秒
价格：基础版 999 元/月，专业版 2999 元/月，企业版需定制报价
联系方式：400-123-4567，邮箱：support@example.com
            """.strip(),
            "metadata": {"category": "产品", "type": "智能客服"}
        },
        {
            "content": """
产品名称：智能文档分析系统
功能：自动识别和处理各类文档
特点：
- 支持 PDF、Word、Excel、PPT、图片等 20+ 格式
- 自动提取关键信息、表格数据
- 支持批量处理，单次最多 1000 个文件
- OCR 识别准确率 > 95%
价格：按处理量计费，0.1 元/页，企业客户享受 8 折优惠
            """.strip(),
            "metadata": {"category": "产品", "type": "文档分析"}
        },
        {
            "content": """
售后服务政策：
1. 所有产品提供 7 天免费试用，无需绑定信用卡
2. 工作日 9:00-18:00 提供在线技术支持
3. 专业版及以上客户享有 1 对 1 专属客服
4. 提供完整的 API 文档和 SDK，支持二次开发
5. 数据安全保障：所有数据采用 AES-256 加密存储，通过 ISO27001 认证
6. 支持私有化部署，满足金融、政府等行业的合规要求
            """.strip(),
            "metadata": {"category": "服务政策"}
        }
    ]

    response = requests.post(f"{BASE_URL}/documents", json=documents)
    print_response(response)


def test_query():
    """测试问答"""
    print("\n" + "=" * 60)
    print("测试问答接口")
    print("=" * 60)

    questions = [
        "智能客服机器人的价格是多少？",
        "我可以试用吗？有什么条件？",
        "文档分析系统支持哪些格式？",
        "如何联系你们？",
        "数据安全怎么保障？",
        "支持私有化部署吗？",
    ]

    for question in questions:
        print(f"\n{'-' * 60}")
        print(f"问题: {question}")
        response = requests.post(
            f"{BASE_URL}/query",
            json={"question": question, "top_k": 3}
        )
        result = response.json()
        print(f"回答: {result['answer']}")
        print(f"\n参考来源:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  [{i}] 相似度: {source['similarity']:.4f}")
            meta = source.get('metadata', {})
            if meta:
                print(f"      元数据: {meta}")


def test_upload_txt():
    """测试上传 TXT 文件"""
    print("\n" + "=" * 60)
    print("测试上传 TXT 文件")
    print("=" * 60)

    # 创建测试文件
    test_file = "test_upload.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("""
智能写作助手产品说明

智能写作助手是一款基于大语言模型的 AI 写作工具，能够帮助用户快速生成高质量文案。

核心功能：
1. 多文体支持：新闻稿、营销文案、公文、小说、论文等多种文体
2. 智能改写：对已有文案进行润色、扩写、缩写、风格转换
3. 团队协作：支持多人在线协作编辑，实时保存
4. 语法检查：自动检测错别字、语法错误、标点问题
5. SEO 优化：自动生成关键词、标题建议、Meta 描述

适用场景：
- 新媒体运营：日更公众号、小红书、微博文案
- 电商运营：商品标题、详情页、促销文案
- 企业行政：通知公告、会议纪要、工作报告
- 学术研究：论文摘要、文献综述、实验报告

价格方案：
- 个人版：49 元/月，每月 10 万字生成额度
- 团队版：199 元/月/人，每月 50 万字生成额度，支持 5 人以上团队
- 企业版：定制报价，无限额度，私有化部署选项
        """.strip())

    with open(test_file, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/upload",
            files={"file": ("test_upload.txt", f, "text/plain")}
        )
    print_response(response)

    # 清理
    os.remove(test_file)

    # 测试上传后的问答
    print("\n测试文件上传后的问答:")
    response = requests.post(
        f"{BASE_URL}/query",
        json={"question": "智能写作助手支持哪些文体？", "top_k": 3}
    )
    result = response.json()
    print(f"回答: {result['answer']}")


def test_list_documents():
    """测试查看文档列表"""
    print("\n" + "=" * 60)
    print("测试查看文档列表")
    print("=" * 60)
    response = requests.get(f"{BASE_URL}/documents")
    print_response(response)


def test_clear():
    """测试清空知识库"""
    print("\n" + "=" * 60)
    print("测试清空知识库")
    print("=" * 60)
    response = requests.delete(f"{BASE_URL}/documents")
    print_response(response)


def test_all():
    """运行全部测试"""
    print("\n" + "=" * 60)
    print("增强版 API 完整测试")
    print("=" * 60)

    try:
        test_health()
        test_add_documents()
        test_query()
        test_upload_txt()
        test_list_documents()
        # test_clear()  # 默认不执行清空
    except requests.exceptions.ConnectionError:
        print("\n错误：无法连接到服务。请确保服务已启动：")
        print("  python week3/main_enhanced.py")


if __name__ == "__main__":
    test_all()
