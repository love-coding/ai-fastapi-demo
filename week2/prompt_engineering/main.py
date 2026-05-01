"""
Prompt Engineering 实战模块
==============================
功能：演示高质量的提示词工程技巧
重点：System Prompt、结构化输出、Few-shot示例
作者：AI学习项目
日期：2026-05-01
"""

# ====================导入依赖====================
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import os
from zhipuai import ZhipuAI
import json

# ==================初始化配置=====================
load_dotenv()
app = FastAPI(title="Prompt Engineering 实战")
client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

# ==================数据模型定义====================
class ChatRequest(BaseModel):
    """
    聊天请求模型
    Attributes:
        question: 用户问题
        context：可选的上下文信息（如知识库内容）
    """
    question: str
    context: str = "" # 默认为空，后续RAG会用到

# =================Prompt 模板定义==================
# System Prompt :定义 AI 的角色、行为规则、输出格式
SYSTEM_PROMPT = """
你是一个专业的知识库问答助手。

## 角色定义
- 名称：小智
- 职业：企业知识库助手
- 特点：专业、准确、友好

## 行为准则
1. 只根据提供的上下文回答问题
2. 如果上下文中没有相关信息，明确告知用户“知识库中没有相关信息”
3. 回答时注明信息来源（引用原文）
4. 不要编造或推测答案

## 输出格式
请按以下 JSON 格式返回：
{
    "answer":"你的回答",
    "source":"引用的原文片段",
    "confidence":"置信度(高/中/低)"
}

## 示例
用户问：公司请假流程是什么？
上下文：员工请假需提前3天在OA系统提交申请，经部门主管审批后生效。

回答：
{
    "answer":"员工请假需提前3天在OA系统提交申请，经部门主管审批后生效。",
    "source":"员工请假需提前3天在OA系统提交申请，经部门主管审批后生效。",
    "confidence":"高"
}
"""

# Few-shot 示例：让 AI 学习如何回答
FEW_SHOT_EXAMPLES = [
    {
        "role":"user",
        "content":"问题：公司几点上班？\n上下文：公司上班时间为早上9:00，下班时间为下午6:00."
    },
    {
        "role":"assistant",
        "content":'{"answer":"公司上班时间为早上9:00。", "source":"公司上班时间为早上9:00","confidence":"高"}'
    }
]

# ==========================辅助函数========================
def build_messages(question:str,context:str) -> list:
    """
    构建完整的消息列表

    Args：
        question:用户问题
        context:上下文信息
    Returns:
         list:完整的消息列表，包含 system prompt、few-shot 示例、当前问题

    消息结构说明：
        1. system:设定 AI 角色和规则
        2. few-shot:示例对话，让 AI 学习回答模式
        3. user：当前用户问题
    """
    messages = [
        # 第一条：System Prompt,定义角色和行为
        {"role":"system","content":SYSTEM_PROMPT},
    ]
    # 添加 Few-shot 示例
    messages.extend(FEW_SHOT_EXAMPLES)
    #构建用户问题（包含上下文）
    user_content = f"问题：{question}\n"
    if context:
        user_content += f"上下文：{context}"
    else:
        user_content += "上下文：（无）"
    messages.append({"role":"user","content":user_content})
    return messages

# ========================API 接口定义=======================
@app.post("/chat")
async def chat(request: ChatRequest):
    """
    智能问答接口
    功能：基于 Prompt Engineering 实现高质量问答
    Args：
        request: ChatRequest 对象，包含问题和可选上下文
    Returns:
        dict: 结构化的回答，包含答案、来源、置信度
    """
    # 构建消息列表
    messages = build_messages(request.question,request.context)

    # 调用大模型
    response = client.chat.completions.create(
        model='glm-4-flash',
        messages=messages
    )

    # 获取回答内容
    content = response.choices[0].message.content

    # 尝试解析 JSON 格式的回答
    try:
        answer_json = json.loads(content)
        return answer_json
    except json.JSONDecodeError:
        # 如果 AI 没有返回标准JSON，包装一下返回
        return {
            "answer":content,
            "source":"",
            "confidence":"未知"
        }

@app.post("/chat/stream")
async def chat_stream(request:ChatRequest):
    """
    流式回答接口
    功能：以流式方式返回问题结果
    """
    messages = build_messages(request.question,request.context)

    def generate():
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=messages,
            stream=True
        )
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    return StreamingResponse(generate(),media_type="text/event-stream")

# =========================启动服务==========================
if __name__ == "__main__":
    import uvicorn
    # 使用 8002端口
    uvicorn.run(app, host='0.0.0.0',port=8002)