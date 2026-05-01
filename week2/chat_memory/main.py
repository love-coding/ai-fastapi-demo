"""
多轮对话服务模块
==============
功能：实现基于大模型的多轮对话，支持上下文记忆
作者：AI学习项目
日期：2026-05-01
"""

# =================================导入依赖=====================================
from fastapi import FastAPI                         # web框架
from pydantic import BaseModel                      # 数据验证
from fastapi.responses import StreamingResponse     # 流式响应
from dotenv import load_dotenv                      # 环境变量加载
import os                                           # 操作系统接口
from zhipuai import ZhipuAI                         # 智谱AI SDK

# ================================初始化配置====================================
# 加载 .env 文件中的环境变量
load_dotenv()

# 创建 FastAPI 应用实例
app = FastAPI(title='多轮对话服务')

# 初始化智谱 AI 客户端，从环境变量读取 API KEY
client = ZhipuAI(api_key=os.getenv('ZHIPU_API_KEY'))

# ================================数据模型定义==================================
class Message(BaseModel):
    """
    单条消息模型

    Attributes:
        role:消息角色，可选值：
            - “user”：用户消息
            - “assistant”:AI 助手消息
            - “system”：系统提示消息
        content：消息内容
    """
    role: str
    content: str

class ChatRequest(BaseModel):
    """
    聊天请求模型

    Attributes:
        messages: 完整的对话历史列表，按时间顺序排列

    Example:
        {
        "messages":[
        {"role": "user","content":"你好"},
        {"role": "assistant", "content":"你好!有什么可以帮助你的？"}，
        {"role": "user", "content":"天气怎么样？"}
        ]
        }
    """
    messages: list[Message]

    # ===================API 接口定义==================================
@app.post("/chat")
async def chat(request: ChatRequest):
    """
    普通聊天接口（非流式）
    功能：接收完整对话历史，返回 AI 回答
    Args：
        request: ChatRequest 对象，包含对话历史

    Returns：
        dict：包含AI回答的字典{"answer":"回答内容"}

    工作流程：
        1. 接收前端传来的对话历史
        2. 转换为大模型需要的格式
        3. 调用智谱 API 获取回答
        4. 返回回答内容
    """
    # 将 Pydantic 模型转换为大模型需要的字典格式
    messages = [{"role":msg.role,"content":msg.content} for msg in request.messages]

    # 调用智谱大模型 API
    response = client.chat.completions.create(
        model='glm-4-flash', # 使用免费的 flash 模型，速度快
        messages=messages    # 传入完整对话历史
    )

    # 提取并返回 AI 的回答
    return {"answer":response.choices[0].message.content}

@app.post('/chat/stream')
async def chat_stream(request: ChatRequest):
    """
    流式聊天接口
    功能：以流式方式返回 AI 回答，实现打字机效果
    Args：
        request: ChatRequest 对象，包含对话历史
    Returns:
        StreamingResponse:流式响应，逐字返回内容
    工作流程：
        1. 接收对话历史
        2. 开启流式模式调用API
        3. 逐块返回内容给前端
    """
    # 转换消息格式
    messages = [{"role":msg.role,"content":msg.content} for msg in request.messages]

    # 定义生成器函数，用于流式输出
    def generate():
        """
        流式生成器
        逐块获取大模型返回的内容并 yield 给客户端
        """
        # 开启流式模式调用 API
        response = client.chat.completions.create(
            model='glm-4-flash',
            messages=messages,
            stream=True #关键字参数：开启流式输出
        )
        # 遍历每个返回块
        for chunk in response:
            # 提取当前块的内容
            content = chunk.choices[0].delta.content
            # 如果有内容这返回：
            if content:
                yield content
    # 返回流式响应
    # media_type="text/event-stream" 是 SSE（Server-sent Events）的标准格式
    return StreamingResponse(generate(), media_type="text/event-stream")

# ===========================启动服务=============================
if __name__ == "__main__":
    import uvicorn
    # 启动 web 服务器
    # host="0.0.0.0":允许外部访问
    # port=8001:使用 8001 端口，避免与 week1 的 8000端口冲突
    uvicorn.run(app,host='0.0.0.0',port=8001)