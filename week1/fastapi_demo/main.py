from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import os
from zhipuai import ZhipuAI

# 加载环境变量
load_dotenv()

app = FastAPI(title='我的第一个AI服务')

# 初始化智谱客户端
client = ZhipuAI(api_key=os.getenv('ZHIPU_API_KEY'))


class Query(BaseModel):
    question: str


@app.get("/")
async def root():
    return {"message": "服务已启动"}

# 普通接口（保留对比用）
@app.post("/chat")
async def chat(query: Query):
    # 调用智谱大模型
    response = client.chat.completions.create(
        model='glm-4-flash',  # 使用免费的flash模型
        messages=[
            {'role': 'user', 'content': query.question}
        ]
    )
    return {'answer': response.choices[0].message.content}

# 流式接口（新增）
@app.post('/chat/stream')
async def chat_stream(query:Query):
    def generate():
        # 开启流失模式
        response = client.chat.completions.create(
            model='glm-4-flash',
            messages=[{'role':'user','content':query.question}],
            stream=True # 关键：开启流式
        )

        # 逐块返回
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    # 返回流式响应
    return StreamingResponse(
        generate(),
        media_type='text/event-stream'
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
