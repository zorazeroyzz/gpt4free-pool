# FreeAPI Pool - 免费API流量池

类似 gpt4free 的免费 AI API 聚合服务，提供 OpenAI 兼容接口和中文 Web UI 管理界面。

## 功能特性

- **OpenAI 兼容 API** - 直接使用 OpenAI SDK 或任何兼容客户端
- **多提供商聚合** - 集成 PollinationsAI、DeepInfra、Cloudflare、HuggingFace、Together 等
- **自动故障转移** - 提供商不可用时自动切换到下一个
- **中文 Web UI** - 完整的管理界面，包含控制面板、提供商管理、模型列表、对话测试
- **流式输出** - 支持 SSE 流式响应
- **12+ 模型** - 覆盖 GPT、Llama、DeepSeek、Qwen、Mistral、Gemma 等

## 快速启动

### 方式一：直接运行

```bash
pip install -r requirements.txt
python main.py
```

### 方式二：Docker

```bash
docker compose up -d
```

启动后访问：
- Web UI: http://localhost:8080
- API: http://localhost:8080/v1
- Swagger 文档: http://localhost:8080/docs

## API 使用

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="free"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "你好！"}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### cURL

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "你好！"}],
    "stream": false
  }'
```

## 可用模型

| 模型 | 分类 | 提供商 |
|------|------|--------|
| gpt-4o-mini | 对话 | PollinationsAI, FreeGPTMirror |
| gpt-4o | 对话 | PollinationsAI, FreeGPTMirror |
| llama-3.3-70b | 对话 | DeepInfra, PollinationsAI, Together, HuggingFace, Cloudflare |
| deepseek-r1 | 推理 | PollinationsAI, DeepInfra, Together, HuggingFace |
| deepseek-chat | 对话 | PollinationsAI |
| qwen-2.5-72b | 对话 | DeepInfra, HuggingFace, Together |
| qwen-coder-32b | 代码 | PollinationsAI, Cloudflare |
| mistral-small | 对话 | PollinationsAI, DeepInfra, Together, HuggingFace |
| gemma-2-27b | 对话 | DeepInfra, Together |
| phi-3.5-mini | 对话 | HuggingFace |

## 项目结构

```
free-api-pool/
├── main.py                 # 主入口
├── models.py               # 模型注册表
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── providers/              # AI 提供商
│   ├── base.py             # 基类 + 轮询逻辑
│   ├── pollinations.py     # Pollinations AI
│   ├── deepinfra.py        # DeepInfra
│   ├── cloudflare.py       # Cloudflare Workers AI
│   ├── huggingface.py      # HuggingFace 推理
│   ├── freegpt.py          # 免费 GPT 镜像
│   └── together.py         # Together AI
├── api/
│   └── routes.py           # FastAPI 路由
└── gui/
    ├── templates/
    │   └── index.html       # Web UI
    └── static/
        ├── css/style.css
        └── js/app.js
```

## 添加新提供商

继承 `BaseProvider` 并实现 `create_completion` 方法：

```python
from providers.base import BaseProvider

class MyProvider(BaseProvider):
    name = "MyProvider"
    label = "我的提供商"
    url = "https://api.example.com"
    working = True
    models = ["model-a", "model-b"]
    default_model = "model-a"

    @classmethod
    async def create_completion(cls, model, messages, stream=True, **kwargs):
        # 实现 API 调用逻辑
        yield "Hello!"
```

然后在 `models.py` 中注册即可。

## 许可证

MIT License
