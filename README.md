# GPT4Free Pool - 免费 AI API 流量池

> 受 [gpt4free](https://github.com/xtekky/gpt4free) 启发，聚合多个免费 AI 推理端点，对外提供统一的 OpenAI 兼容 API。内置中文 Web UI 管理面板。

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **OpenAI 兼容 API** | 完整实现 `/v1/chat/completions`、`/v1/models`，可直接对接 OpenAI SDK、ChatBox、LobeChat 等客户端 |
| **121 个模型** | 覆盖 GPT-4o、O3/O4、Llama 4、DeepSeek R1、Qwen 3、Gemini 2.5、Grok 3、FLUX 等，横跨对话/推理/代码/视觉/图像/音频 6 大类别 |
| **6 个提供商** | PollinationsAI、DeepInfra、Together AI、HuggingFace、Cloudflare Workers AI、FreeGPT Mirror |
| **自动故障转移** | `IterListProvider` 随机轮询多个提供商，单个失败自动跳到下一个 |
| **流式输出** | 完整 SSE 流式响应，支持逐字打印 |
| **中文 Web UI** | 暗色主题管理面板，含控制台、提供商管理、模型浏览、在线对话测试、API 文档 |
| **零配置** | 无需 API Key，开箱即用 |

---

## 快速启动

### 方式一：直接运行

```bash
git clone https://github.com/zorazeroyzz/gpt4free-pool.git
cd gpt4free-pool
pip install -r requirements.txt
python main.py
```

### 方式二：Docker

```bash
git clone https://github.com/zorazeroyzz/gpt4free-pool.git
cd gpt4free-pool
docker compose up -d
```

启动后：

| 地址 | 说明 |
|------|------|
| `http://localhost:8080` | Web UI 管理面板 |
| `http://localhost:8080/v1` | OpenAI 兼容 API |
| `http://localhost:8080/docs` | Swagger 交互文档 |

---

## API 使用

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="free"  # 任意值即可
)

# 流式对话
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "用 Python 写一个快速排序"}],
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
    "model": "deepseek-r1",
    "messages": [{"role": "user", "content": "解释量子纠缠"}],
    "stream": false
  }'
```

### JavaScript (fetch)

```javascript
const response = await fetch("http://localhost:8080/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "llama-3.3-70b",
    messages: [{ role: "user", content: "Hello!" }],
    stream: false,
  }),
});
const data = await response.json();
console.log(data.choices[0].message.content);
```

### 指定提供商

在请求体中添加 `provider` 字段可强制使用指定提供商：

```json
{
  "model": "deepseek-r1",
  "messages": [{"role": "user", "content": "你好"}],
  "provider": "DeepInfra"
}
```

---

## 对接 one-api（LLM API 网关）

[one-api](https://github.com/songquanpeng/one-api) 是一个流行的 LLM API 管理/分发网关。gpt4free-pool 可以作为 one-api 的**上游渠道（Channel）**，使 one-api 的用户免费使用 121 个模型。

### 方式一：Docker Compose 一键部署

```bash
# 同时启动 gpt4free-pool + one-api
docker compose -f docker-compose.oneapi.yml up -d
```

启动后：

| 服务 | 地址 | 说明 |
|------|------|------|
| one-api 管理面板 | `http://localhost:3000` | 默认账号 `root` / 密码 `123456` |
| gpt4free-pool 面板 | `http://localhost:8080` | 中文 Web UI |
| gpt4free-pool API | `http://freeapi-pool:8080/v1` | Docker 内部网络地址 |

### 方式二：手动对接已有的 one-api

假设 gpt4free-pool 运行在 `http://192.168.1.100:8080`。

### 在 one-api 中添加渠道

1. 登录 one-api 管理面板 -> **渠道管理** -> **添加渠道**
2. 填写以下信息：

| 字段 | 值 |
|------|-----|
| 类型 | **OpenAI** |
| 名称 | `gpt4free-pool`（随意） |
| Base URL | `http://freeapi-pool:8080`（Docker 部署）或 `http://你的IP:8080`（手动部署） |
| 密钥 | `free`（如未设置 API_KEY）或你设置的 API_KEY 值 |
| 模型 | 手动输入需要的模型，如 `gpt-4o-mini,deepseek-r1,llama-3.3-70b,qwen-3-235b` |

3. 保存后在渠道列表点击「测试」验证连通性

### 注意事项

- **密钥**：one-api 会发送 `Authorization: Bearer {key}`。如果 gpt4free-pool 未设置 `API_KEY` 环境变量，任意密钥都会被接受
- **Token 计费**：gpt4free-pool 返回 token 用量为 0，one-api 会自动使用 tiktoken 估算用量进行计费
- **模型名称**：one-api 渠道中的模型名必须与 gpt4free-pool 的模型名完全一致（参见 `/v1/models`）
- **故障转移**：建议在 one-api 中为同一模型同时添加免费渠道和付费渠道，通过优先级实现免费优先、付费兜底

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `API_KEY` | API 访问密钥，未设置则无需认证 | 空（不验证） |
| `FREEAPI_KEY` | docker-compose.oneapi.yml 中使用，同 API_KEY | 空 |
| `ONEAPI_ROOT_TOKEN` | one-api 初始 root 令牌 | 空 |

---

## 支持的模型

共 **121 个模型**，分为 6 个类别：

### 对话模型 (Chat)

| 模型 | 来源 | 提供商 |
|------|------|--------|
| `gpt-4o-mini` | OpenAI | PollinationsAI, FreeGPTMirror |
| `gpt-4o` | OpenAI | PollinationsAI, FreeGPTMirror |
| `gpt-4` | OpenAI | FreeGPTMirror, PollinationsAI |
| `gpt-4.1` / `gpt-4.1-mini` / `gpt-4.1-nano` | OpenAI | PollinationsAI |
| `gpt-4.5` | OpenAI | PollinationsAI |
| `gpt-oss-120b` | OpenAI | Together, HuggingFace |
| `llama-3.3-70b` | Meta | DeepInfra, Together, HuggingFace, PollinationsAI |
| `llama-4-scout` | Meta | DeepInfra, PollinationsAI, Together, Cloudflare |
| `llama-4-maverick` | Meta | DeepInfra, Together |
| `llama-3.1-8b` / `llama-3.1-70b` / `llama-3.1-405b` | Meta | DeepInfra, Together, Cloudflare |
| `llama-3.2-3b` / `llama-3.2-90b` | Meta | Together, DeepInfra |
| `deepseek-v3` | DeepSeek | DeepInfra, Together |
| `deepseek-v3-0324` / `deepseek-v3-0324-turbo` | DeepSeek | DeepInfra |
| `qwen-2.5-72b` | Qwen | Together, DeepInfra, HuggingFace |
| `qwen-3-235b` | Qwen | DeepInfra, Together, HuggingFace |
| `qwen-3-32b` / `qwen-3-30b` / `qwen-3-14b` | Qwen | DeepInfra, Together, HuggingFace |
| `qwen-3-4b` / `qwen-3-1.7b` / `qwen-3-0.6b` | Qwen | HuggingFace |
| `gemini-2.0-flash` / `gemini-2.5-flash` / `gemini-2.5-pro` | Google | PollinationsAI |
| `gemini-3-pro-preview` | Google | PollinationsAI |
| `grok-2` / `grok-3` | x.AI | PollinationsAI |
| `kimi-k2` | Moonshot AI | HuggingFace, DeepInfra |
| `command-r` / `command-r-plus` / `command-r7b` / `command-a` | CohereForAI | HuggingFace |
| `sonar` / `sonar-pro` | Perplexity | PollinationsAI |
| `mistral-7b` / `mixtral-8x7b` / `mistral-small-3.1-24b` | Mistral AI | Together, DeepInfra, Cloudflare |
| `gemma-2-9b` / `gemma-3-27b` / `gemma-3n-e4b` | Google | DeepInfra, Together, HuggingFace |
| `phi-3.5-mini` / `phi-4` | Microsoft | HuggingFace, DeepInfra |
| `wizardlm-2-7b` / `wizardlm-2-8x22b` | Microsoft | DeepInfra |
| `nemotron-70b` | Nvidia | Together, HuggingFace |
| `hermes-2-dpo` | NousResearch | Together |
| `dolphin-2.6` / `dolphin-2.9` | Cognitive Computations | DeepInfra |
| `aria` | Opera | PollinationsAI |

### 推理模型 (Reasoning)

| 模型 | 来源 | 提供商 |
|------|------|--------|
| `o1` / `o1-mini` | OpenAI | PollinationsAI |
| `o3-mini` / `o3-mini-high` | OpenAI | PollinationsAI |
| `o4-mini` / `o4-mini-high` | OpenAI | PollinationsAI |
| `deepseek-r1` | DeepSeek | DeepInfra, PollinationsAI, Together, HuggingFace |
| `deepseek-r1-turbo` | DeepSeek | DeepInfra |
| `deepseek-r1-0528` / `deepseek-r1-0528-turbo` | DeepSeek | DeepInfra |
| `deepseek-r1-distill-llama-70b` | DeepSeek | DeepInfra, Together |
| `deepseek-r1-distill-qwen-32b` | DeepSeek | DeepInfra, Cloudflare |
| `deepseek-r1-distill-qwen-1.5b` / `deepseek-r1-distill-qwen-14b` | DeepSeek | Together |
| `deepseek-prover-v2` / `deepseek-prover-v2-671b` | DeepSeek | DeepInfra |
| `qwq-32b` | Qwen | DeepInfra, Together, HuggingFace |
| `grok-3-r1` | x.AI | PollinationsAI |
| `gemini-2.0-flash-thinking` | Google | PollinationsAI |
| `sonar-reasoning` / `sonar-reasoning-pro` | Perplexity | PollinationsAI |
| `r1-1776` | Perplexity | Together, PollinationsAI |
| `phi-4-reasoning-plus` | Microsoft | DeepInfra |

### 代码模型 (Code)

| 模型 | 来源 | 提供商 |
|------|------|--------|
| `qwen-2.5-coder-32b` | Qwen | Together, PollinationsAI, HuggingFace, Cloudflare |
| `codegemma-7b` | Google | DeepInfra |

### 视觉模型 (Vision)

| 模型 | 来源 | 提供商 |
|------|------|--------|
| `gpt-4o` | OpenAI | PollinationsAI, FreeGPTMirror |
| `llama-3.2-11b` | Meta | Together, HuggingFace |
| `phi-4-multimodal` | Microsoft | DeepInfra, HuggingFace |
| `janus-pro-7b` | DeepSeek | DeepInfra, HuggingFace |
| `qwen-2-vl-7b` | Qwen | HuggingFace |
| `qwen-2-vl-72b` / `qwen-2.5-vl-72b` | Qwen | Together |

### 图像生成 (Image)

| 模型 | 来源 | 提供商 |
|------|------|--------|
| `dall-e-3` / `gpt-image` | OpenAI | PollinationsAI |
| `flux` / `flux-pro` / `flux-dev` / `flux-schnell` | Black Forest Labs | PollinationsAI, Together, HuggingFace |
| `flux-kontext` / `flux-canny` / `flux-depth` / `flux-redux` | Black Forest Labs | Together, PollinationsAI |
| `flux-dev-lora` | Black Forest Labs | Together |
| `sdxl-turbo` | Stability AI | PollinationsAI, HuggingFace |
| `sd-3.5-large` | Stability AI | HuggingFace |

### 音频 (Audio)

| 模型 | 来源 | 提供商 |
|------|------|--------|
| `gpt-4o-mini-tts` | OpenAI | PollinationsAI |

---

## 提供商一览

| 提供商 | 端点 | 认证 | 模型数 |
|--------|------|------|--------|
| **PollinationsAI** | `text.pollinations.ai` | 无需 | 29 |
| **DeepInfra** | `api.deepinfra.com` | 无需 | 41 |
| **Together AI** | `api.together.xyz` | 无需 | 46 |
| **HuggingFace** | `api-inference.huggingface.co` | 无需 | 36 |
| **Cloudflare AI** | `playground.ai.cloudflare.com` | 无需 | 9 |
| **FreeGPT Mirror** | `chatgpt.es` | 无需 | 3 |

---

## 管理 API

除了 OpenAI 兼容接口外，还提供以下管理端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/providers` | 获取所有提供商及其状态 |
| `GET` | `/api/providers/{name}` | 获取单个提供商详情 |
| `POST` | `/api/providers/{name}/toggle` | 启用/禁用提供商 |
| `POST` | `/api/providers/{name}/health` | 检测提供商健康状态 |
| `GET` | `/api/models` | 获取模型列表（含分类） |
| `GET` | `/api/stats` | 获取全局统计数据 |
| `POST` | `/api/chat/test` | Web UI 对话测试接口 |
| `GET` | `/health` | 服务健康检查 |

---

## 项目结构

```
gpt4free-pool/
├── main.py                 # 应用入口，FastAPI 实例创建
├── models.py               # 模型注册表（121 个模型定义）
├── requirements.txt        # Python 依赖
├── Dockerfile              # Docker 镜像构建
├── docker-compose.yml      # Docker Compose 编排
├── providers/              # AI 提供商实现
│   ├── base.py             # BaseProvider 基类 + IterListProvider 轮询
│   ├── pollinations.py     # PollinationsAI (29 模型)
│   ├── deepinfra.py        # DeepInfra (41 模型)
│   ├── together.py         # Together AI (46 模型)
│   ├── huggingface.py      # HuggingFace 推理 (36 模型)
│   ├── cloudflare.py       # Cloudflare Workers AI (9 模型)
│   └── freegpt.py          # FreeGPT Mirror (3 模型)
├── api/
│   └── routes.py           # API 路由（OpenAI 兼容 + 管理接口）
└── gui/
    ├── templates/
    │   └── index.html      # 中文 Web UI（暗色主题）
    └── static/
        ├── css/style.css   # 样式表
        └── js/app.js       # 前端逻辑
```

---

## 架构说明

```
用户请求
  │
  ▼
FastAPI (/v1/chat/completions)
  │
  ▼
模型注册表 (models.py)  ──→  查找模型对应的 IterListProvider
  │
  ▼
IterListProvider (自动轮询)
  │
  ├── 提供商 A (随机选中) ──→ 成功 → 返回结果
  ├── 提供商 B (A 失败后)  ──→ 成功 → 返回结果
  └── 提供商 C (B 失败后)  ──→ 成功 → 返回结果
```

每个提供商继承 `BaseProvider`，实现 `create_completion` 异步生成器方法。`IterListProvider` 在运行时随机打乱提供商顺序，逐个尝试直到成功，实现自动负载均衡和故障转移。

---

## 添加新提供商

1. 在 `providers/` 下创建新文件，继承 `BaseProvider`：

```python
from providers.base import BaseProvider

class MyProvider(BaseProvider):
    name = "MyProvider"
    label = "我的提供商"
    url = "https://api.example.com"
    working = True
    needs_auth = False
    supports_stream = True

    models = ["model-a", "model-b"]
    default_model = "model-a"
    model_aliases = {"short-name": "model-a"}

    @classmethod
    async def create_completion(cls, model, messages, stream=True, **kwargs):
        # 实现 API 调用，通过 yield 返回文本片段
        yield "Hello!"
```

2. 在 `models.py` 中导入并注册模型：

```python
from providers.my_provider import MyProvider

register_model(Model(
    name="short-name",
    label="Short Name",
    base_provider="MyCompany",
    best_provider=IterListProvider([MyProvider]),
    category="chat",
))
```

3. 将提供商加入 `ALL_PROVIDERS` 列表。

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| HTTP 客户端 | aiohttp (异步) |
| 数据校验 | Pydantic v2 |
| 前端 | 原生 HTML/CSS/JS（暗色主题） |
| 容器化 | Docker + Docker Compose |
| Python 版本 | 3.11+ |

---

## 许可证

MIT License
