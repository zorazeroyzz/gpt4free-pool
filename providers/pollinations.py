"""
Pollinations AI 提供商
免费、无需认证的AI接口
支持多种模型
"""

import aiohttp
import json
from typing import AsyncGenerator, List, Dict
from providers.base import BaseProvider


class PollinationsAI(BaseProvider):
    name = "PollinationsAI"
    label = "Pollinations AI"
    url = "https://text.pollinations.ai"
    working = True
    needs_auth = False
    supports_stream = True

    models = [
        "openai",
        "openai-large",
        "openai-reasoning",
        "openai-fast",
        "openai-rp",
        "qwen-coder",
        "llama",
        "llama-scaleway",
        "mistral",
        "mistral-small",
        "deepseek",
        "deepseek-r1",
        "deepseek-reasoner",
        "gemini",
        "gemini-thinking",
        "grok",
        "grok-r1",
        "sonar",
        "sonar-pro",
        "sonar-reasoning",
        "sonar-reasoning-pro",
        "flux",
        "flux-pro",
        "flux-dev",
        "flux-schnell",
        "flux-kontext",
        "dall-e-3",
        "gpt-image",
        "sdxl-turbo",
    ]
    default_model = "openai"

    model_aliases = {
        "gpt-4o-mini": "openai",
        "gpt-4o": "openai-large",
        "gpt-4": "openai",
        "o1": "openai-reasoning",
        "o1-mini": "openai-reasoning",
        "o3-mini": "openai-reasoning",
        "o3-mini-high": "openai-reasoning",
        "o4-mini": "openai-reasoning",
        "o4-mini-high": "openai-reasoning",
        "gpt-4.1": "openai-large",
        "gpt-4.1-mini": "openai-fast",
        "gpt-4.1-nano": "openai-fast",
        "gpt-4.5": "openai-large",
        "gpt-4o-mini-tts": "openai",
        "qwen-2.5-coder-32b": "qwen-coder",
        "llama-3.3-70b": "llama",
        "llama-4-scout": "llama",
        "mistral-small-3.1-24b": "mistral-small",
        "deepseek-chat": "deepseek",
        "deepseek-r1": "deepseek-r1",
        "gemini-2.0": "gemini",
        "gemini-2.0-flash": "gemini",
        "gemini-2.0-flash-thinking": "gemini-thinking",
        "gemini-2.5-flash": "gemini",
        "gemini-2.5-pro": "gemini",
        "gemini-3-pro-preview": "gemini",
        "grok-2": "grok",
        "grok-3": "grok",
        "grok-3-r1": "grok-r1",
        "aria": "openai",
        "sonar": "sonar",
        "sonar-pro": "sonar-pro",
        "sonar-reasoning": "sonar-reasoning",
        "sonar-reasoning-pro": "sonar-reasoning-pro",
        "r1-1776": "deepseek-r1",
        "dall-e-3": "dall-e-3",
        "gpt-image": "gpt-image",
        "flux": "flux",
        "flux-pro": "flux-pro",
        "flux-dev": "flux-dev",
        "flux-schnell": "flux-schnell",
        "flux-kontext": "flux-kontext",
        "sdxl-turbo": "sdxl-turbo",
    }

    @classmethod
    async def create_completion(
        cls,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        api_url = f"{cls.url}/openai/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # 透传 one-api / OpenAI 标准参数
        for key in ("stop", "seed", "tools", "tool_choice", "response_format",
                     "top_p", "frequency_penalty", "presence_penalty"):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status != 200:
                    raise Exception(f"Pollinations API 错误: {response.status}")

                if stream:
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
                else:
                    data = await response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content:
                        yield content
