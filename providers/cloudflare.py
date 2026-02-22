"""
Cloudflare Workers AI 提供商
基于 Cloudflare 的免费推理端点
"""

import aiohttp
import json
from typing import AsyncGenerator, List, Dict
from providers.base import BaseProvider


class CloudflareAI(BaseProvider):
    name = "CloudflareAI"
    label = "Cloudflare AI"
    url = "https://playground.ai.cloudflare.com"
    working = True
    needs_auth = False
    supports_stream = True

    models = [
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "@cf/meta/llama-3.1-8b-instruct",
        "@cf/meta/llama-3.2-1b-instruct",
        "@cf/meta/llama-4-scout-17b-16e-instruct",
        "@cf/qwen/qwen2.5-coder-32b-instruct",
        "@cf/qwen/qwen-1.5-7b-chat-awq",
        "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        "@cf/google/gemma-7b-it-lora",
        "@hf/mistral/mistral-7b-instruct-v0.2",
    ]
    default_model = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"

    model_aliases = {
        "llama-3.3-70b": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "llama-3.1-8b": "@cf/meta/llama-3.1-8b-instruct",
        "llama-3.2-1b": "@cf/meta/llama-3.2-1b-instruct",
        "llama-4-scout": "@cf/meta/llama-4-scout-17b-16e-instruct",
        "llama-2-7b": "@cf/meta/llama-3.1-8b-instruct",
        "llama-3-8b": "@cf/meta/llama-3.1-8b-instruct",
        "qwen-2.5-coder-32b": "@cf/qwen/qwen2.5-coder-32b-instruct",
        "qwen-1.5-7b": "@cf/qwen/qwen-1.5-7b-chat-awq",
        "deepseek-r1-distill-qwen-32b": "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        "gemma-7b": "@cf/google/gemma-7b-it-lora",
        "mistral-7b": "@hf/mistral/mistral-7b-instruct-v0.2",
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
        api_url = f"{cls.url}/api/inference"

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
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
                    raise Exception(f"Cloudflare API 错误: {response.status}")

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
                            content = chunk.get("response", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
                else:
                    data = await response.json()
                    content = data.get("result", {}).get("response", "")
                    if content:
                        yield content
