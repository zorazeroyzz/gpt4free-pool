"""
FreeGPT / ChatGPT 免费镜像提供商
利用各种免费的 ChatGPT 镜像站点
"""

import aiohttp
import json
import random
from typing import AsyncGenerator, List, Dict
from providers.base import BaseProvider


class FreeGPTMirror(BaseProvider):
    name = "FreeGPTMirror"
    label = "免费GPT镜像"
    url = "https://chatgpt.es"
    working = True
    needs_auth = False
    supports_stream = True

    models = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4",
    ]
    default_model = "gpt-4o-mini"

    model_aliases = {
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o",
        "gpt-4": "gpt-4",
    }

    # 可用的镜像端点列表
    _mirrors = [
        "https://chatgpt.es/wp-json/mwai-ui/v1/chats/submit",
    ]

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
        # 从消息列表提取最后一条用户消息
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            user_message = messages[-1].get("content", "")

        mirror_url = random.choice(cls._mirrors)

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Origin": "https://chatgpt.es",
            "Referer": "https://chatgpt.es/chat/",
        }

        payload = {
            "botId": "default",
            "newMessage": user_message,
            "stream": True,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(mirror_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status != 200:
                    raise Exception(f"FreeGPT镜像 API 错误: {response.status}")

                buffer = ""
                async for chunk_bytes in response.content.iter_any():
                    text = chunk_bytes.decode("utf-8", errors="ignore")
                    buffer += text
                    # 尝试解析每一行
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                return
                            try:
                                parsed = json.loads(data)
                                content = parsed.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                if data and data != "[DONE]":
                                    yield data
                        elif line and not line.startswith("{"):
                            pass  # skip non-data lines

                # 处理剩余buffer
                if buffer.strip():
                    try:
                        parsed = json.loads(buffer)
                        content = parsed.get("data", buffer)
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        # 可能是纯文本响应
                        clean = buffer.replace("data: ", "").replace("[DONE]", "").strip()
                        if clean:
                            yield clean
