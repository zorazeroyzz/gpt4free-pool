"""
Together AI 提供商
提供免费层的推理服务
"""

import aiohttp
import json
from typing import AsyncGenerator, List, Dict
from providers.base import BaseProvider


class Together(BaseProvider):
    name = "Together"
    label = "Together AI"
    url = "https://api.together.xyz"
    working = True
    needs_auth = False
    supports_stream = True

    models = [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Llama-3-8B-Instruct-Turbo",
        "meta-llama/Llama-3-70B-Instruct-Turbo",
        "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        "meta-llama/Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Llama-3.1-405B-Instruct-Turbo",
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "meta-llama/Llama-2-70B-chat-hf",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2-72B-Instruct",
        "Qwen/Qwen2-VL-72B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "Qwen/Qwen3-235B",
        "Qwen/Qwen3-32B",
        "Qwen/QwQ-32B",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "google/gemma-2b-it",
        "google/gemma-2-27b-it",
        "google/gemma-3-27b-it",
        "google/gemma-3n-e4b-it",
        "NousResearch/Hermes-2-Llama-3-DPO",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct",
        "Perplexity/r1-1776",
        "openai/GPT-OSS-120B",
        "black-forest-labs/FLUX.1-schnell",
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-pro",
        "black-forest-labs/FLUX.1",
        "black-forest-labs/FLUX.1-redux",
        "black-forest-labs/FLUX.1-depth",
        "black-forest-labs/FLUX.1-canny",
        "black-forest-labs/FLUX.1-kontext-dev",
        "black-forest-labs/FLUX.1-dev-lora",
    ]
    default_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    model_aliases = {
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "llama-3-8b": "meta-llama/Llama-3-8B-Instruct-Turbo",
        "llama-3-70b": "meta-llama/Llama-3-70B-Instruct-Turbo",
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct-Turbo",
        "llama-3.1-405b": "meta-llama/Llama-3.1-405B-Instruct-Turbo",
        "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "llama-3.2-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        "llama-3.2-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "llama-4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "llama-2-70b": "meta-llama/Llama-2-70B-chat-hf",
        "deepseek-v3": "deepseek-ai/DeepSeek-V3",
        "deepseek-r1": "deepseek-ai/DeepSeek-R1",
        "deepseek-r1-distill-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-r1-distill-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "qwen-2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "qwen-2-72b": "Qwen/Qwen2-72B-Instruct",
        "qwen-2-vl-72b": "Qwen/Qwen2-VL-72B-Instruct",
        "qwen-2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
        "qwen-3-235b": "Qwen/Qwen3-235B",
        "qwen-3-32b": "Qwen/Qwen3-32B",
        "qwq-32b": "Qwen/QwQ-32B",
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
        "gemma-2b": "google/gemma-2b-it",
        "gemma-2-27b": "google/gemma-2-27b-it",
        "gemma-3-27b": "google/gemma-3-27b-it",
        "gemma-3n-e4b": "google/gemma-3n-e4b-it",
        "hermes-2-dpo": "NousResearch/Hermes-2-Llama-3-DPO",
        "nemotron-70b": "nvidia/Llama-3.1-Nemotron-70B-Instruct",
        "r1-1776": "Perplexity/r1-1776",
        "gpt-oss-120b": "openai/GPT-OSS-120B",
        "flux-schnell": "black-forest-labs/FLUX.1-schnell",
        "flux-dev": "black-forest-labs/FLUX.1-dev",
        "flux-pro": "black-forest-labs/FLUX.1-pro",
        "flux": "black-forest-labs/FLUX.1",
        "flux-redux": "black-forest-labs/FLUX.1-redux",
        "flux-depth": "black-forest-labs/FLUX.1-depth",
        "flux-canny": "black-forest-labs/FLUX.1-canny",
        "flux-kontext": "black-forest-labs/FLUX.1-kontext-dev",
        "flux-dev-lora": "black-forest-labs/FLUX.1-dev-lora",
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
        api_url = f"{cls.url}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status != 200:
                    raise Exception(f"Together API 错误: {response.status}")

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
