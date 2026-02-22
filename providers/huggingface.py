"""
HuggingFace 推理 API 提供商
利用 HuggingFace 的免费推理端点
"""

import aiohttp
import json
from typing import AsyncGenerator, List, Dict
from providers.base import BaseProvider


class HuggingFace(BaseProvider):
    name = "HuggingFace"
    label = "HuggingFace 推理"
    url = "https://api-inference.huggingface.co"
    working = True
    needs_auth = False
    supports_stream = True

    models = [
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2-72B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2.5",
        "Qwen/Qwen2.5-1M",
        "Qwen/Qwen2.5-Max",
        "Qwen/Qwen3-235B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-0.6B",
        "Qwen/QwQ-32B",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "google/gemma-2-9b-it",
        "microsoft/Phi-3.5-mini-instruct",
        "microsoft/phi-4",
        "microsoft/Phi-4-multimodal-instruct",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/Janus-Pro-7B",
        "CohereForAI/c4ai-command-r-v01",
        "CohereForAI/c4ai-command-r-plus",
        "CohereForAI/c4ai-command-r7b-12-2024",
        "CohereForAI/command-a",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "openai/GPT-OSS-120B",
        "moonshotai/Kimi-K2-Instruct",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-3.5-large",
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-schnell",
        "black-forest-labs/FLUX.1",
    ]
    default_model = "meta-llama/Llama-3.3-70B-Instruct"

    model_aliases = {
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "llama-3.2-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
        "qwen-2-72b": "Qwen/Qwen2-72B-Instruct",
        "qwen-2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen-2.5": "Qwen/Qwen2.5",
        "qwen-2.5-1m": "Qwen/Qwen2.5-1M",
        "qwen-2.5-max": "Qwen/Qwen2.5-Max",
        "qwen-3-235b": "Qwen/Qwen3-235B",
        "qwen-3-32b": "Qwen/Qwen3-32B",
        "qwen-3-30b": "Qwen/Qwen3-30B-A3B",
        "qwen-3-14b": "Qwen/Qwen3-14B",
        "qwen-3-4b": "Qwen/Qwen3-4B",
        "qwen-3-1.7b": "Qwen/Qwen3-1.7B",
        "qwen-3-0.6b": "Qwen/Qwen3-0.6B",
        "qwq-32b": "Qwen/QwQ-32B",
        "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
        "gemma-2-9b": "google/gemma-2-9b-it",
        "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
        "phi-4": "microsoft/phi-4",
        "phi-4-multimodal": "microsoft/Phi-4-multimodal-instruct",
        "deepseek-r1": "deepseek-ai/DeepSeek-R1",
        "deepseek-r1-distill-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "janus-pro-7b": "deepseek-ai/Janus-Pro-7B",
        "command-r": "CohereForAI/c4ai-command-r-v01",
        "command-r-plus": "CohereForAI/c4ai-command-r-plus",
        "command-r7b": "CohereForAI/c4ai-command-r7b-12-2024",
        "command-a": "CohereForAI/command-a",
        "nemotron-70b": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "gpt-oss-120b": "openai/GPT-OSS-120B",
        "kimi-k2": "moonshotai/Kimi-K2-Instruct",
        "sdxl-turbo": "stabilityai/stable-diffusion-xl-base-1.0",
        "sd-3.5-large": "stabilityai/stable-diffusion-3.5-large",
        "flux-dev": "black-forest-labs/FLUX.1-dev",
        "flux-schnell": "black-forest-labs/FLUX.1-schnell",
        "flux": "black-forest-labs/FLUX.1",
        "qwen-2.5-coder-32b": "Qwen/Qwen2.5-72B-Instruct",
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
        api_url = f"{cls.url}/models/{model}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
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
                    raise Exception(f"HuggingFace API 错误: {response.status}")

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
