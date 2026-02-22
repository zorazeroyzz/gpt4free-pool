"""
DeepInfra 提供商
提供多种开源模型的免费推理
"""

import aiohttp
import json
from typing import AsyncGenerator, List, Dict
from providers.base import BaseProvider


class DeepInfra(BaseProvider):
    name = "DeepInfra"
    label = "DeepInfra"
    url = "https://api.deepinfra.com"
    working = True
    needs_auth = False
    supports_stream = True

    models = [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-V3-0324",
        "deepseek-ai/DeepSeek-V3-0324-Turbo",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-Turbo",
        "deepseek-ai/DeepSeek-R1-0528",
        "deepseek-ai/DeepSeek-R1-0528-Turbo",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-Prover-V2",
        "deepseek-ai/DeepSeek-Prover-V2-671B",
        "deepseek-ai/Janus-Pro-7B",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen3-235B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-14B",
        "Qwen/QwQ-32B",
        "mistralai/Mistral-Small-3.1-24B-Instruct",
        "google/codegemma-7b-it",
        "google/gemma-1.1-7b-it",
        "google/gemma-2-9b-it",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "microsoft/phi-4",
        "microsoft/Phi-4-multimodal-instruct",
        "microsoft/Phi-4-reasoning-plus",
        "microsoft/WizardLM-2-7B",
        "microsoft/WizardLM-2-8x22B",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct",
        "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
        "cognitivecomputations/dolphin-2.9-llama3-8b",
        "deepinfra/airoboros-70b",
        "lizpreciatior/lzlv_70b_fp16_hf",
        "moonshotai/Kimi-K2-Instruct",
    ]
    default_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    model_aliases = {
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "llama-3.2-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "llama-4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "deepseek-v3": "deepseek-ai/DeepSeek-V3",
        "deepseek-v3-0324": "deepseek-ai/DeepSeek-V3-0324",
        "deepseek-v3-0324-turbo": "deepseek-ai/DeepSeek-V3-0324-Turbo",
        "deepseek-r1": "deepseek-ai/DeepSeek-R1",
        "deepseek-r1-turbo": "deepseek-ai/DeepSeek-R1-Turbo",
        "deepseek-r1-0528": "deepseek-ai/DeepSeek-R1-0528",
        "deepseek-r1-0528-turbo": "deepseek-ai/DeepSeek-R1-0528-Turbo",
        "deepseek-r1-distill-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-r1-distill-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-prover-v2": "deepseek-ai/DeepSeek-Prover-V2",
        "deepseek-prover-v2-671b": "deepseek-ai/DeepSeek-Prover-V2-671B",
        "janus-pro-7b": "deepseek-ai/Janus-Pro-7B",
        "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
        "qwen-3-235b": "Qwen/Qwen3-235B",
        "qwen-3-32b": "Qwen/Qwen3-32B",
        "qwen-3-30b": "Qwen/Qwen3-30B-A3B",
        "qwen-3-14b": "Qwen/Qwen3-14B",
        "qwq-32b": "Qwen/QwQ-32B",
        "mistral-small-3.1-24b": "mistralai/Mistral-Small-3.1-24B-Instruct",
        "codegemma-7b": "google/codegemma-7b-it",
        "gemma-1.1-7b": "google/gemma-1.1-7b-it",
        "gemma-2-9b": "google/gemma-2-9b-it",
        "gemma-3-4b": "google/gemma-3-4b-it",
        "gemma-3-12b": "google/gemma-3-12b-it",
        "gemma-3-27b": "google/gemma-3-27b-it",
        "phi-4": "microsoft/phi-4",
        "phi-4-multimodal": "microsoft/Phi-4-multimodal-instruct",
        "phi-4-reasoning-plus": "microsoft/Phi-4-reasoning-plus",
        "wizardlm-2-7b": "microsoft/WizardLM-2-7B",
        "wizardlm-2-8x22b": "microsoft/WizardLM-2-8x22B",
        "nemotron-70b": "nvidia/Llama-3.1-Nemotron-70B-Instruct",
        "dolphin-2.6": "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
        "dolphin-2.9": "cognitivecomputations/dolphin-2.9-llama3-8b",
        "airoboros-70b": "deepinfra/airoboros-70b",
        "lzlv-70b": "lizpreciatior/lzlv_70b_fp16_hf",
        "kimi-k2": "moonshotai/Kimi-K2-Instruct",
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
        api_url = f"{cls.url}/v1/openai/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
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
                    raise Exception(f"DeepInfra API 错误: {response.status}")

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
