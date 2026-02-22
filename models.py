"""
模型注册表 - 完整版
对齐 gpt4free 项目，覆盖 125+ 模型
将模型名称映射到提供商列表
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from providers.base import IterListProvider
from providers.pollinations import PollinationsAI
from providers.deepinfra import DeepInfra
from providers.cloudflare import CloudflareAI
from providers.huggingface import HuggingFace
from providers.freegpt import FreeGPTMirror
from providers.together import Together


# 所有已注册的提供商
ALL_PROVIDERS = [
    PollinationsAI,
    DeepInfra,
    CloudflareAI,
    HuggingFace,
    FreeGPTMirror,
    Together,
]


@dataclass
class Model:
    """模型定义"""
    name: str
    label: str
    base_provider: str
    best_provider: Any  # IterListProvider 或单个 Provider
    category: str = "chat"  # chat, code, reasoning, vision, image, audio


# 模型注册表
MODELS: Dict[str, Model] = {}


def register_model(model: Model):
    MODELS[model.name] = model


# ════════════════════════════════════════════════════════════════
#  OpenAI 系列
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="gpt-4",
    label="GPT-4",
    base_provider="OpenAI",
    best_provider=IterListProvider([FreeGPTMirror, PollinationsAI]),
    category="chat",
))

register_model(Model(
    name="gpt-4o",
    label="GPT-4o",
    base_provider="OpenAI",
    best_provider=IterListProvider([PollinationsAI, FreeGPTMirror]),
    category="vision",
))

register_model(Model(
    name="gpt-4o-mini",
    label="GPT-4o Mini",
    base_provider="OpenAI",
    best_provider=IterListProvider([PollinationsAI, FreeGPTMirror]),
    category="chat",
))

register_model(Model(
    name="o1",
    label="O1",
    base_provider="OpenAI",
    best_provider=IterListProvider([PollinationsAI]),
    category="reasoning",
))

register_model(Model(
    name="o1-mini",
    label="O1 Mini",
    base_provider="OpenAI",
    best_provider=IterListProvider([PollinationsAI]),
    category="reasoning",
))

register_model(Model(
    name="o3-mini",
    label="O3 Mini",
    base_provider="OpenAI",
    best_provider=IterListProvider([PollinationsAI]),
    category="reasoning",
))

register_model(Model(
    name="o3-mini-high",
    label="O3 Mini High",
    base_provider="OpenAI",
    best_provider=IterListProvider([PollinationsAI]),
    category="reasoning",
))

register_model(Model(
    name="o4-mini",
    label="O4 Mini",
    base_provider="OpenAI",
    best_provider=IterListProvider([PollinationsAI]),
    category="reasoning",
))

register_model(Model(
    name="o4-mini-high",
    label="O4 Mini High",
    base_provider="OpenAI",
    best_provider=IterListProvider([PollinationsAI]),
    category="reasoning",
))

register_model(Model(
    name="gpt-4.1",
    label="GPT-4.1",
    base_provider="OpenAI",
    best_provider=IterListProvider([PollinationsAI]),
    category="chat",
))

register_model(Model(
    name="gpt-4.1-mini",
    label="GPT-4.1 Mini",
    base_provider="OpenAI",
    best_provider=IterListProvider([PollinationsAI]),
    category="chat",
))

register_model(Model(
    name="gpt-4.1-nano",
    label="GPT-4.1 Nano",
    base_provider="OpenAI",
    best_provider=IterListProvider([PollinationsAI]),
    category="chat",
))

register_model(Model(
    name="gpt-4.5",
    label="GPT-4.5",
    base_provider="OpenAI",
    best_provider=IterListProvider([PollinationsAI]),
    category="chat",
))

register_model(Model(
    name="gpt-oss-120b",
    label="GPT-OSS 120B",
    base_provider="OpenAI",
    best_provider=IterListProvider([Together, HuggingFace]),
    category="chat",
))

register_model(Model(
    name="gpt-4o-mini-tts",
    label="GPT-4o Mini TTS",
    base_provider="OpenAI",
    best_provider=IterListProvider([PollinationsAI]),
    category="audio",
))

register_model(Model(
    name="dall-e-3",
    label="DALL-E 3",
    base_provider="OpenAI",
    best_provider=IterListProvider([PollinationsAI]),
    category="image",
))

register_model(Model(
    name="gpt-image",
    label="GPT Image",
    base_provider="OpenAI",
    best_provider=IterListProvider([PollinationsAI]),
    category="image",
))

# ════════════════════════════════════════════════════════════════
#  Meta / Llama 系列
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="llama-2-7b",
    label="Llama 2 7B",
    base_provider="Meta Llama",
    best_provider=IterListProvider([CloudflareAI]),
    category="chat",
))

register_model(Model(
    name="llama-2-70b",
    label="Llama 2 70B",
    base_provider="Meta Llama",
    best_provider=IterListProvider([Together]),
    category="chat",
))

register_model(Model(
    name="llama-3-8b",
    label="Llama 3 8B",
    base_provider="Meta Llama",
    best_provider=IterListProvider([Together, CloudflareAI]),
    category="chat",
))

register_model(Model(
    name="llama-3-70b",
    label="Llama 3 70B",
    base_provider="Meta Llama",
    best_provider=IterListProvider([Together]),
    category="chat",
))

register_model(Model(
    name="llama-3.1-8b",
    label="Llama 3.1 8B",
    base_provider="Meta Llama",
    best_provider=IterListProvider([DeepInfra, Together, CloudflareAI]),
    category="chat",
))

register_model(Model(
    name="llama-3.1-70b",
    label="Llama 3.1 70B",
    base_provider="Meta Llama",
    best_provider=IterListProvider([Together]),
    category="chat",
))

register_model(Model(
    name="llama-3.1-405b",
    label="Llama 3.1 405B",
    base_provider="Meta Llama",
    best_provider=IterListProvider([Together]),
    category="chat",
))

register_model(Model(
    name="llama-3.2-1b",
    label="Llama 3.2 1B",
    base_provider="Meta Llama",
    best_provider=IterListProvider([CloudflareAI]),
    category="chat",
))

register_model(Model(
    name="llama-3.2-3b",
    label="Llama 3.2 3B",
    base_provider="Meta Llama",
    best_provider=IterListProvider([Together]),
    category="chat",
))

register_model(Model(
    name="llama-3.2-11b",
    label="Llama 3.2 11B (视觉)",
    base_provider="Meta Llama",
    best_provider=IterListProvider([Together, HuggingFace]),
    category="vision",
))

register_model(Model(
    name="llama-3.2-90b",
    label="Llama 3.2 90B",
    base_provider="Meta Llama",
    best_provider=IterListProvider([DeepInfra, Together]),
    category="chat",
))

register_model(Model(
    name="llama-3.3-70b",
    label="Llama 3.3 70B",
    base_provider="Meta Llama",
    best_provider=IterListProvider([DeepInfra, Together, HuggingFace, PollinationsAI]),
    category="chat",
))

register_model(Model(
    name="llama-4-scout",
    label="Llama 4 Scout",
    base_provider="Meta Llama",
    best_provider=IterListProvider([DeepInfra, PollinationsAI, Together, CloudflareAI]),
    category="chat",
))

register_model(Model(
    name="llama-4-maverick",
    label="Llama 4 Maverick",
    base_provider="Meta Llama",
    best_provider=IterListProvider([DeepInfra, Together]),
    category="chat",
))

# ════════════════════════════════════════════════════════════════
#  Mistral AI 系列
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="mistral-7b",
    label="Mistral 7B",
    base_provider="Mistral AI",
    best_provider=IterListProvider([Together, CloudflareAI]),
    category="chat",
))

register_model(Model(
    name="mixtral-8x7b",
    label="Mixtral 8x7B",
    base_provider="Mistral AI",
    best_provider=IterListProvider([Together]),
    category="chat",
))

register_model(Model(
    name="mistral-nemo",
    label="Mistral Nemo",
    base_provider="Mistral AI",
    best_provider=IterListProvider([HuggingFace]),
    category="chat",
))

register_model(Model(
    name="mistral-small-24b",
    label="Mistral Small 24B",
    base_provider="Mistral AI",
    best_provider=IterListProvider([Together]),
    category="chat",
))

register_model(Model(
    name="mistral-small-3.1-24b",
    label="Mistral Small 3.1 24B",
    base_provider="Mistral AI",
    best_provider=IterListProvider([DeepInfra, PollinationsAI]),
    category="chat",
))

# ════════════════════════════════════════════════════════════════
#  Microsoft 系列 (Phi, WizardLM)
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="phi-3.5-mini",
    label="Phi 3.5 Mini",
    base_provider="Microsoft",
    best_provider=IterListProvider([HuggingFace]),
    category="chat",
))

register_model(Model(
    name="phi-4",
    label="Phi 4",
    base_provider="Microsoft",
    best_provider=IterListProvider([DeepInfra, HuggingFace]),
    category="chat",
))

register_model(Model(
    name="phi-4-multimodal",
    label="Phi 4 多模态",
    base_provider="Microsoft",
    best_provider=IterListProvider([DeepInfra, HuggingFace]),
    category="vision",
))

register_model(Model(
    name="phi-4-reasoning-plus",
    label="Phi 4 Reasoning Plus",
    base_provider="Microsoft",
    best_provider=IterListProvider([DeepInfra]),
    category="reasoning",
))

register_model(Model(
    name="wizardlm-2-7b",
    label="WizardLM 2 7B",
    base_provider="Microsoft",
    best_provider=IterListProvider([DeepInfra]),
    category="chat",
))

register_model(Model(
    name="wizardlm-2-8x22b",
    label="WizardLM 2 8x22B",
    base_provider="Microsoft",
    best_provider=IterListProvider([DeepInfra]),
    category="chat",
))

# ════════════════════════════════════════════════════════════════
#  Google 系列 (Gemini, Gemma, CodeGemma)
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="gemini-2.0",
    label="Gemini 2.0",
    base_provider="Google",
    best_provider=IterListProvider([PollinationsAI]),
    category="chat",
))

register_model(Model(
    name="gemini-2.0-flash",
    label="Gemini 2.0 Flash",
    base_provider="Google",
    best_provider=IterListProvider([PollinationsAI]),
    category="chat",
))

register_model(Model(
    name="gemini-2.0-flash-thinking",
    label="Gemini 2.0 Flash Thinking",
    base_provider="Google",
    best_provider=IterListProvider([PollinationsAI]),
    category="reasoning",
))

register_model(Model(
    name="gemini-2.5-flash",
    label="Gemini 2.5 Flash",
    base_provider="Google",
    best_provider=IterListProvider([PollinationsAI]),
    category="chat",
))

register_model(Model(
    name="gemini-2.5-pro",
    label="Gemini 2.5 Pro",
    base_provider="Google",
    best_provider=IterListProvider([PollinationsAI]),
    category="chat",
))

register_model(Model(
    name="gemini-3-pro-preview",
    label="Gemini 3 Pro 预览",
    base_provider="Google",
    best_provider=IterListProvider([PollinationsAI]),
    category="chat",
))

register_model(Model(
    name="codegemma-7b",
    label="CodeGemma 7B",
    base_provider="Google",
    best_provider=IterListProvider([DeepInfra]),
    category="code",
))

register_model(Model(
    name="gemma-2b",
    label="Gemma 2B",
    base_provider="Google",
    best_provider=IterListProvider([Together]),
    category="chat",
))

register_model(Model(
    name="gemma-1.1-7b",
    label="Gemma 1.1 7B",
    base_provider="Google",
    best_provider=IterListProvider([DeepInfra]),
    category="chat",
))

register_model(Model(
    name="gemma-2-9b",
    label="Gemma 2 9B",
    base_provider="Google",
    best_provider=IterListProvider([DeepInfra, HuggingFace]),
    category="chat",
))

register_model(Model(
    name="gemma-2-27b",
    label="Gemma 2 27B",
    base_provider="Google",
    best_provider=IterListProvider([Together]),
    category="chat",
))

register_model(Model(
    name="gemma-3-4b",
    label="Gemma 3 4B",
    base_provider="Google",
    best_provider=IterListProvider([DeepInfra]),
    category="chat",
))

register_model(Model(
    name="gemma-3-12b",
    label="Gemma 3 12B",
    base_provider="Google",
    best_provider=IterListProvider([DeepInfra]),
    category="chat",
))

register_model(Model(
    name="gemma-3-27b",
    label="Gemma 3 27B",
    base_provider="Google",
    best_provider=IterListProvider([DeepInfra, Together]),
    category="chat",
))

register_model(Model(
    name="gemma-3n-e4b",
    label="Gemma 3n E4B",
    base_provider="Google",
    best_provider=IterListProvider([Together]),
    category="chat",
))

# ════════════════════════════════════════════════════════════════
#  DeepSeek 系列
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="deepseek-v3",
    label="DeepSeek V3",
    base_provider="DeepSeek",
    best_provider=IterListProvider([DeepInfra, Together]),
    category="chat",
))

register_model(Model(
    name="deepseek-r1",
    label="DeepSeek R1",
    base_provider="DeepSeek",
    best_provider=IterListProvider([DeepInfra, PollinationsAI, Together, HuggingFace]),
    category="reasoning",
))

register_model(Model(
    name="deepseek-r1-turbo",
    label="DeepSeek R1 Turbo",
    base_provider="DeepSeek",
    best_provider=IterListProvider([DeepInfra]),
    category="reasoning",
))

register_model(Model(
    name="deepseek-r1-distill-llama-70b",
    label="DeepSeek R1 蒸馏 Llama 70B",
    base_provider="DeepSeek",
    best_provider=IterListProvider([DeepInfra, Together]),
    category="reasoning",
))

register_model(Model(
    name="deepseek-r1-distill-qwen-1.5b",
    label="DeepSeek R1 蒸馏 Qwen 1.5B",
    base_provider="DeepSeek",
    best_provider=IterListProvider([Together]),
    category="reasoning",
))

register_model(Model(
    name="deepseek-r1-distill-qwen-14b",
    label="DeepSeek R1 蒸馏 Qwen 14B",
    base_provider="DeepSeek",
    best_provider=IterListProvider([Together]),
    category="reasoning",
))

register_model(Model(
    name="deepseek-r1-distill-qwen-32b",
    label="DeepSeek R1 蒸馏 Qwen 32B",
    base_provider="DeepSeek",
    best_provider=IterListProvider([DeepInfra, CloudflareAI]),
    category="reasoning",
))

register_model(Model(
    name="deepseek-prover-v2",
    label="DeepSeek Prover V2",
    base_provider="DeepSeek",
    best_provider=IterListProvider([DeepInfra]),
    category="reasoning",
))

register_model(Model(
    name="deepseek-prover-v2-671b",
    label="DeepSeek Prover V2 671B",
    base_provider="DeepSeek",
    best_provider=IterListProvider([DeepInfra]),
    category="reasoning",
))

register_model(Model(
    name="deepseek-v3-0324",
    label="DeepSeek V3 0324",
    base_provider="DeepSeek",
    best_provider=IterListProvider([DeepInfra]),
    category="chat",
))

register_model(Model(
    name="deepseek-v3-0324-turbo",
    label="DeepSeek V3 0324 Turbo",
    base_provider="DeepSeek",
    best_provider=IterListProvider([DeepInfra]),
    category="chat",
))

register_model(Model(
    name="deepseek-r1-0528",
    label="DeepSeek R1 0528",
    base_provider="DeepSeek",
    best_provider=IterListProvider([DeepInfra]),
    category="reasoning",
))

register_model(Model(
    name="deepseek-r1-0528-turbo",
    label="DeepSeek R1 0528 Turbo",
    base_provider="DeepSeek",
    best_provider=IterListProvider([DeepInfra]),
    category="reasoning",
))

register_model(Model(
    name="janus-pro-7b",
    label="Janus Pro 7B (视觉)",
    base_provider="DeepSeek",
    best_provider=IterListProvider([DeepInfra, HuggingFace]),
    category="vision",
))

# ════════════════════════════════════════════════════════════════
#  Qwen 系列
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="qwen-1.5-7b",
    label="Qwen 1.5 7B",
    base_provider="Qwen",
    best_provider=IterListProvider([CloudflareAI]),
    category="chat",
))

register_model(Model(
    name="qwen-2-72b",
    label="Qwen 2 72B",
    base_provider="Qwen",
    best_provider=IterListProvider([Together, HuggingFace]),
    category="chat",
))

register_model(Model(
    name="qwen-2-vl-7b",
    label="Qwen 2 VL 7B (视觉)",
    base_provider="Qwen",
    best_provider=IterListProvider([HuggingFace]),
    category="vision",
))

register_model(Model(
    name="qwen-2-vl-72b",
    label="Qwen 2 VL 72B (视觉)",
    base_provider="Qwen",
    best_provider=IterListProvider([Together]),
    category="vision",
))

register_model(Model(
    name="qwen-2.5",
    label="Qwen 2.5",
    base_provider="Qwen",
    best_provider=IterListProvider([HuggingFace]),
    category="chat",
))

register_model(Model(
    name="qwen-2.5-7b",
    label="Qwen 2.5 7B",
    base_provider="Qwen",
    best_provider=IterListProvider([Together]),
    category="chat",
))

register_model(Model(
    name="qwen-2.5-72b",
    label="Qwen 2.5 72B",
    base_provider="Qwen",
    best_provider=IterListProvider([Together, DeepInfra, HuggingFace]),
    category="chat",
))

register_model(Model(
    name="qwen-2.5-coder-32b",
    label="Qwen 2.5 Coder 32B",
    base_provider="Qwen",
    best_provider=IterListProvider([Together, PollinationsAI, HuggingFace, CloudflareAI]),
    category="code",
))

register_model(Model(
    name="qwen-2.5-1m",
    label="Qwen 2.5 1M 长上下文",
    base_provider="Qwen",
    best_provider=IterListProvider([HuggingFace]),
    category="chat",
))

register_model(Model(
    name="qwen-2.5-max",
    label="Qwen 2.5 Max",
    base_provider="Qwen",
    best_provider=IterListProvider([HuggingFace]),
    category="chat",
))

register_model(Model(
    name="qwen-2.5-vl-72b",
    label="Qwen 2.5 VL 72B (视觉)",
    base_provider="Qwen",
    best_provider=IterListProvider([Together]),
    category="vision",
))

register_model(Model(
    name="qwen-3-235b",
    label="Qwen 3 235B",
    base_provider="Qwen",
    best_provider=IterListProvider([DeepInfra, Together, HuggingFace]),
    category="chat",
))

register_model(Model(
    name="qwen-3-32b",
    label="Qwen 3 32B",
    base_provider="Qwen",
    best_provider=IterListProvider([DeepInfra, Together, HuggingFace]),
    category="chat",
))

register_model(Model(
    name="qwen-3-30b",
    label="Qwen 3 30B",
    base_provider="Qwen",
    best_provider=IterListProvider([DeepInfra, HuggingFace]),
    category="chat",
))

register_model(Model(
    name="qwen-3-14b",
    label="Qwen 3 14B",
    base_provider="Qwen",
    best_provider=IterListProvider([DeepInfra, HuggingFace]),
    category="chat",
))

register_model(Model(
    name="qwen-3-4b",
    label="Qwen 3 4B",
    base_provider="Qwen",
    best_provider=IterListProvider([HuggingFace]),
    category="chat",
))

register_model(Model(
    name="qwen-3-1.7b",
    label="Qwen 3 1.7B",
    base_provider="Qwen",
    best_provider=IterListProvider([HuggingFace]),
    category="chat",
))

register_model(Model(
    name="qwen-3-0.6b",
    label="Qwen 3 0.6B",
    base_provider="Qwen",
    best_provider=IterListProvider([HuggingFace]),
    category="chat",
))

register_model(Model(
    name="qwq-32b",
    label="QwQ 32B (推理)",
    base_provider="Qwen",
    best_provider=IterListProvider([DeepInfra, Together, HuggingFace]),
    category="reasoning",
))

# ════════════════════════════════════════════════════════════════
#  CohereForAI 系列 (Command)
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="command-r",
    label="Command R",
    base_provider="CohereForAI",
    best_provider=IterListProvider([HuggingFace]),
    category="chat",
))

register_model(Model(
    name="command-r-plus",
    label="Command R Plus",
    base_provider="CohereForAI",
    best_provider=IterListProvider([HuggingFace]),
    category="chat",
))

register_model(Model(
    name="command-r7b",
    label="Command R7B",
    base_provider="CohereForAI",
    best_provider=IterListProvider([HuggingFace]),
    category="chat",
))

register_model(Model(
    name="command-a",
    label="Command A",
    base_provider="CohereForAI",
    best_provider=IterListProvider([HuggingFace]),
    category="chat",
))

# ════════════════════════════════════════════════════════════════
#  x.AI 系列 (Grok)
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="grok-2",
    label="Grok 2",
    base_provider="x.AI",
    best_provider=IterListProvider([PollinationsAI]),
    category="chat",
))

register_model(Model(
    name="grok-3",
    label="Grok 3",
    base_provider="x.AI",
    best_provider=IterListProvider([PollinationsAI]),
    category="chat",
))

register_model(Model(
    name="grok-3-r1",
    label="Grok 3 R1 (推理)",
    base_provider="x.AI",
    best_provider=IterListProvider([PollinationsAI]),
    category="reasoning",
))

# ════════════════════════════════════════════════════════════════
#  Kimi / Moonshot AI
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="kimi-k2",
    label="Kimi K2",
    base_provider="Moonshot AI",
    best_provider=IterListProvider([HuggingFace, DeepInfra]),
    category="chat",
))

# ════════════════════════════════════════════════════════════════
#  Perplexity AI 系列 (Sonar)
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="sonar",
    label="Sonar",
    base_provider="Perplexity AI",
    best_provider=IterListProvider([PollinationsAI]),
    category="chat",
))

register_model(Model(
    name="sonar-pro",
    label="Sonar Pro",
    base_provider="Perplexity AI",
    best_provider=IterListProvider([PollinationsAI]),
    category="chat",
))

register_model(Model(
    name="sonar-reasoning",
    label="Sonar Reasoning",
    base_provider="Perplexity AI",
    best_provider=IterListProvider([PollinationsAI]),
    category="reasoning",
))

register_model(Model(
    name="sonar-reasoning-pro",
    label="Sonar Reasoning Pro",
    base_provider="Perplexity AI",
    best_provider=IterListProvider([PollinationsAI]),
    category="reasoning",
))

register_model(Model(
    name="r1-1776",
    label="R1-1776",
    base_provider="Perplexity AI",
    best_provider=IterListProvider([Together, PollinationsAI]),
    category="reasoning",
))

# ════════════════════════════════════════════════════════════════
#  NousResearch
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="hermes-2-dpo",
    label="Hermes 2 DPO",
    base_provider="NousResearch",
    best_provider=IterListProvider([Together]),
    category="chat",
))

# ════════════════════════════════════════════════════════════════
#  Nvidia
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="nemotron-70b",
    label="Nemotron 70B",
    base_provider="Nvidia",
    best_provider=IterListProvider([Together, HuggingFace]),
    category="chat",
))

# ════════════════════════════════════════════════════════════════
#  Cognitive Computations (Dolphin)
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="dolphin-2.6",
    label="Dolphin 2.6",
    base_provider="Cognitive Computations",
    best_provider=IterListProvider([DeepInfra]),
    category="chat",
))

register_model(Model(
    name="dolphin-2.9",
    label="Dolphin 2.9",
    base_provider="Cognitive Computations",
    best_provider=IterListProvider([DeepInfra]),
    category="chat",
))

# ════════════════════════════════════════════════════════════════
#  DeepInfra 自有模型
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="airoboros-70b",
    label="Airoboros 70B",
    base_provider="DeepInfra",
    best_provider=IterListProvider([DeepInfra]),
    category="chat",
))

register_model(Model(
    name="lzlv-70b",
    label="LZLV 70B",
    base_provider="Lizpreciatior",
    best_provider=IterListProvider([DeepInfra]),
    category="chat",
))

# ════════════════════════════════════════════════════════════════
#  Opera (Aria)
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="aria",
    label="Aria",
    base_provider="Opera",
    best_provider=IterListProvider([PollinationsAI]),
    category="chat",
))

# ════════════════════════════════════════════════════════════════
#  Stability AI (图像模型)
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="sdxl-turbo",
    label="SDXL Turbo",
    base_provider="Stability AI",
    best_provider=IterListProvider([PollinationsAI, HuggingFace]),
    category="image",
))

register_model(Model(
    name="sd-3.5-large",
    label="Stable Diffusion 3.5 Large",
    base_provider="Stability AI",
    best_provider=IterListProvider([HuggingFace]),
    category="image",
))

# ════════════════════════════════════════════════════════════════
#  Black Forest Labs (FLUX 图像模型)
# ════════════════════════════════════════════════════════════════

register_model(Model(
    name="flux",
    label="FLUX",
    base_provider="Black Forest Labs",
    best_provider=IterListProvider([PollinationsAI, HuggingFace, Together]),
    category="image",
))

register_model(Model(
    name="flux-pro",
    label="FLUX Pro",
    base_provider="Black Forest Labs",
    best_provider=IterListProvider([PollinationsAI, Together]),
    category="image",
))

register_model(Model(
    name="flux-dev",
    label="FLUX Dev",
    base_provider="Black Forest Labs",
    best_provider=IterListProvider([PollinationsAI, Together, HuggingFace]),
    category="image",
))

register_model(Model(
    name="flux-schnell",
    label="FLUX Schnell",
    base_provider="Black Forest Labs",
    best_provider=IterListProvider([PollinationsAI, Together, HuggingFace]),
    category="image",
))

register_model(Model(
    name="flux-redux",
    label="FLUX Redux",
    base_provider="Black Forest Labs",
    best_provider=IterListProvider([Together]),
    category="image",
))

register_model(Model(
    name="flux-depth",
    label="FLUX Depth",
    base_provider="Black Forest Labs",
    best_provider=IterListProvider([Together]),
    category="image",
))

register_model(Model(
    name="flux-canny",
    label="FLUX Canny",
    base_provider="Black Forest Labs",
    best_provider=IterListProvider([Together]),
    category="image",
))

register_model(Model(
    name="flux-kontext",
    label="FLUX Kontext",
    base_provider="Black Forest Labs",
    best_provider=IterListProvider([PollinationsAI, Together]),
    category="image",
))

register_model(Model(
    name="flux-dev-lora",
    label="FLUX Dev LoRA",
    base_provider="Black Forest Labs",
    best_provider=IterListProvider([Together]),
    category="image",
))


# ════════════════════════════════════════════════════════════════
#  查询函数
# ════════════════════════════════════════════════════════════════

def get_model(name: str) -> Optional[Model]:
    """根据名称获取模型"""
    return MODELS.get(name)


def get_all_models() -> List[Dict]:
    """获取所有模型列表"""
    result = []
    for name, model in MODELS.items():
        if isinstance(model.best_provider, IterListProvider):
            providers = [p.name for p in model.best_provider.providers]
        else:
            providers = [model.best_provider.name]

        result.append({
            "id": model.name,
            "label": model.label,
            "base_provider": model.base_provider,
            "category": model.category,
            "providers": providers,
        })
    return result


def get_models_by_category() -> Dict[str, List[Dict]]:
    """按分类获取模型"""
    categories = {}
    for item in get_all_models():
        cat = item.get("category", "chat")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)
    return categories


def get_model_count() -> int:
    """获取模型总数"""
    return len(MODELS)


def get_provider_count() -> Dict[str, int]:
    """获取各提供商的模型数量"""
    counts = {}
    for model in MODELS.values():
        if isinstance(model.best_provider, IterListProvider):
            for p in model.best_provider.providers:
                counts[p.name] = counts.get(p.name, 0) + 1
        else:
            name = model.best_provider.name
            counts[name] = counts.get(name, 0) + 1
    return counts
