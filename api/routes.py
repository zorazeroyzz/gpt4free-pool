"""
OpenAI 兼容的 API 接口
实现 /v1/chat/completions, /v1/models 等标准端点
"""

import json
import time
import uuid
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models import MODELS, get_model, get_all_models, ALL_PROVIDERS
from providers.base import IterListProvider


# ============= 请求/响应 Pydantic 模型 =============

class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: List[ChatMessage] = []
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    provider: Optional[str] = None  # 可选：指定提供商


class ChatChoice(BaseModel):
    index: int = 0
    message: Dict[str, str] = {}
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[ChatChoice] = []
    usage: Usage = Usage()


# ============= API 路由 =============

def create_api_app() -> FastAPI:
    """创建 FastAPI 应用"""

    api = FastAPI(
        title="FreeAPI Pool - 免费API流量池",
        description="兼容 OpenAI API 格式的免费 AI 推理服务",
        version="1.0.0",
    )

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/v1/models")
    async def list_models():
        """列出所有可用模型"""
        models_list = []
        for name, model in MODELS.items():
            models_list.append({
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": model.base_provider,
            })
        return {"object": "list", "data": models_list}

    @api.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """对话补全接口 - 兼容 OpenAI 格式"""
        model_name = request.model
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # 查找模型
        model_obj = get_model(model_name)

        if model_obj:
            provider = model_obj.best_provider
        else:
            # 尝试直接查找支持该模型的提供商
            provider = None
            for p in ALL_PROVIDERS:
                if model_name in p.models or model_name in p.model_aliases:
                    provider = p
                    break
            if not provider:
                raise HTTPException(
                    status_code=404,
                    detail=f"模型 '{model_name}' 不存在。请使用 GET /v1/models 查看可用模型列表。"
                )

        # 如果指定了提供商
        if request.provider:
            specific_provider = None
            for p in ALL_PROVIDERS:
                if p.name.lower() == request.provider.lower():
                    specific_provider = p
                    break
            if specific_provider:
                provider = specific_provider

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if request.stream:
            return StreamingResponse(
                _stream_response(
                    provider=provider,
                    model=model_name,
                    messages=messages,
                    completion_id=completion_id,
                    created=created,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return await _non_stream_response(
                provider=provider,
                model=model_name,
                messages=messages,
                completion_id=completion_id,
                created=created,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

    # ---- 提供商管理 API ----

    @api.get("/api/providers")
    async def list_providers():
        """列出所有提供商及其状态"""
        return {"providers": [p.get_info() for p in ALL_PROVIDERS]}

    @api.get("/api/providers/{name}")
    async def get_provider(name: str):
        """获取单个提供商详情"""
        for p in ALL_PROVIDERS:
            if p.name.lower() == name.lower():
                return p.get_info()
        raise HTTPException(status_code=404, detail=f"提供商 '{name}' 未找到")

    @api.post("/api/providers/{name}/toggle")
    async def toggle_provider(name: str):
        """启用/禁用提供商"""
        for p in ALL_PROVIDERS:
            if p.name.lower() == name.lower():
                p.working = not p.working
                return {"name": p.name, "working": p.working, "message": f"提供商 {'已启用' if p.working else '已禁用'}"}
        raise HTTPException(status_code=404, detail=f"提供商 '{name}' 未找到")

    @api.post("/api/providers/{name}/health")
    async def check_provider_health(name: str):
        """检查提供商健康状态"""
        for p in ALL_PROVIDERS:
            if p.name.lower() == name.lower():
                healthy = await p.check_health()
                return {"name": p.name, "healthy": healthy}
        raise HTTPException(status_code=404, detail=f"提供商 '{name}' 未找到")

    @api.get("/api/models")
    async def api_list_models():
        """列出所有模型（管理API）"""
        from models import get_models_by_category
        return {
            "models": get_all_models(),
            "by_category": get_models_by_category(),
        }

    @api.get("/api/stats")
    async def get_stats():
        """获取总体统计数据"""
        total_requests = sum(p.stats.total_requests for p in ALL_PROVIDERS)
        total_success = sum(p.stats.success_count for p in ALL_PROVIDERS)
        total_errors = sum(p.stats.error_count for p in ALL_PROVIDERS)
        total_tokens = sum(p.stats.total_tokens for p in ALL_PROVIDERS)
        active_providers = sum(1 for p in ALL_PROVIDERS if p.working)

        return {
            "total_requests": total_requests,
            "total_success": total_success,
            "total_errors": total_errors,
            "success_rate": f"{(total_success / total_requests * 100) if total_requests > 0 else 0:.1f}%",
            "total_tokens": total_tokens,
            "total_models": len(MODELS),
            "active_providers": active_providers,
            "total_providers": len(ALL_PROVIDERS),
        }

    @api.post("/api/chat/test")
    async def test_chat(request: ChatCompletionRequest):
        """测试对话接口（用于 Web UI 中的聊天测试）"""
        model_name = request.model
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        model_obj = get_model(model_name)
        if model_obj:
            provider = model_obj.best_provider
        else:
            provider = IterListProvider(ALL_PROVIDERS)

        if request.provider:
            for p in ALL_PROVIDERS:
                if p.name.lower() == request.provider.lower():
                    provider = p
                    break

        if request.stream:
            return StreamingResponse(
                _stream_response(
                    provider=provider,
                    model=model_name,
                    messages=messages,
                    completion_id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    created=int(time.time()),
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ),
                media_type="text/event-stream",
            )

        try:
            content = ""
            if isinstance(provider, IterListProvider):
                async for chunk in provider.create_completion(
                    model=model_name, messages=messages, stream=False,
                    temperature=request.temperature, max_tokens=request.max_tokens,
                ):
                    content += chunk
            else:
                resolved = provider.get_model(model_name)
                async for chunk in provider.create_completion(
                    model=resolved, messages=messages, stream=False,
                    temperature=request.temperature, max_tokens=request.max_tokens,
                ):
                    content += chunk

            return {"content": content, "model": model_name, "provider": getattr(provider, 'name', 'auto')}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return api


async def _stream_response(provider, model, messages, completion_id, created, temperature, max_tokens):
    """流式响应生成器"""
    try:
        if isinstance(provider, IterListProvider):
            gen = provider.create_completion(
                model=model, messages=messages, stream=True,
                temperature=temperature, max_tokens=max_tokens,
            )
        else:
            resolved = provider.get_model(model)
            gen = provider.create_completion(
                model=resolved, messages=messages, stream=True,
                temperature=temperature, max_tokens=max_tokens,
            )

        async for chunk in gen:
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        # 发送结束标记
        end_data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(end_data, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_data = {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


async def _non_stream_response(provider, model, messages, completion_id, created, temperature, max_tokens):
    """非流式响应"""
    try:
        content = ""
        if isinstance(provider, IterListProvider):
            async for chunk in provider.create_completion(
                model=model, messages=messages, stream=False,
                temperature=temperature, max_tokens=max_tokens,
            ):
                content += chunk
        else:
            resolved = provider.get_model(model)
            async for chunk in provider.create_completion(
                model=resolved, messages=messages, stream=False,
                temperature=temperature, max_tokens=max_tokens,
            ):
                content += chunk

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": sum(len(m.get("content", "")) for m in messages),
                "completion_tokens": len(content),
                "total_tokens": sum(len(m.get("content", "")) for m in messages) + len(content),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"所有提供商均失败: {str(e)}")
