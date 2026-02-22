"""
Provider 基类定义
所有AI提供商必须继承此基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional, List, Dict, Any
from enum import Enum
import time
import asyncio


class ProviderStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


@dataclass
class ProviderStats:
    """提供商统计数据"""
    total_requests: int = 0
    success_count: int = 0
    error_count: int = 0
    total_tokens: int = 0
    avg_latency_ms: float = 0.0
    last_used: float = 0.0
    last_error: str = ""
    last_error_time: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.success_count / self.total_requests * 100


@dataclass
class ModelInfo:
    """模型信息"""
    id: str
    name: str
    provider: str
    max_tokens: int = 4096
    supports_stream: bool = True
    supports_vision: bool = False
    supports_tools: bool = False


class BaseProvider(ABC):
    """
    AI提供商基类
    所有提供商必须实现 create_completion 方法
    """

    # 提供商元数据
    name: str = "BaseProvider"
    label: str = "基础提供商"
    url: str = ""
    working: bool = True
    needs_auth: bool = False
    supports_stream: bool = True
    supports_system_message: bool = True

    # 模型列表
    models: List[str] = []
    default_model: str = ""
    model_aliases: Dict[str, str] = {}

    # 状态和统计
    status: ProviderStatus = ProviderStatus.ACTIVE
    stats: ProviderStats = field(default_factory=ProviderStats)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'stats') or not isinstance(cls.__dict__.get('stats'), ProviderStats):
            cls.stats = ProviderStats()

    @classmethod
    def get_model(cls, model: str) -> str:
        """解析模型名称，支持别名"""
        if model in cls.model_aliases:
            return cls.model_aliases[model]
        if model in cls.models:
            return model
        if cls.default_model:
            return cls.default_model
        raise ValueError(f"提供商 {cls.name} 不支持模型: {model}")

    @classmethod
    @abstractmethod
    async def create_completion(
        cls,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        创建对话补全
        :param model: 模型名称
        :param messages: 消息列表
        :param stream: 是否流式输出
        :param temperature: 温度参数
        :param max_tokens: 最大token数
        :yields: 文本片段
        """
        raise NotImplementedError

    @classmethod
    async def check_health(cls) -> bool:
        """检查提供商是否可用"""
        try:
            response_text = ""
            async for chunk in cls.create_completion(
                model=cls.default_model,
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                max_tokens=10,
            ):
                response_text += chunk
            return len(response_text) > 0
        except Exception:
            return False

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """获取提供商信息"""
        return {
            "name": cls.name,
            "label": cls.label,
            "url": cls.url,
            "working": cls.working,
            "needs_auth": cls.needs_auth,
            "supports_stream": cls.supports_stream,
            "models": cls.models,
            "default_model": cls.default_model,
            "status": cls.status.value if isinstance(cls.status, ProviderStatus) else cls.status,
            "stats": {
                "total_requests": cls.stats.total_requests,
                "success_count": cls.stats.success_count,
                "error_count": cls.stats.error_count,
                "success_rate": f"{cls.stats.success_rate:.1f}%",
                "avg_latency_ms": f"{cls.stats.avg_latency_ms:.0f}",
                "total_tokens": cls.stats.total_tokens,
            },
        }

    @classmethod
    def record_success(cls, latency_ms: float, tokens: int = 0):
        """记录成功请求"""
        cls.stats.total_requests += 1
        cls.stats.success_count += 1
        cls.stats.total_tokens += tokens
        cls.stats.last_used = time.time()
        # 滑动平均延迟
        if cls.stats.avg_latency_ms == 0:
            cls.stats.avg_latency_ms = latency_ms
        else:
            cls.stats.avg_latency_ms = cls.stats.avg_latency_ms * 0.8 + latency_ms * 0.2

    @classmethod
    def record_error(cls, error: str):
        """记录失败请求"""
        cls.stats.total_requests += 1
        cls.stats.error_count += 1
        cls.stats.last_error = error
        cls.stats.last_error_time = time.time()


class IterListProvider:
    """
    提供商轮询列表
    按顺序尝试每个提供商，直到成功
    """

    def __init__(self, providers: List[type], shuffle: bool = True):
        self.providers = providers
        self.shuffle = shuffle

    async def create_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = True,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        import random

        providers = list(self.providers)
        if self.shuffle:
            random.shuffle(providers)

        errors = {}
        for provider in providers:
            if not provider.working:
                continue
            if isinstance(provider.status, ProviderStatus) and provider.status == ProviderStatus.INACTIVE:
                continue

            try:
                resolved_model = provider.get_model(model)
                start_time = time.time()
                got_content = False

                async for chunk in provider.create_completion(
                    model=resolved_model,
                    messages=messages,
                    stream=stream,
                    **kwargs,
                ):
                    got_content = True
                    yield chunk

                if got_content:
                    latency = (time.time() - start_time) * 1000
                    provider.record_success(latency)
                    return

            except Exception as e:
                error_msg = str(e)
                errors[provider.name] = error_msg
                provider.record_error(error_msg)
                continue

        if errors:
            raise Exception(f"所有提供商均失败: {errors}")
        else:
            raise Exception("没有可用的提供商")
