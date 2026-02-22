"""
可选 API Key 认证中间件
当设置了 API_KEY 环境变量时，校验 /v1/* 请求的 Bearer Token
未设置时，放行所有请求（免费访问模式）
"""

import os
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class OptionalAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = os.environ.get("API_KEY", "")

        # 仅校验 /v1/ API 端点，不影响 WebUI 和管理接口
        if api_key and request.url.path.startswith("/v1/"):
            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "message": "Missing Authorization header. Expected: Bearer <api_key>",
                            "type": "invalid_request_error",
                            "code": "missing_api_key",
                        }
                    },
                )
            token = auth_header[7:]  # Strip "Bearer "
            if token != api_key:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "message": "Invalid API key",
                            "type": "invalid_request_error",
                            "code": "invalid_api_key",
                        }
                    },
                )

        response = await call_next(request)
        return response
