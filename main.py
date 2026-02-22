"""
FreeAPI Pool - 主入口
启动 FastAPI 服务，同时提供 API 和 Web UI
"""

import sys
import os
import uvicorn
from pathlib import Path

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from api.routes import create_api_app


def create_app() -> FastAPI:
    """创建完整的应用实例"""

    # 创建 API 应用
    app = create_api_app()

    # 挂载静态文件
    static_dir = PROJECT_ROOT / "gui" / "static"
    template_dir = PROJECT_ROOT / "gui" / "templates"

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Web UI 首页
    @app.get("/", response_class=HTMLResponse)
    async def web_ui():
        index_file = template_dir / "index.html"
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

    @app.get("/chat", response_class=HTMLResponse)
    @app.get("/chat/", response_class=HTMLResponse)
    async def chat_ui():
        index_file = template_dir / "index.html"
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

    # 健康检查
    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "1.0.0", "service": "FreeAPI Pool"}

    return app


app = create_app()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════╗
║          FreeAPI Pool - 免费API流量池            ║
║                                                  ║
║  Web UI:  http://localhost:8080                   ║
║  API:     http://localhost:8080/v1                ║
║  文档:    http://localhost:8080/docs              ║
╚══════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )
