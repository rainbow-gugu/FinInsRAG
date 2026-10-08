"""FinInsRAG-V3 应用入口"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.feedback import router as feedback_router
from app.core.config import settings
from app.rag.pipeline import get_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时初始化 RAG 流水线"""
    get_pipeline()
    yield


app = FastAPI(
    title="FinInsRAG-V3 - 金融保险领域智能问答系统",
    description="基于 RAG（检索增强生成）的金融保险领域智能问答系统，支持文档上传、混合检索、Rerank 精排、流式生成",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由（统一 /api 前缀）
app.include_router(auth_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(feedback_router, prefix="/api")


@app.get("/")
def root():
    return {
        "name": "FinInsRAG-V3",
        "version": "3.0.0",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/api/health")
def health():
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "vector_store": pipeline.stats(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=True,
    )
