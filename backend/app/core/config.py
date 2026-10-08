from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置，从环境变量或 .env 文件读取"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 应用
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    SECRET_KEY: str = "dev-secret-change-me"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # DashScope
    DASHSCOPE_API_KEY: Optional[str] = None
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 模型
    EMBEDDING_MODEL: str = "text-embedding-v3"
    EMBEDDING_DIMENSIONS: int = 1024

    # LLM（独立于 DashScope，可指向 DeepSeek 官网等 OpenAI 兼容服务）
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_MODEL: str = "deepseek-v4-pro"
    LLM_TEMPERATURE: float = 0.7
    RERANK_MODEL: str = "gte-rerank"
    USE_RERANK: bool = True

    # RAG
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 40
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_KEYWORD_WEIGHT: float = 0.4
    RETRIEVAL_VECTOR_WEIGHT: float = 0.6
    LOW_CONFIDENCE_THRESHOLD: float = 0.25  # 检索 top1 分数低于此值时拒绝硬答
    MAX_CONTEXT_LENGTH: int = 4000

    # 向量库后端：memory（内存 FAISS，开发用）/ milvus（持久化，生产用）
    VECTOR_DB_TYPE: str = "memory"
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "finins_chunks"

    # 语义缓存
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_THRESHOLD: float = 0.92

    # 多模态解析（图片提取 + 视觉模型 OCR）
    USE_MULTIMODAL: bool = True
    OCR_MODEL: str = "qwen-vl-max"
    OCR_LOCAL_FALLBACK: bool = True
    OCR_MIN_IMAGE_BYTES: int = 8192
    OCR_MIN_SIDE: int = 120

    # 上传
    UPLOAD_DIR: str = "data/uploads"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
