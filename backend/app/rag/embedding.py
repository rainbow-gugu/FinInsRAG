"""向量化模块：调用阿里云 text-embedding-v3，支持批量"""
from typing import List, Optional, Union

import numpy as np
from openai import OpenAI

from app.core.config import settings


def _get_client() -> Optional[OpenAI]:
    if not settings.DASHSCOPE_API_KEY:
        return None
    return OpenAI(api_key=settings.DASHSCOPE_API_KEY, base_url=settings.DASHSCOPE_BASE_URL)


def generate_embedding(
    text: Union[str, List[str]],
    model_name: Optional[str] = None,
    dimensions: Optional[int] = None,
    max_batch_size: int = 10,
) -> Union[Optional[List[float]], List[Optional[List[float]]]]:
    """
    生成文本向量。单条返回 list[float]，批量返回 list[list[float]]。
    未配置 API Key 时返回 None（调用方应处理降级）。
    """
    client = _get_client()
    model = model_name or settings.EMBEDDING_MODEL
    dim = dimensions or settings.EMBEDDING_DIMENSIONS

    if client is None:
        return None if isinstance(text, str) else [None] * len(text)

    if isinstance(text, str):
        try:
            resp = client.embeddings.create(
                model=model, input=text, dimensions=dim, encoding_format="float"
            )
            return resp.data[0].embedding
        except Exception:
            return None

    # 批量
    all_embeddings: List[Optional[List[float]]] = []
    for i in range(0, len(text), max_batch_size):
        batch = text[i : i + max_batch_size]
        try:
            resp = client.embeddings.create(
                model=model, input=batch, dimensions=dim, encoding_format="float"
            )
            all_embeddings.extend([item.embedding for item in resp.data])
        except Exception:
            all_embeddings.extend([None] * len(batch))
    return all_embeddings


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
