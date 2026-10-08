"""检索器：三级检索架构 — 混合粗排 → Rerank 精排 → 结果组装"""
from typing import Any, Dict, List, Optional, Tuple

import jieba

from app.core.config import settings
from app.rag.embedding import generate_embedding
from app.rag.vectorstore import InMemoryVectorStore

# 简单同义词表，提升召回
_SYNONYM_MAP = {
    "营收": ["营收", "收入", "营业收入"],
    "利润": ["利润", "盈利", "净利润"],
    "增长": ["增长", "提升", "增加", "上涨"],
    "下降": ["下降", "减少", "降低", "下滑"],
    "研发": ["研发", "研究开发", "R&D"],
    "毛利率": ["毛利率", "毛利", "利润率"],
}


def _build_keywords(question: str) -> Tuple[List[str], List[str]]:
    stop_patterns = [
        "请问", "是什么", "怎么样", "如何", "多少", "哪里", "怎么",
        "为什么", "能不能", "可以", "是否", "有没有", "什么", "哪",
    ]
    clean = question
    for p in stop_patterns:
        clean = clean.replace(p, " ")
    keywords = [k for k in jieba.lcut(clean.strip()) if len(k) >= 2]
    expanded = set(keywords)
    for kw in keywords:
        if kw in _SYNONYM_MAP:
            expanded.update(_SYNONYM_MAP[kw])
    return keywords, list(expanded)


def _rerank_with_dashscope(
    query: str, documents: List[str], top_n: int
) -> Optional[List[Tuple[int, float]]]:
    """调用阿里云 gte-rerank 做精排，失败返回 None"""
    if not settings.DASHSCOPE_API_KEY or not settings.USE_RERANK:
        return None
    try:
        import dashscope
        from dashscope import TextReRank

        dashscope.api_key = settings.DASHSCOPE_API_KEY
        resp = TextReRank.call(
            model=settings.RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
            return_documents=False,
        )
        if resp.status_code == 200:
            scored = [(r.index, r.relevance_score) for r in resp.output.results]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_n]
    except Exception:
        pass
    return None


def _local_rerank(
    query: str,
    query_vector: Optional[List[float]],
    documents: List[Dict],
    keyword_weight: float,
    vector_weight: float,
) -> List[Tuple[int, float]]:
    """本地精排 fallback：关键词重叠 + 向量相似度加权"""
    import numpy as np

    query_tokens = set(jieba.lcut(query))
    scores = []
    for i, doc in enumerate(documents):
        doc_tokens = set(doc.get("content_ltks", "").split())
        overlap = len(query_tokens & doc_tokens)
        kw_score = min(overlap / max(len(query_tokens), 1), 1.0)

        vec_score = 0.0
        if query_vector is not None:
            doc_vec = doc.get("q_1024_vec")
            if doc_vec is not None:
                a = np.array(query_vector)
                b = np.array(doc_vec)
                norm = np.linalg.norm(a) * np.linalg.norm(b)
                vec_score = max(0, float(np.dot(a, b) / norm)) if norm > 0 else 0.0

        scores.append((i, keyword_weight * kw_score + vector_weight * vec_score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


class Retriever:
    """RAG 检索器，封装完整检索流程"""

    def __init__(self, store: InMemoryVectorStore):
        self.store = store

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        keyword_weight: Optional[float] = None,
        vector_weight: Optional[float] = None,
        use_rerank: Optional[bool] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        top_k = top_k or settings.RETRIEVAL_TOP_K
        kw_w = keyword_weight if keyword_weight is not None else settings.RETRIEVAL_KEYWORD_WEIGHT
        vec_w = vector_weight if vector_weight is not None else settings.RETRIEVAL_VECTOR_WEIGHT
        use_rerank = use_rerank if use_rerank is not None else settings.USE_RERANK

        keywords, expanded = _build_keywords(question)
        query_vector = generate_embedding(question)

        # 第一级：混合粗排（带元数据前置过滤）
        candidates = self.store.hybrid_search(
            question, query_vector, top_k=50,
            keyword_weight=kw_w, vector_weight=vec_w, filter=filter,
        )

        # 降级：结果不足时放宽关键词权重重试（过滤条件保持不变）
        if len(candidates) < top_k:
            candidates = self.store.hybrid_search(
                question, query_vector, top_k=50,
                keyword_weight=0.2, vector_weight=0.8, filter=filter,
            )

        # 第二级：Rerank 精排
        if use_rerank and len(candidates) > top_k:
            doc_texts = [c["content"] for c in candidates]
            rerank_results = _rerank_with_dashscope(question, doc_texts, top_k)
            if rerank_results is not None:
                final_chunks = []
                for idx, score in rerank_results:
                    chunk = candidates[idx].copy()
                    chunk["rerank_score"] = round(score, 4)
                    chunk["final_score"] = round(score, 4)
                    final_chunks.append(chunk)
            else:
                local_scores = _local_rerank(
                    question, query_vector, candidates, kw_w, vec_w
                )
                final_chunks = []
                for idx, score in local_scores[:top_k]:
                    chunk = candidates[idx].copy()
                    chunk["final_score"] = round(score, 4)
                    final_chunks.append(chunk)
        else:
            final_chunks = candidates[:top_k]

        return {
            "total": len(candidates),
            "chunks": final_chunks,
            "query": question,
            "query_keywords": keywords,
            "expanded_keywords": expanded,
        }
