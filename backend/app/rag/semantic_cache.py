# -*- coding: utf-8 -*-
"""语义缓存：新问题与历史问题语义相似度超过阈值时直接返回历史答案，跳过 LLM 调用。
存储先用进程内字典（与项目现有内存存储风格一致），后续可替换为 Redis 实现。"""
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.config import settings
from app.rag.embedding import generate_embedding


class SemanticCache:
    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold if threshold is not None else settings.SEMANTIC_CACHE_THRESHOLD
        self.entries: List[Dict[str, Any]] = []

    def best_match(self, question: str):
        """返回 (best_entry, best_similarity)，不做阈值判断；无缓存或向量失败返回 (None, 0)。"""
        if not self.entries:
            return None, 0.0
        qvec = generate_embedding(question)
        if qvec is None:
            return None, 0.0
        qv = np.array(qvec)
        best, best_sim = None, -1.0
        for e in self.entries:
            ev = np.array(e["vector"])
            norm = np.linalg.norm(qv) * np.linalg.norm(ev)
            sim = float(np.dot(qv, ev) / norm) if norm > 0 else 0.0
            if sim > best_sim:
                best, best_sim = e, sim
        return best, best_sim

    def _pack(self, entry: Dict[str, Any], sim: float) -> Dict[str, Any]:
        return {
            "answer": entry["answer"],
            "references": entry.get("references", []),
            "similarity": round(sim, 4),
            "origin_question": entry["question"],
        }

    def lookup(self, question: str) -> Optional[Dict[str, Any]]:
        """命中返回 {answer, references, similarity, origin_question}，否则 None。"""
        best, best_sim = self.best_match(question)
        if best is not None and best_sim >= self.threshold:
            return self._pack(best, best_sim)
        return None

    def add(self, question: str, answer: str, references: Optional[List[Dict[str, Any]]] = None) -> bool:
        """写入一条问答缓存；问题向量生成失败则不写入。"""
        qvec = generate_embedding(question)
        if qvec is None or not answer:
            return False
        # 同一问题不重复缓存
        if any(e["question"] == question for e in self.entries):
            return False
        serializable_refs = []
        for r in references or []:
            serializable_refs.append({
                "chunk_id": r.get("chunk_id"),
                "content": r.get("content", ""),
                "doc_name": r.get("doc_name", ""),
                "score": r.get("final_score", r.get("score", 0)),
            })
        self.entries.append({
            "question": question,
            "vector": qvec,
            "answer": answer,
            "references": serializable_refs,
        })
        return True

    def stats(self) -> Dict[str, Any]:
        return {"size": len(self.entries), "threshold": self.threshold}


_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
