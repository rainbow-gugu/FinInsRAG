"""向量存储：支持两种后端，接口完全一致
- memory : 内存 dict + FAISS HNSW（开发/演示用，重启丢数据）
- milvus : Milvus + HNSW（生产用，持久化，重启不丢）
关键词检索(BM25)统一用本地 jieba 倒排（Milvus 不内置 BM25）。
"""
from typing import Any, Dict, List, Optional, Set

import jieba
import numpy as np

from app.core.config import settings


# ============ 公共：本地关键词倒排（BM25 风格） ============

class KeywordIndex:
    """token -> {chunk_id}，重启后可从向量库惰性重建。"""

    def __init__(self):
        self._index: Dict[str, Set[str]] = {}

    def add(self, chunk_id: str, content_ltks: str):
        for token in content_ltks.split():
            self._index.setdefault(token, set()).add(chunk_id)

    def remove(self, chunk_id: str):
        for ids in self._index.values():
            ids.discard(chunk_id)

    def clear(self):
        self._index.clear()

    def search(self, query: str, top_k: int = 100) -> List[tuple]:
        """返回 [(chunk_id, hit_count), ...] 按命中数降序"""
        keywords = jieba.lcut(query)
        scores: Dict[str, int] = {}
        for kw in keywords:
            for cid in self._index.get(kw, set()):
                scores[cid] = scores.get(cid, 0) + 1
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def _fuse_candidates(
    kw_results: List[Dict],
    vec_results: List[Dict],
    top_k: int,
    keyword_weight: float,
    vector_weight: float,
) -> List[Dict]:
    """混合检索融合：关键词分归一化 + 向量分，加权排序。两个后端共用。"""
    all_candidates: Dict[str, Dict] = {}
    for r in kw_results:
        all_candidates[r["chunk_id"]] = {
            **r, "keyword_score": r["score"], "vector_score": 0.0,
        }
    for r in vec_results:
        if r["chunk_id"] in all_candidates:
            all_candidates[r["chunk_id"]]["vector_score"] = r["score"]
        else:
            all_candidates[r["chunk_id"]] = {
                **r, "keyword_score": 0.0, "vector_score": r["score"],
            }
    for c in all_candidates.values():
        c["final_score"] = (
            keyword_weight * min(c["keyword_score"] / 5.0, 1.0)
            + vector_weight * max(c["vector_score"], 0.0)
        )
    return sorted(all_candidates.values(), key=lambda x: x["final_score"], reverse=True)[:top_k]


# ============ 元数据过滤工具 ============

def _match_metadata(doc: Dict, f: Optional[Dict]) -> bool:
    """内存后端：doc 是否满足元数据条件（AND），f 为空则全放行"""
    if not f:
        return True
    for key in ("doc_name", "doc_type", "kb_id"):
        val = f.get(key)
        if val and doc.get(key) != val:
            return False
    return True


def _build_milvus_expr(f: Optional[Dict]) -> str:
    """过滤 dict -> Milvus 布尔表达式，如 doc_type == "pdf" and kb_id == "k1" """
    if not f:
        return ""
    conds = []
    for key in ("doc_name", "doc_type", "kb_id"):
        val = f.get(key)
        if val:
            safe = str(val).replace('"', '\\"')
            conds.append(f'{key} == "{safe}"')
    return " and ".join(conds)


# ============ Memory 后端（原有 FAISS HNSW，保留作 fallback） ============

class InMemoryVectorStore:
    HNSW_M = 32

    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self._faiss = None
        self._faiss_ids: List[str] = []
        self._kw = KeywordIndex()
        self._kw_built = False

    def _ensure_kw(self):
        if self._kw_built:
            return
        for cid, doc in self.documents.items():
            self._kw.add(cid, doc.get("content_ltks", ""))
        self._kw_built = True

    def _rebuild_faiss(self):
        import faiss
        vector_rows = [
            (cid, np.asarray(doc["q_1024_vec"], dtype="float32"))
            for cid, doc in self.documents.items()
            if doc.get("q_1024_vec") is not None
        ]
        if not vector_rows:
            self._faiss, self._faiss_ids = None, []
            return
        dim = vector_rows[0][1].shape[0]
        index = faiss.IndexHNSWFlat(dim, self.HNSW_M, faiss.METRIC_INNER_PRODUCT)
        ids, matrix = [], []
        for cid, vec in vector_rows:
            norm = np.linalg.norm(vec)
            matrix.append(vec / norm if norm > 0 else vec)
            ids.append(cid)
        arr = np.vstack(matrix).astype("float32")
        index.hnsw.efConstruction = 200
        index.add(arr)
        self._faiss, self._faiss_ids = index, ids

    def insert(self, documents: List[Dict], index_name: str) -> int:
        count = 0
        for doc in documents:
            chunk_id = doc.get("chunk_id", doc.get("id"))
            self.documents[chunk_id] = {**doc, "_index": index_name}
            self._kw.add(chunk_id, doc.get("content_ltks", ""))
            count += 1
        self._kw_built = True
        self._rebuild_faiss()
        return count

    def delete_by_doc(self, doc_name: str) -> int:
        to_delete = [
            cid for cid, doc in self.documents.items()
            if doc.get("doc_name") == doc_name
        ]
        for cid in to_delete:
            del self.documents[cid]
            self._kw.remove(cid)
        self._rebuild_faiss()
        return len(to_delete)

    def search_by_keyword(self, query: str, top_k: int = 50, filter: Optional[Dict] = None) -> List[Dict]:
        self._ensure_kw()
        hits = self._kw.search(query, top_k=100)
        results = []
        for cid, score in hits:
            doc = self.documents.get(cid)
            if not doc or not _match_metadata(doc, filter):
                continue
            results.append({
                "chunk_id": cid, "content": doc["content"],
                "doc_name": doc.get("doc_name", ""),
                "score": float(score), "match_type": "keyword",
            })
        return results[:top_k]

    def search_by_vector(self, query_vector: List[float], top_k: int = 50, filter: Optional[Dict] = None) -> List[Dict]:
        if self._faiss is None:
            return []
        qv = np.asarray(query_vector, dtype="float32")
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv = qv / norm
        # FAISS HNSW 不支持检索时过滤：有 filter 时多召回 3 倍，再后置过滤
        fetch_k = min(top_k * 3, self._faiss.ntotal) if filter else top_k
        self._faiss.hnsw.efSearch = max(64, fetch_k * 4)
        scores, positions = self._faiss.search(qv.reshape(1, -1), fetch_k)
        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos < 0:
                continue
            chunk_id = self._faiss_ids[pos]
            doc = self.documents[chunk_id]
            if not _match_metadata(doc, filter):
                continue
            results.append({
                "chunk_id": chunk_id, "content": doc["content"],
                "doc_name": doc.get("doc_name", ""),
                "score": float(score), "match_type": "vector",
            })
            if len(results) >= top_k:
                break
        return results

    def hybrid_search(
        self, query: str, query_vector: Optional[List[float]],
        top_k: int = 5, keyword_weight: float = 0.4, vector_weight: float = 0.6,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        kw = self.search_by_keyword(query, top_k=100, filter=filter)
        vec = self.search_by_vector(query_vector, top_k=100, filter=filter) if query_vector else []
        return _fuse_candidates(kw, vec, top_k, keyword_weight, vector_weight)

    def list_docs(self) -> List[Dict]:
        doc_map: Dict[str, Dict] = {}
        for doc in self.documents.values():
            name = doc.get("doc_name", "unknown")
            if name not in doc_map:
                doc_map[name] = {
                    "doc_name": name, "doc_type": doc.get("doc_type", ""),
                    "chunk_count": 0,
                }
            doc_map[name]["chunk_count"] += 1
        return list(doc_map.values())

    def stats(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "doc_names": list(set(d.get("doc_name", "") for d in self.documents.values())),
        }


# ============ Milvus 后端（HNSW + 持久化） ============

class MilvusVectorStore:
    """Milvus + HNSW 向量存储，持久化，重启不丢数据。
    向量入库时归一化，metric_type=IP（内积=余弦相似度）。
    关键词检索仍用本地倒排，启动后惰性从 Milvus 重建。"""

    HNSW_M = 32
    HNSW_EFCONSTRUCTION = 200

    def __init__(self):
        from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
        self._Collection = Collection
        self._FieldSchema = FieldSchema
        self._CollectionSchema = CollectionSchema
        self._DataType = DataType
        connections.connect(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        self._collection = self._get_or_create_collection()
        self._collection.load()
        self._kw = KeywordIndex()
        self._kw_built = False

    def _get_or_create_collection(self):
        fields = [
            self._FieldSchema(name="chunk_id", dtype=self._DataType.VARCHAR, max_length=64, is_primary=True),
            self._FieldSchema(name="vector", dtype=self._DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIMENSIONS),
            self._FieldSchema(name="content", dtype=self._DataType.VARCHAR, max_length=65535),
            self._FieldSchema(name="content_ltks", dtype=self._DataType.VARCHAR, max_length=65535),
            self._FieldSchema(name="doc_name", dtype=self._DataType.VARCHAR, max_length=256),
            self._FieldSchema(name="kb_id", dtype=self._DataType.VARCHAR, max_length=64),
            self._FieldSchema(name="doc_type", dtype=self._DataType.VARCHAR, max_length=64),
            self._FieldSchema(name="_index", dtype=self._DataType.VARCHAR, max_length=64),
        ]
        schema = self._CollectionSchema(fields, description="FinInsRAG chunks")
        collection = self._Collection(settings.MILVUS_COLLECTION, schema)
        if not collection.has_index():
            collection.create_index(
                field_name="vector",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "IP",
                    "params": {"M": self.HNSW_M, "efConstruction": self.HNSW_EFCONSTRUCTION},
                },
            )
        return collection

    def _ensure_kw(self):
        """惰性重建关键词倒排：从 Milvus 全量拉取 content_ltks"""
        if self._kw_built:
            return
        rows = self._collection.query(expr="doc_name != ''", output_fields=["chunk_id", "content_ltks"])
        for row in rows:
            self._kw.add(row["chunk_id"], row.get("content_ltks", ""))
        self._kw_built = True

    @staticmethod
    def _normalize(vec: List[float]) -> List[float]:
        arr = np.asarray(vec, dtype="float32")
        norm = np.linalg.norm(arr)
        return (arr / norm if norm > 0 else arr).tolist()

    def insert(self, documents: List[Dict], index_name: str) -> int:
        data = {k: [] for k in [
            "chunk_id", "vector", "content", "content_ltks",
            "doc_name", "kb_id", "doc_type", "_index",
        ]}
        count = 0
        for doc in documents:
            vec = doc.get("q_1024_vec")
            if vec is None:
                continue
            chunk_id = doc.get("chunk_id", doc.get("id"))
            data["chunk_id"].append(chunk_id)
            data["vector"].append(self._normalize(vec))
            data["content"].append(doc.get("content", "")[:65534])
            data["content_ltks"].append(doc.get("content_ltks", "")[:65534])
            data["doc_name"].append(doc.get("doc_name", ""))
            data["kb_id"].append(doc.get("kb_id", ""))
            data["doc_type"].append(doc.get("doc_type", ""))
            data["_index"].append(index_name)
            self._kw.add(chunk_id, doc.get("content_ltks", ""))
            count += 1
        if count > 0:
            self._collection.insert(data)
            self._collection.flush()
        self._kw_built = True
        return count

    def delete_by_doc(self, doc_name: str) -> int:
        safe = doc_name.replace('"', '\\"')
        rows = self._collection.query(
            expr=f'doc_name == "{safe}"', output_fields=["chunk_id"],
        )
        count = len(rows)
        if count > 0:
            self._collection.delete(expr=f'doc_name == "{safe}"')
            self._collection.flush()
            self._kw.clear()
            self._kw_built = False
        return count

    def search_by_keyword(self, query: str, top_k: int = 50, filter: Optional[Dict] = None) -> List[Dict]:
        self._ensure_kw()
        hits = self._kw.search(query, top_k=100)
        if not hits:
            return []
        chunk_ids = [cid for cid, _ in hits]
        id_list = "[" + ", ".join(f'"{cid}"' for cid in chunk_ids) + "]"
        # chunk_id 列表条件 + 元数据条件（AND）
        expr_parts = [f"chunk_id in {id_list}"]
        meta_expr = _build_milvus_expr(filter)
        if meta_expr:
            expr_parts.append(meta_expr)
        rows = self._collection.query(
            expr=" and ".join(expr_parts),
            output_fields=["chunk_id", "content", "doc_name"],
        )
        row_map = {r["chunk_id"]: r for r in rows}
        results = []
        for cid, score in hits:
            row = row_map.get(cid)
            if not row:
                continue
            results.append({
                "chunk_id": cid, "content": row["content"],
                "doc_name": row.get("doc_name", ""),
                "score": float(score), "match_type": "keyword",
            })
        return results[:top_k]

    @staticmethod
    def _iter_hits(results):
        for hits in results:
            for hit in hits:
                yield hit

    def _hits_to_docs(self, results) -> List[Dict]:
        out = []
        for hit in self._iter_hits(results):
            out.append({
                "chunk_id": hit.id,
                "content": hit.entity.get("content", ""),
                "doc_name": hit.entity.get("doc_name", ""),
                "score": float(hit.score),
                "match_type": "vector",
            })
        return out

    def search_by_vector(self, query_vector: List[float], top_k: int = 50, filter: Optional[Dict] = None) -> List[Dict]:
        """向量检索：优先前置过滤(expr)；前置命中不足一半时降级为放大召回+后置过滤。"""
        qv = self._normalize(query_vector)
        expr = _build_milvus_expr(filter)
        meta_fields = ["content", "doc_name", "doc_type", "kb_id"]

        # 第一轮：前置过滤 ANN（检索时就带条件）
        search_kwargs: Dict[str, Any] = dict(
            data=[qv], anns_field="vector",
            param={"metric_type": "IP", "params": {"ef": max(64, top_k * 4)}},
            limit=top_k, output_fields=meta_fields,
        )
        if expr:
            search_kwargs["expr"] = expr
        out = self._hits_to_docs(self._collection.search(**search_kwargs))

        # 降级：有过滤条件但前置命中不足一半 -> 去掉 expr，放大 5 倍召回，在结果上后置过滤
        if filter and len(out) < max(1, top_k // 2):
            fallback_kwargs = dict(
                data=[qv], anns_field="vector",
                param={"metric_type": "IP", "params": {"ef": max(128, top_k * 8)}},
                limit=top_k * 5, output_fields=meta_fields,
            )
            existing = {d["chunk_id"] for d in out}
            for hit in self._iter_hits(self._collection.search(**fallback_kwargs)):
                entity = {f: hit.entity.get(f, "") for f in meta_fields}
                if hit.id in existing or not _match_metadata(entity, filter):
                    continue
                out.append({
                    "chunk_id": hit.id,
                    "content": entity["content"],
                    "doc_name": entity["doc_name"],
                    "score": float(hit.score),
                    "match_type": "vector",
                })
                existing.add(hit.id)
                if len(out) >= top_k:
                    break
        return out[:top_k]

    def hybrid_search(
        self, query: str, query_vector: Optional[List[float]],
        top_k: int = 5, keyword_weight: float = 0.4, vector_weight: float = 0.6,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        kw = self.search_by_keyword(query, top_k=100, filter=filter)
        vec = self.search_by_vector(query_vector, top_k=100, filter=filter) if query_vector else []
        return _fuse_candidates(kw, vec, top_k, keyword_weight, vector_weight)

    def list_docs(self) -> List[Dict]:
        rows = self._collection.query(
            expr="doc_name != ''", output_fields=["doc_name", "doc_type"],
        )
        doc_map: Dict[str, Dict] = {}
        for row in rows:
            name = row.get("doc_name", "unknown")
            if name not in doc_map:
                doc_map[name] = {
                    "doc_name": name, "doc_type": row.get("doc_type", ""),
                    "chunk_count": 0,
                }
            doc_map[name]["chunk_count"] += 1
        return list(doc_map.values())

    def stats(self) -> Dict:
        count = self._collection.num_entities
        rows = self._collection.query(expr="doc_name != ''", output_fields=["doc_name"])
        return {
            "total_documents": count,
            "doc_names": list(set(r.get("doc_name", "") for r in rows)),
        }


# ============ 工厂 ============

_store: Optional[Any] = None


def get_vector_store():
    """根据配置返回 memory 或 milvus 后端，接口完全一致。"""
    global _store
    if _store is None:
        if settings.VECTOR_DB_TYPE == "milvus":
            _store = MilvusVectorStore()
        else:
            _store = InMemoryVectorStore()
    return _store
