"""RAG 流水线：编排 文档摄入 → 检索 → 生成 完整流程"""
import os
from typing import Any, Dict, Generator, List, Optional

from app.core.config import settings
from app.rag.chunker import naive_merge, tokenize_chunks
from app.rag.embedding import generate_embedding
from app.rag.generator import PromptBuilder, generate_recommended_questions, stream_chat
from app.rag.multimodal import process_file
from app.rag.parser import parse_document
from app.rag.retriever import Retriever
from app.rag.semantic_cache import get_semantic_cache
from app.rag.vectorstore import get_vector_store


class RAGPipeline:
    """RAG 完整流水线，单例使用"""

    def __init__(self):
        self.store = get_vector_store()
        self.retriever = Retriever(self.store)

    # ---------- 离线：文档摄入 ----------

    def ingest_file(self, file_path: str, user_id: str = "default") -> Dict[str, Any]:
        """
        摄入一个文档：解析 → 分块 → 向量化 → 入库
        返回摄入统计信息
        """
        doc_name = os.path.basename(file_path)

        # 1. 解析
        doc_type, text = parse_document(file_path)

        # 2. 分块
        chunks = naive_merge(
            text,
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP,
        )
        chunks = tokenize_chunks(chunks)

        # 2.5 多模态：提取内嵌图片与矢量图表区域，OCR 文本追加为分块
        images_extracted = ocr_succeeded = 0
        if settings.USE_MULTIMODAL:
            ocr_recs = process_file(file_path)
            images_extracted = len(ocr_recs)
            for rec in ocr_recs:
                if not rec.get("ocr_text"):
                    continue
                ocr_succeeded += 1
                extra = naive_merge(
                    rec["ocr_text"],
                    chunk_size=settings.CHUNK_SIZE,
                    overlap=settings.CHUNK_OVERLAP,
                )
                for c in extra:
                    c["chunk_type"] = "image_ocr"
                    c["source_image"] = rec["name"]
                chunks.extend(tokenize_chunks(extra))

        # 3. 向量化
        texts = [c["content"] for c in chunks]
        embeddings = generate_embedding(texts)
        valid_count = 0
        for chunk, emb in zip(chunks, embeddings):
            if emb is not None:
                chunk["q_1024_vec"] = emb
                chunk["vector_dim"] = len(emb)
                valid_count += 1
            chunk["doc_name"] = doc_name
            chunk["doc_type"] = doc_type
            chunk["kb_id"] = user_id

        # 4. 入库
        inserted = self.store.insert(chunks, index_name=user_id)

        return {
            "doc_name": doc_name,
            "doc_type": doc_type,
            "total_chunks": len(chunks),
            "vectorized_chunks": valid_count,
            "inserted": inserted,
            "char_count": len(text),
            "images_extracted": images_extracted,
            "ocr_succeeded": ocr_succeeded,
        }

    def ingest_text(
        self, text: str, doc_name: str, user_id: str = "default"
    ) -> Dict[str, Any]:
        """直接摄入纯文本（用于演示或测试）"""
        chunks = naive_merge(
            text, chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP
        )
        chunks = tokenize_chunks(chunks)
        texts = [c["content"] for c in chunks]
        embeddings = generate_embedding(texts)
        for chunk, emb in zip(chunks, embeddings):
            if emb is not None:
                chunk["q_1024_vec"] = emb
            chunk["doc_name"] = doc_name
            chunk["kb_id"] = user_id
        inserted = self.store.insert(chunks, index_name=user_id)
        return {"doc_name": doc_name, "total_chunks": len(chunks), "inserted": inserted}

    def list_documents(self) -> List[Dict[str, Any]]:
        """列出已摄入的文档（按 doc_name 聚合）——调 store 接口，不直接访问内部"""
        return self.store.list_docs()

    def delete_document(self, doc_name: str) -> int:
        return self.store.delete_by_doc(doc_name)

    # ---------- 在线：检索 + 生成 ----------

    def chat_stream(
        self,
        question: str,
        session_id: str = "default",
        filter: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        完整 RAG 对话（流式）：
        1. 检索相关文档
        2. 构建 Prompt
        3. 流式生成回答
        yield SSE 事件帧
        """
        # 0. 语义缓存：命中则直接返回历史答案，跳过检索与 LLM
        if settings.SEMANTIC_CACHE_ENABLED:
            hit = get_semantic_cache().lookup(question)
            if hit is not None:
                yield {"type": "cache_status", "data": {
                    "hit": True, "similarity": hit["similarity"],
                    "origin_question": hit["origin_question"]}}
                yield {"type": "references", "data": {
                    "total": len(hit["references"]),
                    "references": hit["references"]}}
                yield {"type": "content", "text": hit["answer"]}
                yield {"type": "done", "data": {
                    "answer": hit["answer"], "thinking": "",
                    "recommended_questions": [], "cache_hit": True}}
                return
            yield {"type": "cache_status", "data": {"hit": False}}

        # 1. 检索
        retrieval_result = self.retriever.retrieve(question, filter=filter)
        references = retrieval_result["chunks"]

        # 1.5 置信度拦截：top1 分数太低，不调 LLM，直接提示人工核实
        if not references or references[0].get("final_score", 0) < settings.LOW_CONFIDENCE_THRESHOLD:
            yield {"type": "references", "data": {"total": 0, "references": []}}
            yield {"type": "content", "text": "抱歉，在参考资料中未找到足够相关的内容，建议人工核实或换个问法。"}
            yield {"type": "done", "data": {"answer": "", "thinking": "", "recommended_questions": []}}
            return

        # 2. 构建 Prompt
        prompt = PromptBuilder.build(
            question, references, max_ref_length=settings.MAX_CONTEXT_LENGTH
        )

        # 3. 先推送检索结果事件（前端可展示引用来源）
        yield {
            "type": "references",
            "data": {
                "total": retrieval_result["total"],
                "references": [
                    {
                        "chunk_id": r.get("chunk_id"),
                        "content": r.get("content", ""),
                        "doc_name": r.get("doc_name", ""),
                        "score": r.get("final_score", r.get("score", 0)),
                    }
                    for r in references
                ],
            },
        }

        # 4. 流式生成
        thinking = ""
        answer = ""
        for event in stream_chat(prompt):
            if event["type"] == "thinking":
                thinking = event.get("full_text", "")
                yield {"type": "thinking", "text": event["text"]}
            elif event["type"] == "content":
                answer = event.get("full_text", "")
                yield {"type": "content", "text": event["text"]}
            elif event["type"] == "done":
                thinking = event.get("thinking", "")
                answer = event.get("content", "")
            elif event["type"] == "error":
                yield {"type": "error", "text": event["text"]}
                return

        # 5. 写入语义缓存，并生成推荐问题
        if settings.SEMANTIC_CACHE_ENABLED:
            get_semantic_cache().add(question, answer, references)
        doc_names = list(set(r.get("doc_name", "") for r in references if r.get("doc_name")))
        recommended = generate_recommended_questions(question, doc_names)

        # 6. 结束事件
        yield {
            "type": "done",
            "data": {
                "answer": answer,
                "thinking": thinking,
                "recommended_questions": recommended,
            },
        }

    def stats(self) -> Dict[str, Any]:
        return self.store.stats()


# 全局单例
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
