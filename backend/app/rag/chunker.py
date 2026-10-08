"""文本分块模块：naive_merge 算法，按段落优先、句子兜底、带重叠"""
import re
from typing import Any, Dict, List

import jieba


def split_sentences(text: str) -> List[str]:
    pattern = r"([。！？；\n]+|(?<![0-9])[.?!](?![0-9]))"
    parts = re.split(pattern, text)
    sentences = []
    current = ""
    for part in parts:
        current += part
        if re.search(r"[。！？；\n]", part) or (
            re.search(r"[.?!]", part) and len(current) > 5
        ):
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return [s for s in sentences if s]


def naive_merge(
    text: str,
    chunk_size: int = 300,
    overlap: int = 40,
) -> List[Dict[str, Any]]:
    """
    核心分块算法：
    1. 按段落拆分，逐段累加达到 chunk_size 切出
    2. 单段落超长时按句子切分
    3. 相邻块保留 overlap 字符重叠
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""
    chunk_id = 0

    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk.strip():
                chunk_id += 1
                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_id:04d}",
                        "content": current_chunk.strip(),
                        "char_count": len(current_chunk.strip()),
                    }
                )

            if len(para) > chunk_size:
                sentences = split_sentences(para)
                sub_chunk = ""
                for sent in sentences:
                    if len(sub_chunk) + len(sent) <= chunk_size:
                        sub_chunk += sent
                    else:
                        if sub_chunk.strip():
                            chunk_id += 1
                            chunks.append(
                                {
                                    "chunk_id": f"chunk_{chunk_id:04d}",
                                    "content": sub_chunk.strip(),
                                    "char_count": len(sub_chunk.strip()),
                                }
                            )
                        sub_chunk = sent
                current_chunk = sub_chunk + "\n\n" if sub_chunk.strip() else ""
            else:
                current_chunk = para + "\n\n"

    if current_chunk.strip():
        chunk_id += 1
        chunks.append(
            {
                "chunk_id": f"chunk_{chunk_id:04d}",
                "content": current_chunk.strip(),
                "char_count": len(current_chunk.strip()),
            }
        )

    # 添加重叠
    for i, chunk in enumerate(chunks):
        if i > 0 and overlap > 0:
            prev_content = chunks[i - 1]["content"]
            if len(prev_content) >= overlap:
                chunk["content"] = prev_content[-overlap:] + " " + chunk["content"]
                chunk["char_count"] = len(chunk["content"])

    return chunks


def tokenize_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """为每个 chunk 生成 jieba 分词，用于 BM25 全文检索"""
    for chunk in chunks:
        words = jieba.lcut(chunk["content"])
        chunk["content_ltks"] = " ".join(words)
    return chunks
