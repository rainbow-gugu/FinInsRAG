"""元数据 tag 前后置过滤 —— 召回精确度评估（Milvus 实测）

合成保险条款数据（每个产品唯一、金标准可精确计算），在 Milvus HNSW 上对比：
  1) strict-pre  严格前置：search 直接带 expr，不做兜底
  2) post@N      纯后置：不带 expr 取 N 条，Python 过滤（N=10 / 50）
  3) auto        项目实际行为：前置不足一半自动降级放大召回+后置过滤

指标：
  Hit Rate        金标准文档是否出现在 top5（内容找对没有）
  Filter Precision 返回结果中真正满足 filter 的比例（应=1.0，否则是漏过滤）
  Avg Returned    平均返回条数（前置在高选择性下可能凑不齐）

用法：
  cd backend
  python3 eval/eval_metadata_filter.py
前提：Milvus 已启动（见 milvus/docker-compose.yml），.env 配好 DashScope key。
"""
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import jieba

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings  # noqa: E402
from app.rag.embedding import generate_embedding  # noqa: E402

EVAL_COLLECTION = "finins_filter_eval"
_HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DB = os.path.join(_HERE, "runtime", "filter_eval.db")  # Milvus Lite 本地文件
TOP_K = 5
random.seed(42)

# ---------- 合成数据 ----------

# 24 个唯一保险产品名
PRODUCTS = [
    "安鑫保", "康悦人生", "e生无忧", "百万守护", "重疾无忧", "医路安康",
    "福寿年年", "鑫享人生", "康健百年", "守护专家", "e保通", "安心百万",
    "康宁宝贝", "银发安康", "尊享医疗", "普惠保", "全民健康", "长护无忧",
    "意外全保", "母婴安康", "少儿守护", "企业补充", "高端医疗", "惠民保",
]

# (条款主题, 问题模板, 内容模板, 取值池)
CLAUSES = [
    ("等待期", "{p}的等待期是多久", "{p}的等待期为{n}天，续保保单不再设等待期。", [30, 60, 90]),
    ("保额", "{p}的年度保额上限是多少", "{p}的年度保额上限为{n}万元，重大疾病医疗双倍赔付。", [100, 200, 400, 600]),
    ("免赔额", "{p}的年免赔额是多少", "{p}的年免赔额为{n}元，超出免赔额部分按比例报销。", [5000, 10000, 20000]),
    ("续保年龄", "{p}最高可以续保到多少岁", "{p}的最高续保年龄为{n}岁，通过健康告知方可投保。", [60, 65, 70, 80]),
    ("报销比例", "{p}的报销比例是多少", "{p}经医保结算后报销比例为{n}%，未经医保报销比例相应降低。", [70, 80, 90, 100]),
]


def build_fixtures():
    """返回 (chunks, cases)。chunks 含元标签；cases 含金标准 expected_doc。"""
    chunks: List[Dict] = []
    for i, p in enumerate(PRODUCTS):
        doc_name = f"产品条款_{p}.md"
        doc_type = "pdf" if i % 2 == 0 else "docx"
        kb_id = ["A", "B", "C"][i % 3]
        chosen = random.sample(CLAUSES, 3)
        for j, (topic, q_t, c_t, vals) in enumerate(chosen):
            n = vals[i % len(vals)]
            content = c_t.format(p=p, n=n)
            chunks.append({
                "chunk_id": f"{p}_{j}",
                "content": content,
                "doc_name": doc_name,
                "doc_type": doc_type,
                "kb_id": kb_id,
                "topic": topic,
                "question": q_t.format(p=p),
            })

    # 六类过滤场景，每类采样若干问题
    def pick(pred):
        return [c for c in chunks if pred(c)]

    scenarios = [
        ("no_filter  (选择性100%)", None, lambda c: True),
        ("by_type=pdf(选择性50%)", {"doc_type": "pdf"}, lambda c: c["doc_type"] == "pdf"),
        ("by_kb=A   (选择性33%)", {"kb_id": "A"}, lambda c: c["kb_id"] == "A"),
        ("by_name   (选择性~4%)", None, lambda c: True),  # filter 按用例动态填该文档
        ("combo kb+type(~17%)", None, lambda c: True),    # 动态
        ("no_match  (选择性0%)", {"doc_name": "__不存在的文档__.md"}, lambda c: False),
    ]

    cases = []
    for name, base_filter, pred in scenarios:
        pool = pick(pred)
        if name.startswith("by_name"):
            sample = random.sample(pool, 12)
            for c in sample:
                cases.append({"scenario": name, "question": c["question"],
                              "filter": {"doc_name": c["doc_name"]},
                              "expected_doc": c["doc_name"], "should_hit": True})
        elif name.startswith("combo"):
            sub = [c for c in chunks if c["kb_id"] == "A" and c["doc_type"] == "pdf"]
            for c in random.sample(sub, min(10, len(sub))):
                cases.append({"scenario": name, "question": c["question"],
                              "filter": {"kb_id": "A", "doc_type": "pdf"},
                              "expected_doc": c["doc_name"], "should_hit": True})
        elif name.startswith("no_match"):
            for c in random.sample(chunks, 8):
                cases.append({"scenario": name, "question": c["question"],
                              "filter": base_filter,
                              "expected_doc": None, "should_hit": False})
        else:
            for c in random.sample(pool, min(12, len(pool))):
                cases.append({"scenario": name, "question": c["question"],
                              "filter": base_filter,
                              "expected_doc": c["doc_name"], "should_hit": True})
    return chunks, cases


# ---------- Milvus 接入 ----------

def get_eval_collection():
    from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                          connections)
    dim = settings.EMBEDDING_DIMENSIONS
    os.makedirs(os.path.dirname(EVAL_DB), exist_ok=True)
    # Milvus Lite：嵌入式，本地文件持久化，无需 Docker/etcd/minio
    # 生产 standalone：改用 connections.connect(host=..., port=19530)
    connections.connect(uri=EVAL_DB)
    try:
        Collection(EVAL_COLLECTION).drop()   # 重跑先清理，保证干净
    except Exception:
        pass
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=32),
    ]
    col = Collection(EVAL_COLLECTION, CollectionSchema(fields))
    col.create_index(field_name="vector", index_params={
        "index_type": "HNSW", "metric_type": "IP",
        "params": {"M": 32, "efConstruction": 200}})
    return col


def ingest(col, chunks):
    import numpy as np
    data = defaultdict(list)
    for c in chunks:
        vec = generate_embedding(c["content"])
        arr = np.asarray(vec, dtype="float32")
        norm = np.linalg.norm(arr)
        data["chunk_id"].append(c["chunk_id"])
        data["vector"].append((arr / norm if norm > 0 else arr).tolist())
        data["content"].append(c["content"][:8000])
        data["doc_name"].append(c["doc_name"])
        data["doc_type"].append(c["doc_type"])
        data["kb_id"].append(c["kb_id"])
    col.insert([data[k] for k in ["chunk_id", "vector", "content", "doc_name", "doc_type", "kb_id"]])
    col.load()


# ---------- 三种检索策略 ----------

def _norm(v):
    import numpy as np
    a = np.asarray(v, dtype="float32")
    n = np.linalg.norm(a)
    return (a / n if n > 0 else a).tolist()


def _hits_to_rows(hits):
    rows = []
    for h in hits:
        rows.append({
            "chunk_id": h.id,
            "doc_name": h.entity.get("doc_name", ""),
            "doc_type": h.entity.get("doc_type", ""),
            "kb_id": h.entity.get("kb_id", ""),
            "score": float(h.score),
        })
    return rows


def _satisfies(row, f: Optional[Dict]) -> bool:
    if not f:
        return True
    return all(row.get(k) == v for k, v in f.items())


def search_strict_pre(col, qvec, f):
    expr = _to_expr(f)
    kw = dict(data=[qvec], anns_field="vector",
              param={"metric_type": "IP", "params": {"ef": 64}},
              limit=TOP_K, output_fields=["doc_name", "doc_type", "kb_id"])
    if expr:
        kw["expr"] = expr
    return _hits_to_rows(col.search(**kw)[0])


def search_post(col, qvec, f, n):
    res = col.search(data=[qvec], anns_field="vector",
                     param={"metric_type": "IP", "params": {"ef": max(64, n)}},
                     limit=n, output_fields=["doc_name", "doc_type", "kb_id"])
    rows = [r for r in _hits_to_rows(res[0]) if _satisfies(r, f)]
    return rows[:TOP_K]


def _to_expr(f):
    if not f:
        return ""
    return " and ".join(f'{k} == "{v}"' for k, v in f.items())


# ---------- 评估 ----------

def evaluate(col, cases):
    stats = defaultdict(lambda: defaultdict(lambda: {"hit": 0, "prec_ok": 0,
                                                      "total": 0, "returned": 0}))
    for case in cases:
        qvec = _norm(generate_embedding(case["question"]))
        f = case["filter"]
        strategies = {
            "strict-pre": search_strict_pre(col, qvec, f),
            "post@10": search_post(col, qvec, f, 10),
            "post@50": search_post(col, qvec, f, 50),
        }
        for sname, rows in strategies.items():
            s = stats[case["scenario"]][sname]
            s["total"] += 1
            s["returned"] += len(rows)
            # 过滤精确度：返回结果是否都满足 filter
            if all(_satisfies(r, f) for r in rows):
                s["prec_ok"] += 1
            # 命中：该出现的金标准是否出现 / 不该出现时是否为空
            docs = [r["doc_name"] for r in rows]
            if case["should_hit"]:
                if case["expected_doc"] in docs:
                    s["hit"] += 1
            else:
                if len(rows) == 0:
                    s["hit"] += 1  # no_match 正确返回空算命中
    return stats


def report(stats):
    print("\n" + "=" * 92)
    print("元数据过滤召回精确度评估结果（Milvus HNSW）")
    print("=" * 92)
    header = f"{'场景':<24}{'策略':<11}{'Hit Rate':>10}{'Filter Prec':>12}{'Avg Returned':>14}"
    print(header)
    print("-" * 92)
    for scenario, m in stats.items():
        for sname in ["strict-pre", "post@10", "post@50"]:
            s = m[sname]
            t = s["total"] or 1
            print(f"{scenario:<24}{sname:<11}{s['hit']/t:>9.0%}{s['prec_ok']/t:>11.0%}"
                  f"{s['returned']/t:>13.1f}")
        print("-" * 92)
    print("说明：Hit Rate=金标准找对(no_match场景看是否正确返空)；Filter Prec=结果全部满足条件；")
    print("      Avg Returned=平均返回条数，<5 表示该策略在该选择性下凑不齐结果。")


def main():
    print("1/4 生成合成保险条款数据 ...")
    chunks, cases = build_fixtures()
    print(f"    文档数={len(set(c['doc_name'] for c in chunks))}  chunk数={len(chunks)}  用例数={len(cases)}")
    print("2/4 连接 Milvus 并建评估 collection ...")
    col = get_eval_collection()
    print("3/4 向量化并摄入（调用 embedding，稍候）...")
    ingest(col, chunks)
    print("4/4 跑前置/后置对比评估 ...")
    stats = evaluate(col, cases)
    report(stats)


if __name__ == "__main__":
    main()
