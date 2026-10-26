"""Chạy một cấu hình truy xuất trên bộ câu hỏi chuẩn (golden set), lưu điểm từng câu và điểm trung bình."""

import csv
import json
import sys

from .chunking import chunk_corpus
from .config import EXPERIMENTS_DIR, MRR_K, RERANK_CANDIDATES, THIN_OVERLAP, TOP_K
from .corpus import load_corpus
from .golden import load_golden
from .metrics import hit_at_k, hits, overlap_chars, reciprocal_rank
from .retrieval import build_bm25, build_index, dense_search, hybrid_search, rerank

CONFIGS = {
    "B0": {"folder": "B0_baseline", "chunking": "fixed", "search": "dense"},
    "E1-sentence": {"folder": "E1_chunking/sentence", "chunking": "sentence", "search": "dense"},
    "E1-recursive": {"folder": "E1_chunking/recursive", "chunking": "recursive", "search": "dense"},
    "E2-hybrid": {"folder": "E2_hybrid_search", "chunking": "fixed", "search": "hybrid"},
    "E3-rerank": {"folder": "E3_reranking", "chunking": "fixed", "search": "hybrid+rerank"},
}


def run(name: str) -> dict:
    config = CONFIGS[name]
    docs = load_corpus()
    sections = {s.section_id: s for doc in docs for s in doc.sections}
    chunks = {c.chunk_id: c for c in chunk_corpus(docs, config["chunking"])}
    index = build_index(list(chunks.values()), config["chunking"])
    if config["search"].startswith("hybrid"):
        bm25 = build_bm25(list(chunks.values()))
    if config["search"] == "hybrid":
        search = lambda question: hybrid_search(index, bm25, list(chunks), question, MRR_K)
    elif config["search"] == "hybrid+rerank":
        search = lambda question: rerank(question, [chunks[i] for i in hybrid_search(index, bm25, list(chunks), question, RERANK_CANDIDATES)], MRR_K)
    else:
        search = lambda question: dense_search(index, question, MRR_K)

    rows = []
    for q in load_golden():
        found = [chunks[i] for i in search(q.question)]
        hit_list = hits(found, q.relevant_sections)
        hit = hit_at_k(hit_list, TOP_K)
        # Câu "trúng nhờ chạm mép": có trúng, nhưng không chunk nào trong top-5 chứa đủ THIN_OVERLAP ký tự của section đúng
        best_overlap = max(overlap_chars(c, q.relevant_sections, sections) for c in found[:TOP_K])
        rows.append({
            "qa_id": q.qa_id,
            "hit@5": hit,
            "rr@10": reciprocal_rank(hit_list, MRR_K),
            "thin_hit": int(hit and best_overlap < THIN_OVERLAP),
            "top10": " ".join(c.chunk_id for c in found),
        })

    n = len(rows)
    summary = {
        "config": name,
        "chunking": config["chunking"],
        "search": config["search"],
        "n_chunks": len(chunks),
        "n_questions": n,
        "hit@5": sum(r["hit@5"] for r in rows) / n,
        "mrr@10": sum(r["rr@10"] for r in rows) / n,
        "thin_hits": sum(r["thin_hit"] for r in rows),
    }

    folder = EXPERIMENTS_DIR / config["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "per_question.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1]), ensure_ascii=False, indent=2))
