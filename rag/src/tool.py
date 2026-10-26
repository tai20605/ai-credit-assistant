"""RAG Tool cho tác nhân: nhận một câu hỏi, trả về 5 đoạn chính sách theo cấu hình cuối (cắt fixed, hybrid dense + BM25, rerank)."""

import json
import sys
from functools import cache

from .chunking import chunk_corpus
from .config import RERANK_CANDIDATES, TOP_K
from .corpus import load_corpus
from .retrieval import build_bm25, build_index, hybrid_search, rerank


@cache
def _load():
    # Chỉ chạy ở lần gọi đầu: cắt corpus và nhúng mọi chunk; các lần gọi sau dùng lại index trong bộ nhớ
    docs = load_corpus()
    chunks = {c.chunk_id: c for c in chunk_corpus(docs, "fixed")}
    return {d.doc_id: d for d in docs}, chunks, build_index(list(chunks.values()), "rag_tool"), build_bm25(list(chunks.values()))


def search_policy(question: str) -> list[dict]:
    """Trả về 5 đoạn gần câu hỏi nhất, gần nhất đứng đầu; mỗi đoạn kèm tên tài liệu và tên mục để Agent dẫn nguồn."""
    docs, chunks, index, bm25 = _load()
    candidates = [chunks[i] for i in hybrid_search(index, bm25, list(chunks), question, RERANK_CANDIDATES)]
    results = []
    for chunk_id in rerank(question, candidates, TOP_K):
        chunk = chunks[chunk_id]
        doc = docs[chunk.doc_id]
        titles = {s.section_id: s.title for s in doc.sections}
        results.append({
            "chunk_id": chunk_id,
            "document": doc.title,
            "sections": [titles[s] for s in chunk.section_ids],
            "text": chunk.text,
        })
    return results


if __name__ == "__main__":
    print(json.dumps(search_policy(sys.argv[1]), ensure_ascii=False, indent=2))
