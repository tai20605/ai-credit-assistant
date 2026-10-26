"""Tìm chunk cho một câu hỏi - dense (BGE-M3 trên ChromaDB), hybrid (dense + BM25, gộp bằng RRF), rerank (cross-encoder)."""

from functools import cache

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from underthesea import word_tokenize

from .chunking import Chunk
from .config import EMBEDDING_DIR, HYBRID_CANDIDATES, RERANK_MAX_LENGTH, RERANKER_DIR, RRF_K


@cache
def embedder():
    return SentenceTransformer(str(EMBEDDING_DIR))


def embed(texts: list[str]):
    # Chuẩn hóa về độ dài 1 để khoảng cách cosine của Chroma chỉ phụ thuộc vào hướng của vector
    return embedder().encode(texts, batch_size=8, normalize_embeddings=True)


def build_index(chunks: list[Chunk], name: str) -> chromadb.Collection:
    """Nhúng mọi chunk rồi nạp vào một collection Chroma nằm trong bộ nhớ."""
    collection = chromadb.EphemeralClient().create_collection(name, configuration={"hnsw": {"space": "cosine"}})
    collection.add(ids=[c.chunk_id for c in chunks], embeddings=embed([c.text for c in chunks]))
    return collection


def dense_search(index: chromadb.Collection, question: str, k: int) -> list[str]:
    """Trả về id của k chunk gần câu hỏi nhất, gần nhất đứng đầu."""
    return index.query(query_embeddings=embed([question]), n_results=k)["ids"][0]


def tokenize(text: str) -> list[str]:
    # Tách theo từ tiếng Việt ("tài sản bảo đảm" là một từ), viết thường, bỏ dấu câu
    return [w.lower() for w in word_tokenize(text) if any(ch.isalnum() for ch in w)]


def build_bm25(chunks: list[Chunk]) -> BM25Okapi:
    return BM25Okapi([tokenize(c.text) for c in chunks])


def bm25_search(bm25: BM25Okapi, chunk_ids: list[str], question: str, k: int) -> list[str]:
    """Trả về id của k chunk có điểm BM25 cao nhất; chunk_ids xếp cùng thứ tự với lúc build_bm25."""
    scores = bm25.get_scores(tokenize(question))
    return [chunk_ids[i] for i in scores.argsort(kind="stable")[::-1][:k]]


def rrf(rankings: list[list[str]], k: int) -> list[str]:
    """Gộp nhiều bảng xếp hạng: mỗi chunk nhận 1 / (RRF_K + hạng) từ mỗi bảng có nó, rồi xếp theo tổng."""
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)[:k]


def hybrid_search(index: chromadb.Collection, bm25: BM25Okapi, chunk_ids: list[str], question: str, k: int) -> list[str]:
    return rrf([
        dense_search(index, question, HYBRID_CANDIDATES),
        bm25_search(bm25, chunk_ids, question, HYBRID_CANDIDATES),
    ], k)


@cache
def reranker():
    return CrossEncoder(str(RERANKER_DIR), max_length=RERANK_MAX_LENGTH)


def rerank(question: str, candidates: list[Chunk], k: int) -> list[str]:
    """Cho cross-encoder đọc từng cặp (câu hỏi, chunk) rồi xếp lại ứng viên theo điểm mới; trả về id của k chunk điểm cao nhất."""
    scores = reranker().predict([(question, c.text) for c in candidates], batch_size=4)
    # Điểm bằng nhau thì giữ thứ tự của bước tìm kiếm
    return [candidates[i].chunk_id for i in (-scores).argsort(kind="stable")[:k]]



if __name__ == "__main__":
    import numpy as np

    from .chunking import STRATEGIES, chunk_corpus
    from .config import MRR_K
    from .corpus import load_corpus
    from .golden import load_golden

    # Vector phải dài đúng 1, và HNSW (tìm xấp xỉ) phải cho cùng top-10 với cách so câu hỏi với mọi chunk
    docs, golden = load_corpus(), load_golden()
    questions = embed([q.question for q in golden])
    print(f"model {next(embedder().parameters()).dtype}, vector {questions.shape[1]} chiều")
    for strategy in STRATEGIES:
        chunks = chunk_corpus(docs, strategy)
        ids = [c.chunk_id for c in chunks]
        vectors = embed([c.text for c in chunks])
        norms = np.linalg.norm(np.vstack([vectors, questions]), axis=1)
        index = build_index(chunks, strategy)
        same_list = same_set = 0
        for q, qv in zip(golden, questions):
            exact = [ids[i] for i in (-(vectors @ qv)).argsort(kind="stable")[:MRR_K]]
            approx = dense_search(index, q.question, MRR_K)
            same_list += exact == approx
            same_set += set(exact) == set(approx)
        print(f"{strategy:9s} {len(chunks)} chunk | độ dài vector {norms.min():.6f}–{norms.max():.6f} | "
              f"{len(golden)} câu: top-10 trùng cả thứ tự {same_list}, trùng tập {same_set}")
