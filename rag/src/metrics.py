"""Chấm điểm danh sách chunk mà retriever trả về cho một câu hỏi."""

import numpy as np

from .chunking import Chunk
from .config import N_BOOTSTRAP, SEED
from .corpus import Section


def hits(chunks: list[Chunk], relevant: set[str]) -> list[bool]:
    """Một chunk trúng khi nó giao với ít nhất một section đúng."""
    return [bool(relevant & set(c.section_ids)) for c in chunks]


def hit_at_k(hit_list: list[bool], k: int) -> float:
    return float(any(hit_list[:k]))


def reciprocal_rank(hit_list: list[bool], k: int) -> float:
    for rank, hit in enumerate(hit_list[:k], start=1):
        if hit:
            return 1 / rank
    return 0.0


def overlap_chars(chunk: Chunk, relevant: set[str], sections: dict[str, Section]) -> int:
    """Số ký tự của phần giao lớn nhất giữa chunk và một section đúng (0 nếu chunk không trúng)."""
    return max(
        (min(chunk.end, sections[s].end) - max(chunk.start, sections[s].start) for s in chunk.section_ids if s in relevant),
        default=0,
    )


def paired_bootstrap_ci(old: list[float], new: list[float]) -> tuple[float, float, float]:
    """Hiệu số trung bình (new - old) và khoảng tin cậy 95% của nó; hai danh sách xếp cùng thứ tự câu hỏi."""
    diff = np.array(new) - np.array(old)
    # Mỗi hàng là một lần lấy mẫu lại: rút n câu hỏi có hoàn lại từ n câu hỏi gốc
    picks = np.random.default_rng(SEED).integers(0, len(diff), size=(N_BOOTSTRAP, len(diff)))
    low, high = np.percentile(diff[picks].mean(axis=1), [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)
