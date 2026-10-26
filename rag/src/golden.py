"""Đọc golden: 80 câu hỏi, mỗi câu kèm đáp án mong đợi và tập section đúng."""

import csv
import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .config import GOLDEN_PATH
from .corpus import load_corpus


@dataclass
class Question:
    qa_id: str
    question: str
    expected_answer: str
    relevant_sections: set[str]


def load_golden(path: Path = GOLDEN_PATH) -> list[Question]:
    text = unicodedata.normalize("NFC", path.read_text(encoding="utf-8-sig"))
    questions = []
    for row in csv.DictReader(io.StringIO(text)):
        # Hit@5 và MRR@10 chỉ cần biết "section này có đúng không", nên gộp ";" và "|" thành một tập
        sections = set(re.split(r"[;|]", row["relevant_section_ids"]))
        questions.append(Question(row["qa_id"], row["question"], row["expected_answer"], sections))
    return questions


if __name__ == "__main__":
    questions = load_golden()
    known = {s.section_id for doc in load_corpus() for s in doc.sections}
    for q in questions:
        assert q.relevant_sections <= known, f"{q.qa_id}: {q.relevant_sections - known}"
    print(f"{len(questions)} câu, mọi section_id đều có trong corpus")
