"""Đọc 32 tài liệu AT Credit và tách mỗi tài liệu thành các section (mỗi section một điều)."""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .config import CORPUS_DIR

TITLE = re.compile(r"^# (.+)$", re.MULTILINE)
DOC_ID = re.compile(r"^- Mã văn bản: (\S+)$", re.MULTILINE)
# "## 2. Hạn mức cho vay" -> số 2, tiêu đề "Hạn mức cho vay"
SECTION_HEADING = re.compile(r"^## (\d+)\.\s*(.+)$", re.MULTILINE)


@dataclass
class Section:
    section_id: str
    title: str
    start: int
    end: int


@dataclass
class Document:
    doc_id: str
    title: str
    body: str
    sections: list[Section]

    def sections_in_span(self, start: int, end: int) -> list[str]:
        """Các section giao với body[start:end]; một chunk có thể vắt qua hai section."""
        return [s.section_id for s in self.sections if s.start < end and start < s.end]


def split_sections(doc_id: str, body: str) -> list[Section]:
    """Mỗi section chạy từ heading của nó đến ngay trước heading kế tiếp."""
    headings = list(SECTION_HEADING.finditer(body))
    sections = []
    for i, h in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        sections.append(Section(f"{doc_id}#{h.group(1)}", h.group(2).strip(), h.start(), end))
    return sections


def load_document(path: Path) -> Document:
    body = unicodedata.normalize("NFC", path.read_text(encoding="utf-8-sig")).strip()
    doc_id = DOC_ID.search(body).group(1)
    return Document(doc_id, TITLE.search(body).group(1).strip(), body, split_sections(doc_id, body))


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[Document]:
    return [load_document(p) for p in sorted(corpus_dir.glob("*.md"))]


if __name__ == "__main__":
    docs = load_corpus()
    n_sections = sum(len(d.sections) for d in docs)
    print(f"{len(docs)} tài liệu, {n_sections} section, {sum(len(d.body.split()) for d in docs):,} từ")
