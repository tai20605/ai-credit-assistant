"""Cắt tài liệu thành chunk theo 3 cách của E1: fixed, sentence, recursive."""

from dataclasses import dataclass
from functools import cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from underthesea import sent_tokenize

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIR
from .corpus import Document

# Recursive thử cắt ở ranh giới lớn trước (heading điều), rồi mới xuống đoạn, dòng (khoản, điểm), câu, từ
SEPARATORS = ["\n## ", "\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    chunk_id: str  # "QĐ-04::003"
    doc_id: str
    text: str
    start: int
    end: int
    section_ids: list[str]


@cache
def tokenizer():
    return AutoTokenizer.from_pretrained(EMBEDDING_DIR)


def count_tokens(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False))


def fixed_spans(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, base=0) -> list[tuple[int, int]]:
    """Cửa sổ trượt đúng `size` token, mỗi lần dịch đi `size - overlap` token."""
    # offsets[i] = (vị trí ký tự bắt đầu, vị trí ký tự kết thúc) của token thứ i
    offsets = tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    spans, start = [], 0
    while True:
        end = min(start + size, len(offsets))
        spans.append((base + offsets[start][0], base + offsets[end - 1][1]))
        if end == len(offsets):
            return spans
        start = end - overlap


def split_sentences(body: str) -> list[tuple[int, int]]:
    """Vị trí từng câu. Tách theo dòng trước, vì heading và gạch đầu dòng không có dấu chấm."""
    sentences, line_start = [], 0
    for line in body.split("\n"):
        if line.strip():
            sentences += sentences_in_line(line, line_start)
        line_start += len(line) + 1
    return sentences


def sentences_in_line(line: str, line_start: int) -> list[tuple[int, int]]:
    found, cursor = [], 0
    for sent in sent_tokenize(line):
        sent = sent.strip()
        i = line.find(sent, cursor)
        if not sent or i < 0:
            found = []  # underthesea làm biến đổi chuỗi, không dò lại được vị trí
            break
        found.append((line_start + i, line_start + i + len(sent)))
        cursor = i + len(sent)
    if found:
        return found
    # Không tách được thì coi cả dòng là một câu
    i = line.find(line.strip())
    return [(line_start + i, line_start + i + len(line.strip()))]


def sentence_spans(body: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[tuple[int, int]]:
    """Gom các câu liền nhau đến khi đủ `size` token; không bao giờ cắt giữa câu."""
    sents = []  # (start, end, số token)
    for s, e in split_sentences(body):
        n = count_tokens(body[s:e])
        if n <= size:
            sents.append((s, e, n))
        else:  # câu dài hơn cả một chunk (hiếm): buộc phải cắt theo token
            sents += [(ws, we, count_tokens(body[ws:we])) for ws, we in fixed_spans(body[s:e], size, 0, base=s)]

    spans, i = [], 0
    while True:
        j, total = i, 0
        while j < len(sents) and (j == i or total + sents[j][2] <= size):
            total += sents[j][2]
            j += 1
        spans.append((sents[i][0], sents[j - 1][1]))
        if j == len(sents):
            return spans
        # Overlap: chunk sau bắt đầu lại từ vài câu cuối của chunk trước (tổng <= overlap token)
        k, back = j, 0
        while k - 1 > i and back + sents[k - 1][2] <= overlap:
            back += sents[k - 1][2]
            k -= 1
        i = k


def recursive_spans(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[tuple[int, int]]:
    """Cắt ở ranh giới lớn nhất có thể; chỉ xuống ranh giới nhỏ hơn khi đoạn vẫn quá dài."""
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer(), chunk_size=size, chunk_overlap=overlap, separators=SEPARATORS, keep_separator="start"
    )
    # LangChain chỉ trả về văn bản, nên phải tự dò lại vị trí của từng chunk
    spans, prev = [], -1
    for piece in splitter.split_text(text):
        start = text.find(piece, prev + 1)
        assert start >= 0, f"Không dò được vị trí chunk: {piece[:60]!r}"
        spans.append((start, start + len(piece)))
        prev = start
    return spans


STRATEGIES = {"fixed": fixed_spans, "sentence": sentence_spans, "recursive": recursive_spans}


def chunk_corpus(docs: list[Document], strategy: str) -> list[Chunk]:
    chunks = []
    for doc in docs:
        for i, (start, end) in enumerate(STRATEGIES[strategy](doc.body)):
            chunk_id = f"{doc.doc_id}::{i:03d}"
            chunks.append(Chunk(chunk_id, doc.doc_id, doc.body[start:end], start, end, doc.sections_in_span(start, end)))
    return chunks


if __name__ == "__main__":
    from statistics import median

    from .corpus import load_corpus

    docs = load_corpus()
    sec_tokens = [count_tokens(d.body[s.start:s.end]) for d in docs for s in d.sections]
    print(f"corpus {sum(count_tokens(d.body) for d in docs):,} token; section: trung vị {median(sec_tokens):.0f}, "
          f"max {max(sec_tokens)}, {sum(n > CHUNK_SIZE for n in sec_tokens)} section > {CHUNK_SIZE} token")
    for strategy in STRATEGIES:
        chunks = chunk_corpus(docs, strategy)
        n = [count_tokens(c.text) for c in chunks]
        heading_only = sum(c.text.startswith("## ") and "\n" not in c.text for c in chunks)
        print(f"{strategy:9s} {len(chunks)} chunk | token min {min(n)} trung vị {median(n):.0f} max {max(n)} | "
              f"vắt >= 2 section {sum(len(c.section_ids) > 1 for c in chunks)} | không thuộc section {sum(not c.section_ids for c in chunks)} | "
              f"chỉ có heading {heading_only}")
