"""Sinh câu trả lời: đưa câu hỏi cùng 5 chunk đầu của một cấu hình cho Gemini, lưu câu trả lời từng câu."""

import csv
import json
import os
import sys
from functools import cache

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types

from .chunking import chunk_corpus
from .config import EXPERIMENTS_DIR, LLM_MODEL, ROOT, SEED, TOP_K
from .corpus import load_corpus
from .golden import load_golden
from .run import CONFIGS

SYSTEM = """Bạn là trợ lý nội bộ của AT Credit, công ty tài chính tiêu dùng, trả lời câu hỏi của nhân viên về quy định, quy trình và sản phẩm cho vay.
Chỉ dùng thông tin trong các đoạn trích được cung cấp, không dùng kiến thức bên ngoài.
Nếu các đoạn trích không đủ để trả lời một ý nào đó thì nói rõ là tài liệu được cung cấp không nêu ý đó, không suy đoán.
Trả lời bằng tiếng Việt, ngắn gọn, đi thẳng vào từng ý được hỏi."""


def build_prompt(question: str, contexts: list[str]) -> str:
    passages = "\n\n".join(f"[Đoạn {i}]\n{text}" for i, text in enumerate(contexts, start=1))
    return f"Các đoạn trích:\n\n{passages}\n\nCâu hỏi: {question}"


@cache
def client():
    load_dotenv(ROOT / ".env")
    # vertexai=True: gọi Gemini qua Google Cloud (Agent Platform, tên cũ Vertex AI) thay vì Gemini API của AI Studio
    return genai.Client(vertexai=True, api_key=os.environ["GOOGLE_CLOUD_API_KEY"], http_options=types.HttpOptions(
        # Gặp lỗi quá hạn mức (429) hoặc lỗi máy chủ thì tự chờ rồi gọi lại
        retry_options=types.HttpRetryOptions(attempts=6),
        # Ép kết nối qua IPv4: máy này không ra được mạng bằng IPv6 nên httpx treo ở bước kết nối
        client_args={"transport": httpx.HTTPTransport(local_address="0.0.0.0")},
    ))


def answer(question: str, contexts: list[str]) -> str:
    response = client().models.generate_content(
        model=LLM_MODEL,
        contents=build_prompt(question, contexts),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0,
            seed=SEED,
            # Tắt bước "nghĩ thầm" của Gemini để câu trả lời chỉ phụ thuộc vào prompt
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text.strip()


def generate(name: str) -> int:
    config = CONFIGS[name]
    folder = EXPERIMENTS_DIR / config["folder"]
    chunks = {c.chunk_id: c for c in chunk_corpus(load_corpus(), config["chunking"])}
    # Lấy đúng 5 chunk đã được chấm Hit@5 ở bước truy xuất
    with open(folder / "per_question.csv", encoding="utf-8") as f:
        top = {row["qa_id"]: row["top10"].split()[:TOP_K] for row in csv.DictReader(f)}

    path = folder / "answers.jsonl"
    # Chạy lại thì bỏ qua câu đã có câu trả lời
    done = {json.loads(line)["qa_id"] for line in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    with open(path, "a", encoding="utf-8") as f:
        for q in load_golden():
            if q.qa_id in done:
                continue
            contexts = [chunks[i].text for i in top[q.qa_id]]
            row = {"qa_id": q.qa_id, "question": q.question, "contexts": contexts, "answer": answer(q.question, contexts)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done.add(q.qa_id)
    return len(done)


if __name__ == "__main__":
    print(f"{generate(sys.argv[1])} câu đã có câu trả lời")
