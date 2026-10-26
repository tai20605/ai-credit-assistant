"""Chấm câu trả lời bằng RAGAS - Faithfulness và Answer Relevancy, giám khảo là Gemini, rồi so sánh hai cấu hình."""

import asyncio
import json
import math
import sys
import types as pytypes

# ragas 0.4.3 import một module mà langchain-community 0.4 đã bỏ; đặt module giả để ragas nạp được (không dùng tới)
sys.modules["langchain_community.chat_models.vertexai"] = pytypes.SimpleNamespace(ChatVertexAI=type("ChatVertexAI", (), {}))

from google.genai import types
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from .config import EXPERIMENTS_DIR, LLM_MODEL, SEED
from .generate import client
from .metrics import paired_bootstrap_ci
from .retrieval import embed
from .run import CONFIGS

METRICS = ("faithfulness", "answer_relevancy")


class GeminiJudge(InstructorBaseRagasLLM):
    """Giám khảo: nhận prompt của RAGAS, bắt Gemini trả về JSON đúng khuôn response_model."""

    def __init__(self):
        self.outputs = []

    def generate(self, prompt, response_model):
        response = client().models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                seed=SEED,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
                response_schema=response_model,
            ),
        )
        output = response_model.model_validate_json(response.text)
        # Giữ lại để lưu ra file: sau này đọc được giám khảo đã tách và phán từng nhận định thế nào
        self.outputs.append(output.model_dump())
        return output

    async def agenerate(self, prompt, response_model):
        # Gọi bản đồng bộ trong luồng phụ để dùng lại đúng client đã ép IPv4 của generate.py
        return await asyncio.to_thread(self.generate, prompt, response_model)


class BgeM3(BaseRagasEmbedding):
    """Answer Relevancy cần nhúng câu hỏi: dùng lại BGE-M3 của bước truy xuất."""

    def embed_text(self, text, **kwargs):
        return embed([text])[0].tolist()

    async def aembed_text(self, text, **kwargs):
        return self.embed_text(text)


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def score(name: str) -> dict[str, dict]:
    """Chấm từng câu trả lời trong answers.jsonl, ghi nối vào ragas.jsonl; chạy lại thì bỏ qua câu đã chấm."""
    folder = EXPERIMENTS_DIR / CONFIGS[name]["folder"]
    path = folder / "ragas.jsonl"
    rows = {row["qa_id"]: row for row in read_jsonl(path)}
    judge = GeminiJudge()
    faithfulness = Faithfulness(llm=judge)
    relevancy = AnswerRelevancy(llm=judge, embeddings=BgeM3())
    with open(path, "a", encoding="utf-8") as f:
        for a in read_jsonl(folder / "answers.jsonl"):
            if a["qa_id"] in rows:
                continue
            judge.outputs = []
            faith = faithfulness.score(user_input=a["question"], response=a["answer"], retrieved_contexts=a["contexts"]).value
            n_faith = len(judge.outputs)
            row = {
                "qa_id": a["qa_id"],
                "faithfulness": faith,
                "answer_relevancy": relevancy.score(user_input=a["question"], response=a["answer"]).value,
                "faithfulness_judge": judge.outputs[:n_faith],
                "relevancy_judge": judge.outputs[n_faith:],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows[row["qa_id"]] = row
            print(row["qa_id"], f"{faith:.3f}", f"{row['answer_relevancy']:.3f}", flush=True)

    summary = {"config": name, "n_questions": len(rows)}
    for metric in METRICS:
        values = [r[metric] for r in rows.values() if not math.isnan(r[metric])]
        summary[metric] = sum(values) / len(values)
        summary[f"{metric}_n"] = len(values)
    (folder / "ragas.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def compare(old: str, new: str) -> dict:
    old_rows, new_rows = score(old), score(new)
    result = {"old": old, "new": new}
    for metric in METRICS:
        # Chỉ so trên những câu cả hai cấu hình đều có điểm
        ids = [q for q in old_rows if q in new_rows and not math.isnan(old_rows[q][metric]) and not math.isnan(new_rows[q][metric])]
        diff, low, high = paired_bootstrap_ci([old_rows[q][metric] for q in ids], [new_rows[q][metric] for q in ids])
        verdict = "cải thiện" if low > 0 else "kém đi" if high < 0 else "không đủ bằng chứng"
        result[metric] = {"n": len(ids), "diff": diff, "ci_low": low, "ci_high": high, "verdict": verdict}
    path = EXPERIMENTS_DIR / CONFIGS[new]["folder"] / f"ragas_vs_{old}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=2))
