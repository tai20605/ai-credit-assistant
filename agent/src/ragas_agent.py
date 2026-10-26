"""RAGAS Faithfulness + Answer Relevancy cho 12 câu ML+RAG của bộ định tuyến; ngữ cảnh = đầu ra ML Tool + mọi đoạn RAG Tool đã trả trong lượt."""

import csv
import json
import math

from agent.src.danh_gia import EXP, ROUTING
from rag.src.ragas_eval import METRICS, AnswerRelevancy, BgeM3, Faithfulness, GeminiJudge, read_jsonl


def contexts(calls):
    out = []
    for c in calls:
        out += [json.dumps(c["result"], ensure_ascii=False)] if c["tool"] == "predict_risk" else [p["text"] for p in c["result"]]
    return out


def main():
    ids = [r["route_id"] for r in csv.DictReader(ROUTING.open(encoding="utf-8")) if r["label"] == "ML+RAG"]
    runs = {r["route_id"]: r for r in read_jsonl(EXP / "routing_runs.jsonl")}
    path = EXP / "ragas.jsonl"
    rows = {r["route_id"]: r for r in read_jsonl(path)}
    judge = GeminiJudge()
    faithfulness, relevancy = Faithfulness(llm=judge), AnswerRelevancy(llm=judge, embeddings=BgeM3())
    with open(path, "a", encoding="utf-8") as f:
        for k in ids:
            if k in rows:
                continue
            a = runs[k]
            judge.outputs = []
            faith = faithfulness.score(user_input=a["question"], response=a["answer"], retrieved_contexts=contexts(a["tool_calls"])).value
            n = len(judge.outputs)
            row = {"route_id": k, "faithfulness": faith,
                   "answer_relevancy": relevancy.score(user_input=a["question"], response=a["answer"]).value,
                   "faithfulness_judge": judge.outputs[:n], "relevancy_judge": judge.outputs[n:]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows[k] = row
            print(k, f"{faith:.3f}", f"{row['answer_relevancy']:.3f}", flush=True)
    tom_tat = {"n": len(rows)}
    for m in METRICS:
        v = [r[m] for r in rows.values() if not math.isnan(r[m])]
        tom_tat |= {m: sum(v) / len(v), f"{m}_n": len(v)}
    (EXP / "ragas.json").write_text(json.dumps(tom_tat, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(tom_tat, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
