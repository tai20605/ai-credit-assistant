"""Chấm routing accuracy (tuyến thực tế = tập công cụ đã gọi) và hai chỉ số máy chấm của câu ML+RAG (đúng số liệu hồ sơ, đúng căn cứ chính sách)."""

import csv
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from agent.src.ml_tool import predict_risk
from rag.src.chunking import chunk_corpus
from rag.src.corpus import load_corpus

ROOT = Path(__file__).resolve().parents[2]
ROUTING = ROOT / "data" / "agent" / "routing.csv"
EXP = ROOT / "agent" / "experiments"
TUYEN = {frozenset(): "không gọi công cụ", frozenset({"predict_risk"}): "ML", frozenset({"search_policy"}): "RAG",
         frozenset({"predict_risk", "search_policy"}): "ML+RAG"}


def tuyen(calls):
    return TUYEN[frozenset(c["tool"] for c in calls)]


def co_diem(answer, diem):
    # chấp nhận phẩy hoặc chấm thập phân; điểm có phần thập phân 0 chấp nhận cả dạng số nguyên ("21%")
    dang = {f"{diem:.1f}", f"{diem:.1f}".replace(".", ",")} | ({f"{diem:.0f}"} if diem == round(diem) else set())
    return any(re.search(rf"(?<![\d.,]){re.escape(d)}\s*%", answer) for d in dang)


def co_dai(answer, dai):
    # tên dải phải đứng ngay sau "dải" hoặc "mức" (có thể kèm "rủi ro"), bỏ dấu markdown và dấu hai chấm
    t = re.sub(r"[*_:]", " ", answer.lower())
    t = re.sub(r"\s+", " ", t)
    return dai in re.findall(r"(?:dải|mức) (?:rủi ro )?(thấp|trung bình|cao)", t)


def main():
    rows = {r["route_id"]: r for r in csv.DictReader(ROUTING.open(encoding="utf-8"))}
    runs = {r["route_id"]: r for r in map(json.loads, (EXP / "routing_runs.jsonl").read_text(encoding="utf-8").splitlines())}
    assert set(runs) == set(rows), "chưa chạy đủ 35 câu"
    sec_cua_chunk = {c.chunk_id: set(c.section_ids) for c in chunk_corpus(load_corpus(), "fixed")}

    bang = []
    for k, r in rows.items():
        calls = runs[k]["tool_calls"]
        d = {"route_id": k, "label": r["label"], "loai": r["loai"], "tuyen": tuyen(calls), "so_lan_goi": len(calls),
             "ma_da_goi": ";".join(str(c["args"].get("customer_id")) for c in calls if c["tool"] == "predict_risk")}
        d["dung_tuyen"] = d["tuyen"] == r["label"]
        if r["label"] == "ML+RAG":
            ml = predict_risk(int(r["customer_ids"]))
            ml_agent = [c["result"] for c in calls if c["tool"] == "predict_risk"]
            assert all(m.get("model_version", ml["model_version"]) == ml["model_version"] for m in ml_agent)
            dung = set(r["relevant_section_ids"].split("|"))
            chunks = [p["chunk_id"] for c in calls if c["tool"] == "search_policy" for p in c["result"]]
            d.update(diem=ml["risk_score_pct"], dai=ml["risk_band"].split(" (")[0],
                     dung_so_lieu=co_diem(runs[k]["answer"], ml["risk_score_pct"]) and co_dai(runs[k]["answer"], ml["risk_band"].split(" (")[0]),
                     dung_can_cu=any(sec_cua_chunk[i] & dung for i in chunks))
            d["dat_ca_hai"] = d["dung_so_lieu"] and d["dung_can_cu"]
        bang.append(d)
    df = pd.DataFrame(bang)
    df.to_csv(EXP / "per_question.csv", index=False)

    cot = ["ML", "RAG", "ML+RAG"] + (["không gọi công cụ"] if (df.tuyen == "không gọi công cụ").any() else [])
    ma_tran = pd.crosstab(df.label, df.tuyen).reindex(index=["ML", "RAG", "ML+RAG"], columns=cot, fill_value=0)
    m = df[df.label == "ML+RAG"]
    ket_qua = {
        "routing_accuracy": float(df.dung_tuyen.mean()), "dung_tuyen": int(df.dung_tuyen.sum()), "n": len(df),
        "accuracy_theo_nhan": df.groupby("label").dung_tuyen.mean().to_dict(),
        "accuracy_theo_loai": df.groupby("loai").dung_tuyen.mean().to_dict(),
        "ma_tran_nham_lan": ma_tran.to_dict(orient="index"),
        "ml_rag": {"n": len(m), "dung_so_lieu": int(m.dung_so_lieu.sum()), "dung_can_cu": int(m.dung_can_cu.sum()), "dat_ca_hai": int(m.dat_ca_hai.sum())},
        "so_lan_goi": dict(Counter(df.so_lan_goi.astype(int).tolist())),
    }
    (EXP / "routing.json").write_text(json.dumps(ket_qua, ensure_ascii=False, indent=2), encoding="utf-8")
    print(ma_tran.to_string())
    print(df.drop(columns=["label"]).to_string(index=False))
    print(json.dumps(ket_qua, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
