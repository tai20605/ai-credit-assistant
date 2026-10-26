"""Gộp kết quả đã lưu của B0, E1, E2, E3 thành một bảng tổng hợp; không chạy lại thí nghiệm nào."""

import csv
import json

from .config import EXPERIMENTS_DIR
from .run import CONFIGS

# (cấu hình, cấu hình đem ra so); mỗi bước so với cấu hình thắng của bước trước
STEPS = [
    ("B0", None),
    ("E1-sentence", "B0"),
    ("E1-recursive", "B0"),
    ("E2-hybrid", "B0"),  # E1 không có cách cắt nào vượt B0 nên giữ fixed
    ("E3-rerank", "E2-hybrid"),
]


def num(x: float, signed=False) -> str:
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def summary() -> str:
    lines = [
        "| Cấu hình | Cắt | Tìm kiếm | Số chunk | Hit@5 | MRR@10 | Hạng 1 | Giao < 100 ký tự | So với | Hiệu MRR@10 [CI 95%] | Kết luận |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, old in STEPS:
        folder = EXPERIMENTS_DIR / CONFIGS[name]["folder"]
        m = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        with open(folder / "per_question.csv", encoding="utf-8") as f:
            rank1 = sum(float(row["rr@10"]) == 1 for row in csv.DictReader(f))
        cells = [name, CONFIGS[name]["chunking"], CONFIGS[name]["search"], str(m["n_chunks"]),
                 num(m["hit@5"]), num(m["mrr@10"]), f"{rank1}/{m['n_questions']}", str(m["thin_hits"])]
        if old:
            vs = json.loads((folder / f"vs_{old}.json").read_text(encoding="utf-8"))
            rr = vs["rr@10"]
            cells += [old, f"{num(rr['diff'], True)} [{num(rr['ci_low'], True)}; {num(rr['ci_high'], True)}]", vs["verdict"]]
        else:
            cells += ["–", "–", "–"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    table = summary()
    (EXPERIMENTS_DIR / "tong_hop.md").write_text(table + "\n", encoding="utf-8")
    print(table)
