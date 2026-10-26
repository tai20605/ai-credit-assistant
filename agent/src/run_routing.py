"""Chạy tác nhân một lần trên 35 câu định tuyến đã khóa bằng MANIFEST.sha256, ghi mọi câu trả lời và lần gọi công cụ ra routing_runs.jsonl."""

import csv
import hashlib
import json
from pathlib import Path

from agent.src.agent import ask

ROOT = Path(__file__).resolve().parents[2]
ROUTING = ROOT / "data" / "agent" / "routing.csv"
OUT = ROOT / "agent" / "experiments" / "routing_runs.jsonl"
AGENT_SHA = "caa396729010ab827738d958c0569f9c96e74058078b5e580563c2074ee3ca0d"  # agent.py khi cố định prompt và mô tả tool, trước lần chạy chính thức


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert sha(ROUTING) == (ROUTING.parent / "MANIFEST.sha256").read_text().split()[0], "routing.csv đã thay đổi sau khi đóng băng"
    assert sha(ROOT / "agent" / "src" / "agent.py") == AGENT_SHA, "agent.py đã thay đổi sau khi cố định prompt"
    OUT.parent.mkdir(exist_ok=True)
    # Chạy lại (ví dụ sau khi mất mạng) thì bỏ qua câu đã có kết quả, không chạy lại câu nào
    done = {json.loads(l)["route_id"] for l in OUT.read_text(encoding="utf-8").splitlines()} if OUT.exists() else set()
    with open(OUT, "a", encoding="utf-8") as f:
        for r in csv.DictReader(ROUTING.open(encoding="utf-8")):
            if r["route_id"] in done:
                continue
            out = ask(r["question"])
            f.write(json.dumps({"route_id": r["route_id"], **out}, ensure_ascii=False) + "\n")
            f.flush()
            print(r["route_id"], r["label"], sorted({c["tool"] for c in out["tool_calls"]}), flush=True)


if __name__ == "__main__":
    main()
