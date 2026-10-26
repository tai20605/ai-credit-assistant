"""So sánh hai cấu hình truy xuất đã chạy bằng paired bootstrap CI 95% trên điểm từng câu."""

import csv
import json
import sys

from .config import EXPERIMENTS_DIR
from .metrics import paired_bootstrap_ci
from .run import CONFIGS


def load_scores(name: str) -> dict[str, dict]:
    with open(EXPERIMENTS_DIR / CONFIGS[name]["folder"] / "per_question.csv", encoding="utf-8") as f:
        return {row["qa_id"]: row for row in csv.DictReader(f)}


def compare(old: str, new: str) -> dict:
    old_scores, new_scores = load_scores(old), load_scores(new)
    result = {"old": old, "new": new}
    for metric in ("rr@10", "hit@5"):
        diff, low, high = paired_bootstrap_ci(
            [float(old_scores[q][metric]) for q in old_scores],
            [float(new_scores[q][metric]) for q in old_scores],
        )
        result[metric] = {"diff": diff, "ci_low": low, "ci_high": high}

    # Chỉ MRR@10 quyết định; CI chứa 0 thì giữ cấu hình cũ
    low, high = result["rr@10"]["ci_low"], result["rr@10"]["ci_high"]
    result["verdict"] = "cải thiện" if low > 0 else "kém đi" if high < 0 else "không đủ bằng chứng"

    path = EXPERIMENTS_DIR / CONFIGS[new]["folder"] / f"vs_{old}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=2))
