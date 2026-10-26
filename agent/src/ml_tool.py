"""ML Tool cho Agent: điểm rủi ro, dải rủi ro và năm yếu tố tăng / giảm mỗi chiều (QĐ-26) của một hồ sơ thuộc tập kiểm tra."""

import hashlib
import json
import sys
from functools import cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA, MODEL = ROOT / "ml" / "data", ROOT / "ml" / "models" / "LGBM.joblib"
DONG = 24  # 1 đơn vị tiền của dữ liệu = 24 đồng (bảng khớp dữ liệu – chính sách)
SO_YEU_TO = 5
GHI_CHU = ("Điểm rủi ro là xác suất ước lượng khách hàng gặp khó khăn trả nợ (Điều 2 QĐ-26), không phải nhóm nợ, không phải nợ xấu. "
           "Mức ảnh hưởng tính trên thang nội bộ của mô hình trước hiệu chỉnh, không cộng lại thành điểm rủi ro (khoản 2 Điều 5 QĐ-26).")


@cache
def _load():
    # nhãn bị bỏ ngay khi đọc; chỉ giữ hồ sơ tập kiểm tra (mô hình chưa thấy)
    f = pd.read_parquet(DATA / "features.parquet").drop(columns="TARGET")
    test = pd.read_csv(DATA / "split.csv").query("tap == 'test'").SK_ID_CURR
    X = f[f.SK_ID_CURR.isin(test)].set_index("SK_ID_CURR")
    ten = pd.read_csv(DATA / "ten_dac_trung.csv", index_col="cot", keep_default_na=False)
    assert set(ten.index) == set(X.columns)
    # Bước hiệu chỉnh (notebook 09) chọn không hiệu chỉnh LightGBM nên xác suất của mô hình chính là điểm đã hiệu chỉnh
    assert json.loads((DATA / "hieu_chinh.json").read_text())["LGBM"]["cach"] == "khong"
    phien_ban = "LGBM-" + hashlib.sha256(MODEL.read_bytes()).hexdigest()[:12]
    return X, ten, joblib.load(MODEL), phien_ban


def _so(v, k=1):
    s = f"{v:,.{k}f}".replace(",", " ").replace(".", ",").replace(" ", ".")
    return s.rstrip("0").rstrip(",") if "," in s else s


def _gia_tri(v, dang, don_vi):
    if pd.isna(v):
        return "không có dữ liệu"
    if dang == "tien":
        dong = v * DONG
        if dong == 0:
            return "0 đồng"
        return f"{_so(dong / 1e6)} triệu đồng" if abs(dong) >= 1e6 else f"{_so(dong / 1e3, 0)} nghìn đồng"
    if dang == "pct":
        return f"{_so(v * 100)}%"
    if dang in ("co", "yn"):
        return "có" if v in (1, "Y") else "không"
    if dang == "ngay":
        return f"{_so(-v, 0)} ngày trước ngày nộp hồ sơ"
    if dang == "diem":
        return _so(v, 3)
    if dang == "chu":
        return str(v)
    return f"{_so(v)} {don_vi}".strip()


def dai_rui_ro(diem):
    return "thấp (dưới 5%)" if diem < 5 else "trung bình (5% đến dưới 15%)" if diem < 15 else "cao (từ 15%)"


def predict_risk(customer_id: int) -> dict:
    """Trả về điểm rủi ro (%) của hồ sơ theo mã hồ sơ, dải rủi ro, tối đa 5 yếu tố làm tăng và 5 yếu tố làm giảm điểm kèm giá trị trong hồ sơ, và phiên bản mô hình."""
    X, ten, pipe, phien_ban = _load()
    try:
        ma = int(customer_id)
    except (TypeError, ValueError):
        ma = None
    if ma not in X.index:
        return {"error": f"Không tra cứu được mã hồ sơ {customer_id}: mã không có trong danh sách hồ sơ được tra cứu."}

    x = X.loc[[ma]]
    xp = pipe[0].transform(x)
    diem = float(np.round(pipe[-1].predict_proba(xp)[0, 1] * 100, 1))  # làm tròn trước khi xếp dải (khoản 2 Điều 3 QĐ-26)
    shap = pd.Series(pipe[-1].booster_.predict(xp, pred_contrib=True)[0, :-1], index=xp.columns)

    def yeu_to(chon):
        return [{"factor": ten.at[c, "ten"], "value": _gia_tri(x.at[ma, c], ten.at[c, "dang"], ten.at[c, "don_vi"]), "impact": round(float(v), 3)}
                for c, v in chon.items()]

    return {
        "customer_id": ma,
        "risk_score_pct": diem,
        "risk_band": dai_rui_ro(diem),
        "factors_increasing": yeu_to(shap[shap > 0].nlargest(SO_YEU_TO)),
        "factors_decreasing": yeu_to(shap[shap < 0].nsmallest(SO_YEU_TO)),
        "model_version": phien_ban,
        "note": GHI_CHU,
    }


if __name__ == "__main__":
    print(json.dumps(predict_risk(sys.argv[1]), ensure_ascii=False, indent=2))
