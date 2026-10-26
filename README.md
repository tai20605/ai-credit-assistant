# AT Credit Assistant

Trợ lý AI nội bộ cho nhân viên thẩm định tín dụng tiêu dùng. Nhân viên hỏi bằng tiếng Việt tự nhiên, trợ lý tự chọn công cụ phù hợp:

- **Chấm điểm rủi ro hồ sơ** (`predict_risk`): xác suất khó khăn trả nợ, dải rủi ro và các yếu tố chính làm tăng / giảm điểm (LightGBM + SHAP).
- **Tra cứu chính sách** (`search_policy`): trả về nguyên văn các điều khoản quy định nội bộ làm căn cứ (RAG: BM25 + BGE-M3, gộp RRF, xếp hạng lại bằng BGE-Reranker-v2-m3).
- **Câu hỏi kết hợp**: tác nhân LangGraph dùng function calling của Gemini để gọi một hoặc cả hai công cụ rồi tổng hợp câu trả lời có dẫn nguồn.

> AT Credit là công ty **hư cấu**. Toàn bộ quy định trong `data/rag/corpus/` được soạn để mô phỏng một công ty tài chính tiêu dùng theo pháp luật Việt Nam, không phải quy định của tổ chức thật. Điểm rủi ro không dùng để ra quyết định tín dụng thật.

## Kiến trúc

```
Trình duyệt ──► FastAPI (agent/src/api.py) ──► Tác nhân LangGraph (agent/src/agent.py) ◄──► Gemini 3.8 Flash
                 │  xác thực X-API-Key                  │
                 │  chặn số điện thoại / CCCD           ├──► predict_risk   (agent/src/ml_tool.py)
                 │  nhật ký hỏi đáp (JSON, stdout)      │      pipeline tiền xử lý + LightGBM, SHAP
                 └─ giao diện web (agent/src/web/)      └──► search_policy  (rag/src/tool.py)
                                                               BM25 + BGE-M3 (ChromaDB) → RRF → reranker → top-5
```

## Yêu cầu

- Python 3.11, khuyến nghị GPU CUDA (chạy được trên CPU nhưng truy xuất chậm hơn khoảng 13 lần).
- Khóa Google Cloud Vertex AI có quyền gọi Gemini.
- Dữ liệu Home Credit Default Risk (Kaggle) để huấn luyện mô hình rủi ro.

## Cài đặt

```bash
conda create -n credit-ai python=3.11 -y && conda activate credit-ai
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121   # chọn bản theo CUDA, hoặc bản CPU
pip install -r requirements.txt
```

Tải mô hình nhúng và mô hình xếp hạng lại:

```bash
hf download BAAI/bge-m3 --local-dir rag/models/BAAI__bge-m3
hf download BAAI/bge-reranker-v2-m3 --local-dir rag/models/BAAI__bge-reranker-v2-m3
```

Tạo file `.env` ở thư mục gốc:

```
GOOGLE_CLOUD_API_KEY=<khóa Vertex AI>
```

### Huấn luyện mô hình rủi ro

1. Tải dữ liệu từ [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data) (cần chấp nhận điều khoản cuộc thi), giải nén 7 bảng và `HomeCredit_columns_description.csv` vào `data/ml/home_credit/`.
2. Chạy lần lượt các notebook `ml/notebooks/03` → `13`:

```bash
for nb in ml/notebooks/*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```

Kết quả: `ml/data/features.parquet`, `ml/data/split.csv` và `ml/models/LGBM.joblib` mà `predict_risk` cần. Notebook `08_sieu_tham_so` (tinh chỉnh siêu tham số) chạy khoảng 2,7 giờ.

## Chạy

Mọi lệnh chạy từ thư mục gốc.

### Máy chủ web và API

```bash
API_KEY=<khóa truy cập> PYTHONPATH=. uvicorn agent.src.api:app --port 8000
```

Mở `http://localhost:8000` và đăng nhập bằng `API_KEY`. Máy chủ nạp toàn bộ mô hình một lần khi khởi động (khoảng 30 giây), không khởi động nếu thiếu `API_KEY`.

| Phương thức | Đường dẫn | Mô tả |
|---|---|---|
| `GET` | `/health` | Kiểm tra máy chủ |
| `GET` | `/auth/check` | Kiểm tra khóa (204 hoặc 401) |
| `POST` | `/ask` | `{"question": "..."}` (≤ 2.000 ký tự) → `answer`, `tool_calls`, `sources` |
| `POST` | `/ask/stream` | Như `/ask`, trả từng bước qua Server-Sent Events: `tool_start`, `tool_result`, `answer`, `error`, `done` |
| `POST` | `/feedback` | `{"answer_id": "...", "comment": "..."}`: báo câu trả lời sai |

Mọi endpoint trừ `/health` cần header `X-API-Key`. Câu hỏi chứa số giống số điện thoại hoặc số CCCD bị từ chối (422); trợ lý chỉ tra hồ sơ theo mã hồ sơ.

```bash
curl -s localhost:8000/ask -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "Hồ sơ 292498 rủi ro thế nào, có phải trình cấp trên không?"}'
```

### Dòng lệnh

```bash
PYTHONPATH=. python -m agent.src.ml_tool 292498
PYTHONPATH=. python -m rag.src.tool "Khách quá hạn bao nhiêu ngày thì bị chuyển nợ quá hạn?"
PYTHONPATH=. python -m agent.src.agent "Hồ sơ 292498 rủi ro thế nào, có phải trình cấp trên không?"
```

`predict_risk` chỉ phục vụ hồ sơ thuộc tập kiểm tra (mô hình chưa thấy khi huấn luyện) và không bao giờ trả nhãn thật; mã không hợp lệ trả về `{"error": ...}`.

## Đánh giá

### Mô hình rủi ro

Chia ngẫu nhiên phân tầng 80/20 (seed 42), so sánh Logistic Regression, Random Forest và LightGBM. Trên tập kiểm tra 61.503 hồ sơ:

| Mô hình | AUC [CI 95%] | PR-AUC | KS |
|---|---|---|---|
| LightGBM (dùng trong sản phẩm) | 0,7905 [0,7839; 0,7970] | 0,294 | 0,440 |
| Random Forest | 0,7721 | 0,259 | 0,417 |
| Logistic Regression | 0,7706 | 0,260 | 0,405 |

### Truy xuất chính sách

Bộ 80 câu hỏi chuẩn (`data/rag/golden/golden.csv`), mỗi câu gán sẵn các điều khoản đúng. Mỗi bước chỉ đổi một yếu tố; chấp nhận khi CI 95% (paired bootstrap, 10.000 lần) của hiệu MRR@10 nằm hẳn trên 0.

| Cấu hình | Hit@5 | MRR@10 |
|---|---|---|
| B0: cắt cố định 512 token, dense BGE-M3 | 0,875 | 0,736 |
| E3: cắt cố định + hybrid (BM25 + dense, RRF) + reranker (dùng trong sản phẩm) | 0,975 | 0,896 |

Chi tiết từng bước ở `rag/experiments/tong_hop.md`. Chạy lại:

```bash
for c in B0 E1-sentence E1-recursive E2-hybrid E3-rerank; do PYTHONPATH=. python -m rag.src.run $c; done
PYTHONPATH=. python -m rag.src.summary
```

### Tác nhân

Bộ 35 câu hỏi định tuyến (`data/agent/routing.csv`, nhãn ML / RAG / ML+RAG), khóa bằng `data/agent/MANIFEST.sha256`:

- Độ chính xác định tuyến: 34/35 (97,1%).
- Câu ML+RAG: 10/12 câu vừa đúng điểm rủi ro vừa trích đúng điều khoản (kiểm tra tự động).
- RAGAS (Gemini làm giám khảo, chỉ mang tính tham khảo): Faithfulness 0,912, Answer Relevancy 0,802.

```bash
PYTHONPATH=. python -m agent.src.run_routing    
PYTHONPATH=. python -m agent.src.danh_gia
PYTHONPATH=. python -m agent.src.ragas_agent
```

## Cấu trúc thư mục

```
agent/src/            tác nhân, ML Tool, API, giao diện web, script đánh giá tác nhân
agent/experiments/    kết quả chạy bộ câu định tuyến
rag/src/              đọc corpus, cắt đoạn, truy xuất, chấm điểm, RAGAS, RAG Tool
rag/experiments/      kết quả các cấu hình truy xuất
ml/notebooks/         quy trình mô hình rủi ro: chia tập → làm sạch → EDA → đặc trưng → huấn luyện → hiệu chỉnh → ngưỡng → đánh giá → SHAP → kiểm ML Tool
ml/data/              tham số, ngưỡng, tên hiển thị đặc trưng
ml/hinh/              biểu đồ do notebook vẽ
data/rag/corpus/      32 văn bản quy định của AT Credit (Markdown)
data/rag/golden/      bộ câu hỏi chuẩn đánh giá truy xuất
data/agent/           bộ câu hỏi đánh giá định tuyến
```
