from pathlib import Path

# --- Đường dẫn ---
ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "data" / "rag" / "corpus"
GOLDEN_PATH = ROOT / "data" / "rag" / "golden" / "golden.csv"
EXPERIMENTS_DIR = ROOT / "rag" / "experiments"

# --- Model ---
EMBEDDING_DIR = ROOT / "rag" / "models" / "BAAI__bge-m3"
RERANKER_DIR = ROOT / "rag" / "models" / "BAAI__bge-reranker-v2-m3"
LLM_MODEL = "gemini-3.8-flash"

# --- Tham số giữ nguyên ở mọi thí nghiệm ---
SEED = 42
TOP_K = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
RERANK_CANDIDATES = 20 # hybrid nộp bấy nhiêu chunk cho reranker chấm lại
RERANK_MAX_LENGTH = 1024 # token của cặp (câu hỏi, chunk); đủ lớn để không cặp nào bị cắt cụt
HYBRID_CANDIDATES = 20 # mỗi nhánh (dense, BM25) nộp bấy nhiêu chunk cho RRF
RRF_K = 60
MRR_K = 10
N_BOOTSTRAP = 10_000
THIN_OVERLAP = 100 # số ký tự: phần giao nhỏ hơn mức này coi là "chỉ chạm mép" section
