"""API HTTP cho Agent và trang web cho nhân viên: hỏi đáp (trả một lần hoặc theo từng bước), kiểm tra khóa, báo câu trả lời sai, nhật ký hỏi đáp (Điều 8 QĐ-26)."""

import json
import logging
import os
import re
import secrets
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field, field_validator

from agent.src import ml_tool
from agent.src.agent import _llm, ask, graph
from rag.src import tool as rag_tool

log = logging.getLogger("uvicorn.error")

# Nhật ký hỏi đáp (khoản 6 Điều 8 QĐ-26): mỗi dòng một JSON trên stdout để công cụ thu log gom lại
nhat_ky = logging.getLogger("at_credit.nhat_ky")
nhat_ky.addHandler(logging.StreamHandler(sys.stdout))
nhat_ky.setLevel(logging.INFO)
nhat_ky.propagate = False


def ghi(loai: str, **truong):
    nhat_ky.info(json.dumps({"severity": "INFO", "message": loai, **truong}, ensure_ascii=False))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("API_KEY"):
        raise RuntimeError("Chưa đặt biến môi trường API_KEY")
    # Nạp mọi thứ nặng (BGE-M3, reranker, index, mô hình LightGBM, client Gemini) một lần trước khi nhận yêu cầu
    t = time.perf_counter()
    rag_tool._load(), ml_tool._load(), _llm(), graph()
    log.info("đã nạp mô hình sau %.0f giây", time.perf_counter() - t)
    yield


app = FastAPI(title="Trợ lý nội bộ AT Credit", lifespan=lifespan)

# OWASP HTTP Headers Cheat Sheet; trang chỉ nạp tài nguyên cùng địa chỉ, không có script/style nội tuyến
BAO_MAT = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


@app.middleware("http")
async def header_bao_mat(request: Request, call_next):
    r = await call_next(request)
    p = request.url.path
    # /docs của FastAPI nạp Swagger UI từ CDN nên không áp CSP
    r.headers.update({k: v for k, v in BAO_MAT.items() if not p.startswith(("/docs", "/redoc"))})
    r.headers.setdefault("Cache-Control", "no-store" if p.startswith(("/ask", "/auth", "/feedback")) else "no-cache")
    return r


def kiem_khoa(x_api_key: str = Header(default="")):
    if not secrets.compare_digest(x_api_key, os.environ["API_KEY"]):
        raise HTTPException(401, "Sai hoặc thiếu X-API-Key")


# Khoản 4 Điều 8 QĐ-26: chỉ tra theo mã hồ sơ (6 chữ số); từ chối số điện thoại (liền hoặc tách nhóm) và số CCCD 12 chữ số.
# Không bắt số đứng sau chữ số + dấu ngăn, để số tiền viết "1 000 000 000" hay "1.000.000.000" không bị chặn nhầm
DINH_DANH = re.compile(r"(?<![\d.,])(?<!\d )(?:0\d{9}|\+?84\d{9}|\d{12}|(?:0|\+84[ .-]?)\d{2,3}[ .-]\d{3}[ .-]?\d{3,4})(?!\d)")


class CauHoi(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def khong_dinh_danh(cls, v: str) -> str:
        if DINH_DANH.search(v):
            raise ValueError("Câu hỏi có số giống số điện thoại hoặc số CCCD. Trợ lý chỉ tra cứu theo mã hồ sơ (khoản 4 Điều 8 QĐ-26).")
        return v


class LanGoi(BaseModel):
    tool: str
    args: dict


class Nguon(BaseModel):
    document: str
    sections: list[str]


class TraLoi(BaseModel):
    answer: str
    tool_calls: list[LanGoi]
    sources: list[Nguon]


class BaoSai(BaseModel):
    answer_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    comment: str = Field(default="", max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/auth/check", status_code=204, dependencies=[Depends(kiem_khoa)])
def kiem_tra_khoa():
    return Response(status_code=204)


# Hàm thường (không async): FastAPI chạy trong luồng phụ nên một câu hỏi lâu không chặn các yêu cầu khác
@app.post("/ask", response_model=TraLoi, dependencies=[Depends(kiem_khoa)])
def hoi(q: CauHoi):
    t = time.perf_counter()
    try:
        out = ask(q.question)
    except Exception:
        log.exception("Agent lỗi")
        raise HTTPException(502, "Không tạo được câu trả lời, vui lòng thử lại")
    calls = out["tool_calls"]
    ghi("hoi_dap", answer_id=uuid.uuid4().hex, question=q.question, answer=out["answer"],
        tools=[{"tool": c["tool"], "args": c["args"]} for c in calls], seconds=round(time.perf_counter() - t, 1))
    nguon = {(p["document"], tuple(p["sections"])) for c in calls if c["tool"] == "search_policy" for p in c["result"]}
    return TraLoi(answer=out["answer"], tool_calls=[LanGoi(tool=c["tool"], args=c["args"]) for c in calls],
                  sources=[Nguon(document=d, sections=list(s)) for d, s in sorted(nguon)])


def su_kien(ten: str, du_lieu: dict) -> str:
    return f"event: {ten}\ndata: {json.dumps(du_lieu, ensure_ascii=False)}\n\n"


# Cùng đồ thị, cùng đầu vào như ask(); chỉ khác là gửi mỗi bước (gọi tool, kết quả tool, câu trả lời) ngay khi xong
@app.post("/ask/stream", dependencies=[Depends(kiem_khoa)])
def hoi_tung_buoc(q: CauHoi):
    def chay():
        t, ma, tools, tra_loi = time.perf_counter(), uuid.uuid4().hex, [], ""
        try:
            for buoc in graph().stream({"messages": [HumanMessage(q.question)]}, stream_mode="updates"):
                for cap_nhat in buoc.values():
                    for m in cap_nhat["messages"]:
                        if isinstance(m, ToolMessage):
                            yield su_kien("tool_result", {"id": m.tool_call_id, "tool": m.name, "result": json.loads(m.content)})
                        elif m.tool_calls:
                            for c in m.tool_calls:
                                tools.append({"tool": c["name"], "args": c["args"]})
                                yield su_kien("tool_start", {"id": c["id"], "tool": c["name"], "args": c["args"]})
                        else:
                            tra_loi = m.text
                            yield su_kien("answer", {"text": tra_loi})
        except Exception:
            log.exception("Agent lỗi")
            yield su_kien("error", {"detail": "Không tạo được câu trả lời, vui lòng thử lại"})
            return
        giay = round(time.perf_counter() - t, 1)
        ghi("hoi_dap", answer_id=ma, question=q.question, answer=tra_loi, tools=tools, seconds=giay)
        yield su_kien("done", {"answer_id": ma, "seconds": giay})

    return StreamingResponse(chay(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})


# Khoản 6 Điều 8 QĐ-26: nhân viên báo câu trả lời sai; ghi kèm mã câu trả lời để đối chiếu với nhật ký hỏi đáp
@app.post("/feedback", status_code=204, dependencies=[Depends(kiem_khoa)])
def bao_sai(b: BaoSai):
    ghi("bao_sai", answer_id=b.answer_id, comment=b.comment)
    return Response(status_code=204)


# Trang web ở "/"; gắn sau cùng để không che các đường dẫn API
app.mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True), name="web")
