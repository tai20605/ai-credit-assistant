"""Agent (A4): đồ thị trạng thái LangGraph, Gemini tự chọn gọi ML Tool, RAG Tool hoặc cả hai rồi tổng hợp câu trả lời."""

import json
import sys
from functools import cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph

from agent.src.ml_tool import predict_risk as _predict_risk
from rag.src.config import LLM_MODEL, SEED
from rag.src.generate import client
from rag.src.tool import search_policy as _search_policy

SYSTEM = """Bạn là trợ lý nội bộ của AT Credit, công ty tài chính tiêu dùng, trả lời câu hỏi của nhân viên.
Bạn có hai công cụ:
- predict_risk: kết quả mô hình chấm điểm rủi ro của một hồ sơ cụ thể, tra theo mã hồ sơ.
- search_policy: tra cứu quy định, quy trình và sản phẩm trong văn bản nội bộ của AT Credit.
Xác định từng vế của câu hỏi cần thông tin gì rồi gọi công cụ tương ứng; một câu hỏi có thể cần một hoặc cả hai công cụ, hoặc gọi một công cụ nhiều lần.
Chỉ dùng thông tin do công cụ trả về, không dùng kiến thức bên ngoài và không tự đoán số liệu của hồ sơ.
Nếu công cụ báo lỗi hoặc kết quả không đủ để trả lời một vế thì nói rõ điều đó.
Khi nêu quy định thì ghi tên văn bản và tên mục làm căn cứ.
Trả lời bằng tiếng Việt, ngắn gọn, đi thẳng vào từng vế được hỏi."""


@tool
def predict_risk(customer_id: int) -> str:
    """Tra kết quả mô hình chấm điểm rủi ro của một hồ sơ vay theo mã hồ sơ (số nguyên).
    Trả về điểm rủi ro (%), dải rủi ro, tối đa 5 yếu tố làm tăng và 5 yếu tố làm giảm điểm kèm giá trị trong hồ sơ, và phiên bản mô hình;
    hoặc thông báo lỗi nếu mã hồ sơ không tra cứu được."""
    return json.dumps(_predict_risk(customer_id), ensure_ascii=False)


@tool
def search_policy(question: str) -> str:
    """Tìm trong văn bản nội bộ của AT Credit (quy chế, quy định, sản phẩm, trích văn bản pháp luật) 5 đoạn liên quan nhất tới câu hỏi.
    Mỗi đoạn kèm tên văn bản và tên mục. Đối số là câu hỏi về quy định, viết bằng tiếng Việt."""
    return json.dumps(_search_policy(question), ensure_ascii=False)


TOOLS = {t.name: t for t in (predict_risk, search_policy)}


@cache
def _llm():
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, seed=SEED, thinking_budget=0, vertexai=True, api_key="-")
    # Dùng chung client của phần RAG (Vertex, khóa trong .env, ép IPv4, tự gọi lại khi 429/5xx) thay cho client LangChain tự tạo
    llm.client = client()
    return llm.bind_tools(list(TOOLS.values()))


def call_model(state: MessagesState):
    return {"messages": [_llm().invoke([SystemMessage(SYSTEM)] + state["messages"])]}


def run_tools(state: MessagesState):
    return {"messages": [ToolMessage(TOOLS[c["name"]].invoke(c["args"]), tool_call_id=c["id"], name=c["name"])
                         for c in state["messages"][-1].tool_calls]}


def next_step(state: MessagesState):
    return "run_tools" if state["messages"][-1].tool_calls else END


@cache
def graph():
    g = StateGraph(MessagesState)
    g.add_node("call_model", call_model)
    g.add_node("run_tools", run_tools)
    g.add_edge(START, "call_model")
    g.add_conditional_edges("call_model", next_step, ["run_tools", END])
    g.add_edge("run_tools", "call_model")
    return g.compile()


def ask(question: str) -> dict:
    """Chạy Agent cho một câu hỏi; trả về câu trả lời cuối và mọi lần gọi công cụ (tên, đối số, kết quả) theo thứ tự."""
    messages = graph().invoke({"messages": [HumanMessage(question)]})["messages"]
    results = {m.tool_call_id: json.loads(m.content) for m in messages if isinstance(m, ToolMessage)}
    calls = [{"tool": c["name"], "args": c["args"], "result": results[c["id"]]}
             for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
    return {"question": question, "answer": messages[-1].text, "tool_calls": calls}


if __name__ == "__main__":
    out = ask(sys.argv[1])
    for c in out["tool_calls"]:
        print(f"[{c['tool']}] {json.dumps(c['args'], ensure_ascii=False)}")
    print(out["answer"])
