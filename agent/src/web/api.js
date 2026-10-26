// Gọi API cùng địa chỉ: kiểm tra khóa, hỏi theo từng bước, báo câu trả lời sai

export class LoiApi extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function loi(r) {
  if (r.status === 401) return new LoiApi(401, "Khóa truy cập không đúng hoặc đã hết hiệu lực.");
  let detail = `Máy chủ báo lỗi (${r.status}). Vui lòng thử lại.`;
  try {
    const j = await r.json();
    if (typeof j.detail === "string") detail = j.detail;
    // Lỗi kiểm tra đầu vào của FastAPI: danh sách, mỗi lỗi có msg
    else if (Array.isArray(j.detail) && j.detail[0]?.msg) detail = j.detail[0].msg.replace(/^Value error, /, "");
  } catch {}
  return new LoiApi(r.status, detail);
}

const dau = khoa => ({ "Content-Type": "application/json", "X-API-Key": khoa });

export async function kiemKhoa(khoa) {
  const r = await fetch("/auth/check", { headers: dau(khoa) });
  if (!r.ok) throw await loi(r);
}

// EventSource chỉ gửi được GET không header, nên đọc luồng text/event-stream bằng fetch
export async function hoiTungBuoc(khoa, cauHoi, khiCoSuKien, signal) {
  const r = await fetch("/ask/stream", { method: "POST", headers: dau(khoa), body: JSON.stringify({ question: cauHoi }), signal });
  if (!r.ok) throw await loi(r);
  const doc = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let dem = "";
  for (;;) {
    const { value, done } = await doc.read();
    if (done) return;
    dem += value;
    let i;
    while ((i = dem.indexOf("\n\n")) >= 0) {
      const khoi = dem.slice(0, i);
      dem = dem.slice(i + 2);
      let ten = "message", data = "";
      for (const dong of khoi.split("\n")) {
        if (dong.startsWith("event: ")) ten = dong.slice(7);
        else if (dong.startsWith("data: ")) data += dong.slice(6);
      }
      if (data) khiCoSuKien(ten, JSON.parse(data));
    }
  }
}

export async function baoSai(khoa, answerId, comment) {
  const r = await fetch("/feedback", { method: "POST", headers: dau(khoa), body: JSON.stringify({ answer_id: answerId, comment }) });
  if (!r.ok) throw await loi(r);
}
