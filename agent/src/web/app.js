// Giao diện trợ lý: đăng nhập bằng khóa, gửi câu hỏi, hiện tiến trình từng bước, câu trả lời, thẻ điểm rủi ro, căn cứ và báo sai

import { baoSai, hoiTungBuoc, kiemKhoa } from "./api.js";
import { renderMarkdown } from "./markdown.js";

const $ = id => document.getElementById(id);
const KHOA = "at-credit-khoa";
// Cùng luật với máy chủ (khoản 4 Điều 8 QĐ-26); máy chủ vẫn kiểm lại
const DINH_DANH = /(?<![\d.,])(?<!\d )(?:0\d{9}|\+?84\d{9}|\d{12}|(?:0|\+84[ .-]?)\d{2,3}[ .-]\d{3}[ .-]?\d{3,4})(?!\d)/;
const so = (x, k = 1) => x.toLocaleString("vi-VN", { maximumFractionDigits: k });

let khoa = "";
let dangHoi = null;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function icon(ten) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${ten}`);
  svg.appendChild(use);
  svg.setAttribute("aria-hidden", "true");
  return svg;
}

function nut(ten, nhan, khiBam) {
  const b = el("button", "btn btn-ghost btn-sm");
  b.type = "button";
  b.append(icon(ten), el("span", null, nhan));
  b.onclick = khiBam;
  return b;
}

// sessionStorage: khóa mất khi đóng thẻ; trình duyệt chặn lưu trữ thì vẫn dùng được trong phiên
function docKhoa() { try { return sessionStorage.getItem(KHOA) || ""; } catch { return ""; } }
function ghiKhoa(k) { try { k ? sessionStorage.setItem(KHOA, k) : sessionStorage.removeItem(KHOA); } catch {} }

function hienDangNhap(thongBao = "") {
  khoa = "";
  ghiKhoa("");
  $("ung-dung").hidden = true;
  $("dang-nhap").hidden = false;
  $("loi-khoa").textContent = thongBao;
  $("o-khoa").value = "";
  $("o-khoa").focus();
}

function hienUngDung() {
  $("dang-nhap").hidden = true;
  $("ung-dung").hidden = false;
  $("o-hoi").focus();
}

// ---------- Thẻ kết quả mô hình chấm điểm ----------

const DAI = { thap: ["Rủi ro thấp", "dưới 5%"], tb: ["Rủi ro trung bình", "5% đến dưới 15%"], cao: ["Rủi ro cao", "từ 15%"] };
const maDai = s => (s.startsWith("thấp") ? "thap" : s.startsWith("trung") ? "tb" : "cao");

// Ba dải chia đều thanh; trong mỗi dải đặt điểm theo tỷ lệ (dải cao kéo tới 100%)
function viTri(diem) {
  if (diem < 5) return (diem / 5) * 33.33;
  if (diem < 15) return 33.33 + ((diem - 5) / 10) * 33.33;
  return 66.67 + ((Math.min(diem, 100) - 15) / 85) * 33.33;
}

function cotYeuTo(tieuDe, ds, lonNhat, chieu) {
  const cot = el("div", `factors-col ${chieu}`);
  cot.appendChild(el("h4", null, tieuDe));
  if (!ds.length) cot.appendChild(el("p", "muted", "Không có"));
  const ul = el("ul");
  for (const f of ds) {
    const li = el("li");
    const dau = el("div", "f-head");
    dau.append(el("span", "f-name", f.factor), el("span", "f-value", f.value));
    const thanh = el("div", "f-bar");
    const muc = el("span");
    muc.style.width = `${Math.max(4, (Math.abs(f.impact) / lonNhat) * 100)}%`;
    thanh.title = `Mức ảnh hưởng ${f.impact > 0 ? "+" : ""}${so(f.impact, 3)}`;
    thanh.appendChild(muc);
    li.append(dau, thanh);
    ul.appendChild(li);
  }
  cot.appendChild(ul);
  return cot;
}

function theDiem(kq, maHoSo) {
  const the = el("section", "card risk");
  if (kq.error) {
    the.classList.add("risk-error");
    the.append(el("h3", null, `Hồ sơ ${maHoSo}`), el("p", null, kq.error));
    return the;
  }
  const dai = maDai(kq.risk_band);
  const dau = el("header", "risk-head");
  const ten = el("div");
  ten.append(el("span", "eyebrow", "Kết quả mô hình chấm điểm"), el("h3", null, `Hồ sơ ${kq.customer_id}`));
  dau.append(ten, el("span", `badge badge-${dai}`, DAI[dai][0]));

  const diem = el("div", "risk-score");
  diem.append(el("span", "num", `${so(kq.risk_score_pct)}%`), el("span", "muted", `Điểm rủi ro · dải ${DAI[dai][1]}`));

  const thang = el("div", "scale");
  thang.setAttribute("role", "img");
  thang.setAttribute("aria-label", `Điểm ${so(kq.risk_score_pct)}% trên thang ba dải rủi ro`);
  for (const k of ["thap", "tb", "cao"]) thang.appendChild(el("span", `seg seg-${k}`));
  const kim = el("span", "pin");
  kim.style.left = `${viTri(kq.risk_score_pct)}%`;
  thang.appendChild(kim);
  const nhan = el("div", "scale-labels");
  nhan.append(el("span", null, "Thấp"), el("span", null, "Trung bình"), el("span", null, "Cao"),
              el("span", "mark mark-5", "5%"), el("span", "mark mark-15", "15%"));

  const tat = [...kq.factors_increasing, ...kq.factors_decreasing].map(f => Math.abs(f.impact));
  const lonNhat = Math.max(...tat, 1e-9);
  const yt = el("div", "factors");
  yt.append(cotYeuTo("Yếu tố làm tăng điểm", kq.factors_increasing, lonNhat, "up"),
            cotYeuTo("Yếu tố làm giảm điểm", kq.factors_decreasing, lonNhat, "down"));

  const chu = el("footer", "risk-notes");
  chu.append(el("p", null, kq.note),
             el("p", null, "Không cung cấp con số điểm rủi ro cho khách hàng (khoản 4 Điều 5 QĐ-26)."),
             el("p", "mono", `Phiên bản mô hình ${kq.model_version}`));
  the.append(dau, diem, thang, nhan, yt, chu);
  return the;
}

// ---------- Căn cứ: các đoạn văn bản RAG Tool đã trả ----------

function khoiCanCu(doan) {
  const theoVb = new Map();
  for (const d of doan) {
    if (!theoVb.has(d.document)) theoVb.set(d.document, []);
    theoVb.get(d.document).push(d);
  }
  const khoi = el("details", "card sources");
  khoi.appendChild(el("summary", null, `Căn cứ đã tra: ${doan.length} đoạn từ ${theoVb.size} văn bản`));
  const ds = el("div", "sources-list");
  for (const [vb, cac] of theoVb) {
    const nhom = el("div", "source-doc");
    nhom.appendChild(el("h4", null, vb));
    for (const d of cac) {
      const muc = el("details", "source");
      muc.appendChild(el("summary", null, d.sections.length ? d.sections.join(" · ") : "Phần đầu văn bản"));
      muc.appendChild(el("blockquote", null, d.text.trim()));
      nhom.appendChild(muc);
    }
    ds.appendChild(nhom);
  }
  khoi.appendChild(ds);
  return khoi;
}

// ---------- Một lượt hỏi đáp ----------

const TEN_BUOC = {
  predict_risk: a => `Chấm điểm rủi ro hồ sơ ${a.customer_id}`,
  search_policy: a => `Tra cứu quy định: “${a.question}”`,
};

function luotMoi(cauHoi) {
  $("chao")?.remove();
  const luot = el("article", "turn");
  const hoi = el("div", "q");
  hoi.appendChild(el("div", "bubble", cauHoi));
  const dap = el("div", "a");
  const avatar = el("div", "avatar");
  avatar.appendChild(icon("logo"));
  const than = el("div", "a-body");
  dap.append(avatar, than);
  luot.append(hoi, dap);
  $("luong").appendChild(luot);

  const tienTrinh = el("details", "steps");
  tienTrinh.open = true;
  const tom = el("summary");
  const nhanTom = el("span", null, "Đang xử lý");
  const dongHo = el("span", "timer", "0 giây");
  tom.append(nhanTom, dongHo);
  const ol = el("ol");
  tienTrinh.append(tom, ol);
  const traLoi = el("div", "answer md");
  const cacThe = el("div", "cards");
  const cuoi = el("div", "a-foot");
  than.append(tienTrinh, traLoi, cacThe, cuoi);

  const batDau = performance.now();
  const hen = setInterval(() => { dongHo.textContent = `${Math.round((performance.now() - batDau) / 1000)} giây`; }, 500);
  const buoc = new Map();
  const doan = new Map();
  const doiSo = new Map();
  const cong = [];
  let dangNghi = null;

  function themBuoc(id, nhan) {
    const li = el("li", "step running");
    li.append(el("span", "dot"), el("span", null, nhan));
    ol.appendChild(li);
    buoc.set(id, li);
    luot.scrollIntoView({ block: "end", behavior: "smooth" });
    return li;
  }
  function xong(li) { if (li) { li.classList.replace("running", "done"); li.querySelector(".dot").replaceChildren(icon("check")); } }
  function nghi(nhan) { dangNghi = themBuoc(`nghi-${buoc.size}`, nhan); }
  function ketThuc() {
    clearInterval(hen);
    for (const li of buoc.values()) if (li.classList.contains("running")) li.classList.replace("running", "stopped");
  }

  nghi("Phân tích câu hỏi");

  return {
    luot,
    suKien(ten, d) {
      if (ten === "tool_start") {
        xong(dangNghi);
        cong.push(d.tool);
        doiSo.set(d.id, d.args);
        themBuoc(d.id, TEN_BUOC[d.tool]?.(d.args) ?? d.tool);
      } else if (ten === "tool_result") {
        xong(buoc.get(d.id));
        if (d.tool === "predict_risk") cacThe.appendChild(theDiem(d.result, doiSo.get(d.id)?.customer_id));
        else if (d.tool === "search_policy") {
          for (const x of d.result) doan.set(x.chunk_id, x);
        }
        if (![...buoc.values()].some(li => li.classList.contains("running"))) nghi("Tổng hợp kết quả và soạn câu trả lời");
      } else if (ten === "answer") {
        xong(dangNghi);
        renderMarkdown(traLoi, d.text);
        traLoi.dataset.md = d.text;
      } else if (ten === "done") {
        ketThuc();
        if (doan.size) cacThe.appendChild(khoiCanCu([...doan.values()]));
        tienTrinh.open = false;
        const dung = [...new Set(cong)].map(t => (t === "predict_risk" ? "mô hình chấm điểm" : "tra cứu quy định"));
        nhanTom.textContent = dung.length ? `Đã dùng ${dung.join(" và ")}` : "Không cần tra cứu";
        dongHo.textContent = `${so(d.seconds)} giây`;
        luot.dataset.answerId = d.answer_id;
      } else if (ten === "error") {
        throw new Error(d.detail);
      }
    },
    loi(thongBao, thuLai) {
      ketThuc();
      nhanTom.textContent = "Chưa có câu trả lời";
      const hop = el("div", "alert");
      hop.appendChild(el("p", null, thongBao));
      if (thuLai) hop.appendChild(nut("retry", "Thử lại", thuLai));
      cuoi.replaceChildren(hop);
    },
    hanhDong(hoiLai) {
      const tl = traLoi.dataset.md || "";
      const hang = el("div", "actions");
      const chep = nut("copy", "Sao chép", async () => {
        try { await navigator.clipboard.writeText(tl); chep.lastChild.textContent = "Đã chép"; }
        catch { chep.lastChild.textContent = "Không chép được"; }
        setTimeout(() => { chep.lastChild.textContent = "Sao chép"; }, 1500);
      });
      hang.append(chep, nut("retry", "Hỏi lại", hoiLai), nut("flag", "Báo câu trả lời sai", () => moBaoSai(luot, cuoi)));
      cuoi.replaceChildren(hang);
    },
  };
}

function moBaoSai(luot, cuoi) {
  if (cuoi.querySelector(".report")) return;
  const form = el("form", "report");
  const nhan = el("label", null, "Câu trả lời sai ở đâu? (không bắt buộc, không ghi thông tin cá nhân của khách hàng)");
  const o = el("textarea");
  o.maxLength = 2000;
  o.rows = 2;
  o.id = `bao-${luot.dataset.answerId}`;
  nhan.htmlFor = o.id;
  const gui = el("button", "btn btn-primary btn-sm", "Gửi báo cáo");
  gui.type = "submit";
  const huy = el("button", "btn btn-ghost btn-sm", "Hủy");
  huy.type = "button";
  huy.onclick = () => form.remove();
  const hang = el("div", "report-actions");
  hang.append(huy, gui);
  const tt = el("p", "field-error");
  tt.setAttribute("role", "alert");
  form.append(nhan, o, tt, hang);
  form.onsubmit = async e => {
    e.preventDefault();
    gui.disabled = true;
    try {
      await baoSai(khoa, luot.dataset.answerId, o.value.trim());
      form.replaceWith(el("p", "muted report-done", "Đã gửi báo cáo tới Phòng Chính sách tín dụng. Cảm ơn bạn."));
    } catch (err) {
      if (err.status === 401) return hienDangNhap(err.message);
      tt.textContent = err.message;
      gui.disabled = false;
    }
  };
  cuoi.appendChild(form);
  o.focus();
}

// ---------- Gửi câu hỏi ----------

function trangThaiGui(dangChay) {
  const b = $("nut-gui");
  b.classList.toggle("is-stop", dangChay);
  b.setAttribute("aria-label", dangChay ? "Dừng" : "Gửi câu hỏi");
  b.replaceChildren(icon(dangChay ? "stop" : "send"));
}

async function gui(cauHoi) {
  if (dangHoi || !cauHoi) return;
  if (DINH_DANH.test(cauHoi)) {
    $("canh-bao").textContent = "Câu hỏi có số giống số điện thoại hoặc số CCCD. Trợ lý chỉ tra cứu theo mã hồ sơ (khoản 4 Điều 8 QĐ-26).";
    return;
  }
  $("canh-bao").textContent = "";
  $("o-hoi").value = "";
  coGian();
  const luot = luotMoi(cauHoi);
  dangHoi = new AbortController();
  trangThaiGui(true);
  try {
    await hoiTungBuoc(khoa, cauHoi, luot.suKien, dangHoi.signal);
    luot.hanhDong(() => gui(cauHoi));
  } catch (err) {
    if (err.status === 401) return hienDangNhap(err.message);
    if (err.name === "AbortError") luot.loi("Đã dừng theo yêu cầu.", () => gui(cauHoi));
    else if (err.status === 422) luot.loi(err.message);
    else luot.loi(err instanceof TypeError ? "Mất kết nối tới máy chủ. Kiểm tra mạng rồi thử lại." : err.message, () => gui(cauHoi));
  } finally {
    dangHoi = null;
    trangThaiGui(false);
    luot.luot.scrollIntoView({ block: "end", behavior: "smooth" });
    $("o-hoi").focus();
  }
}

function coGian() {
  const o = $("o-hoi");
  o.style.height = "auto";
  o.style.height = `${Math.min(o.scrollHeight, 200)}px`;
  const n = o.value.length;
  $("dem-ky-tu").textContent = n > 1500 ? `${n}/2000` : "";
}

// ---------- Khởi động ----------

$("form-khoa").onsubmit = async e => {
  e.preventDefault();
  const k = $("o-khoa").value.trim();
  if (!k) { $("loi-khoa").textContent = "Nhập khóa truy cập."; return; }
  $("nut-vao").disabled = true;
  $("loi-khoa").textContent = "";
  try {
    await kiemKhoa(k);
    khoa = k;
    ghiKhoa(k);
    hienUngDung();
  } catch (err) {
    $("loi-khoa").textContent = err instanceof TypeError ? "Không kết nối được máy chủ." : err.message;
  } finally {
    $("nut-vao").disabled = false;
  }
};

$("form-hoi").onsubmit = e => {
  e.preventDefault();
  if (dangHoi) dangHoi.abort();
  else gui($("o-hoi").value.trim());
};
$("o-hoi").addEventListener("input", coGian);
$("o-hoi").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("form-hoi").requestSubmit(); }
});
for (const b of document.querySelectorAll(".suggest button")) b.onclick = () => gui(b.textContent);
$("nut-moi").onclick = () => { if (!dangHoi) location.reload(); };
$("nut-thoat").onclick = () => { dangHoi?.abort(); hienDangNhap(); };

khoa = docKhoa();
if (!khoa) hienDangNhap();
else kiemKhoa(khoa).then(hienUngDung, err => hienDangNhap(err.status === 401 ? err.message : ""));
