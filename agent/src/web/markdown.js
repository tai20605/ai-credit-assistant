// Hiển thị Markdown của Gemini (tiêu đề, đoạn, danh sách lồng, đường kẻ, đậm, nghiêng, mã) bằng phần tử DOM;
// chữ chỉ gán qua createTextNode nên thẻ HTML lẫn trong câu trả lời hiện như chữ thường (OWASP: safe sink)

function inline(cha, s) {
  for (const p of s.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)/)) {
    if (!p) continue;
    let tag = null, txt = p;
    if (p.startsWith("**") && p.endsWith("**") && p.length > 4) [tag, txt] = ["strong", p.slice(2, -2)];
    else if (p.startsWith("`") && p.endsWith("`") && p.length > 2) [tag, txt] = ["code", p.slice(1, -1)];
    else if (p.startsWith("*") && p.endsWith("*") && p.length > 2) [tag, txt] = ["em", p.slice(1, -1)];
    if (!tag) { cha.appendChild(document.createTextNode(txt)); continue; }
    const el = document.createElement(tag);
    el.appendChild(document.createTextNode(txt));
    cha.appendChild(el);
  }
}

export function renderMarkdown(cha, md) {
  cha.replaceChildren();
  // Ngăn xếp danh sách đang mở: mỗi tầng ghi độ thụt đầu dòng và phần tử ul/ol
  let ngan = [];
  for (const dong of md.replace(/\r/g, "").split("\n")) {
    if (!dong.trim()) continue;
    const muc = dong.match(/^(\s*)(?:[-*+]|(\d+)[.)])\s+(.*)$/);
    if (muc) {
      const thut = muc[1].length, kieu = muc[2] ? "OL" : "UL";
      while (ngan.length && ngan.at(-1).thut > thut) ngan.pop();
      let tang = ngan.at(-1);
      if (tang && tang.thut === thut && tang.ds.tagName !== kieu) { ngan.pop(); tang = ngan.at(-1); }
      if (!tang || tang.thut < thut) {
        const ds = document.createElement(kieu);
        (tang ? tang.ds.lastElementChild ?? tang.ds : cha).appendChild(ds);
        tang = { thut, ds };
        ngan.push(tang);
      }
      const li = document.createElement("li");
      if (muc[2]) li.value = +muc[2];
      inline(li, muc[3]);
      tang.ds.appendChild(li);
      continue;
    }
    const d = dong.trim();
    // Dòng thường thụt vào ngay dưới một mục: nối vào mục đó
    if (ngan.length && /^\s+/.test(dong)) {
      const li = ngan.at(-1).ds.lastElementChild;
      if (li) { li.appendChild(document.createElement("br")); inline(li, d); continue; }
    }
    ngan = [];
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(d)) { cha.appendChild(document.createElement("hr")); continue; }
    const td = d.match(/^(#{1,6})\s+(.*)$/);
    const el = document.createElement(td ? (td[1].length <= 2 ? "h3" : "h4") : "p");
    inline(el, td ? td[2] : d);
    cha.appendChild(el);
  }
}
