const fileInput = document.querySelector("#files");
const fileList = document.querySelector("#fileList");
const ocrBtn = document.querySelector("#ocrBtn");
const uploadStatus = document.querySelector("#uploadStatus");

const uploadCard = document.querySelector("#uploadCard");
const selectCard = document.querySelector("#selectCard");
const selectList = document.querySelector("#selectList");
const backBtn = document.querySelector("#backBtn");
const convertBtn = document.querySelector("#convertBtn");

const progressCard = document.querySelector("#progressCard");
const progressFill = document.querySelector("#progressFill");
const progressText = document.querySelector("#progressText");
const progressStage = document.querySelector("#progressStage");
const downloadBtn = document.querySelector("#downloadBtn");
const errorMsg = document.querySelector("#errorMsg");

let jobId = null;
let imagesData = []; // [{index,width,height,boxes:[...]}]
let selection = []; // per image: array of booleans
let pollTimer = null;
let pptBlob = null;
let pptObjectUrl = null;

function show(card) {
  [uploadCard, selectCard, progressCard].forEach((c) => (c.hidden = c !== card));
}

function renderFiles() {
  const files = Array.from(fileInput.files || []);
  fileList.replaceChildren(
    ...files.map((f, i) => {
      const d = document.createElement("div");
      d.textContent = `${i + 1}. ${f.name}`;
      return d;
    })
  );
  ocrBtn.disabled = files.length === 0;
  uploadStatus.textContent = "";
}

fileInput.addEventListener("change", renderFiles);
renderFiles();

// ---- 第一步：OCR ----
ocrBtn.addEventListener("click", async () => {
  const files = Array.from(fileInput.files || []);
  if (!files.length) return;
  const form = new FormData();
  files.forEach((f) => form.append("files", f));

  ocrBtn.disabled = true;
  uploadStatus.textContent = "正在识别文字，请稍等...";
  try {
    const res = await fetch("/api/ocr", { method: "POST", body: form });
    if (!res.ok) {
      const e = await res.json().catch(() => ({ detail: "识别失败" }));
      throw new Error(e.detail || "识别失败");
    }
    const data = await res.json();
    jobId = data.job_id;
    imagesData = data.images;
    // default: select boxes unless flagged keep (logo / background)
    selection = imagesData.map((img) => img.boxes.map((b) => !b.keep));
    renderSelect();
    show(selectCard);
  } catch (e) {
    uploadStatus.textContent = e.message || "识别失败";
  } finally {
    ocrBtn.disabled = false;
  }
});

// ---- 第二步：选择 ----
function renderSelect() {
  selectList.replaceChildren();
  imagesData.forEach((img, imgIdx) => {
    const wrap = document.createElement("div");
    wrap.className = "img-select";

    const toolbar = document.createElement("div");
    toolbar.className = "img-toolbar";
    const label = document.createElement("span");
    label.textContent = `第 ${imgIdx + 1} 页（${img.boxes.length} 处文字）`;
    const allBtn = document.createElement("button");
    allBtn.type = "button";
    allBtn.className = "ghost small";
    allBtn.textContent = "全选";
    allBtn.onclick = () => setAll(imgIdx, true);
    const noneBtn = document.createElement("button");
    noneBtn.type = "button";
    noneBtn.className = "ghost small";
    noneBtn.textContent = "全不选";
    noneBtn.onclick = () => setAll(imgIdx, false);
    toolbar.append(label, allBtn, noneBtn);

    const stage = document.createElement("div");
    stage.className = "img-stage";
    const image = document.createElement("img");
    image.src = `/api/preview/${jobId}/${img.index}`;
    image.alt = `第 ${imgIdx + 1} 页`;

    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", `0 0 ${img.width} ${img.height}`);
    svg.setAttribute("preserveAspectRatio", "none");

    img.boxes.forEach((box, boxIdx) => {
      const poly = document.createElementNS(svgNS, "polygon");
      poly.setAttribute("points", box.box.map((p) => p.join(",")).join(" "));
      poly.classList.add("box");
      if (selection[imgIdx][boxIdx]) poly.classList.add("on");
      const title = document.createElementNS(svgNS, "title");
      title.textContent = box.text;
      poly.appendChild(title);
      poly.addEventListener("click", () => {
        selection[imgIdx][boxIdx] = !selection[imgIdx][boxIdx];
        poly.classList.toggle("on", selection[imgIdx][boxIdx]);
      });
      svg.appendChild(poly);
    });

    stage.append(image, svg);

    const textList = document.createElement("div");
    textList.className = "text-list";
    img.boxes.forEach((box, boxIdx) => {
      const row = document.createElement("label");
      row.className = "text-row";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = selection[imgIdx][boxIdx];
      cb.addEventListener("change", () => {
        selection[imgIdx][boxIdx] = cb.checked;
        svg.children[boxIdx].classList.toggle("on", cb.checked);
      });
      const txt = document.createElement("span");
      txt.textContent = box.text;
      row.append(cb, txt);
      textList.appendChild(row);
    });

    wrap.append(toolbar, stage, textList);
    selectList.appendChild(wrap);
  });
}

function setAll(imgIdx, value) {
  selection[imgIdx] = selection[imgIdx].map(() => value);
  renderSelect();
}

backBtn.addEventListener("click", () => {
  jobId = null;
  imagesData = [];
  selection = [];
  show(uploadCard);
});

// ---- 第三步：生成 ----
convertBtn.addEventListener("click", async () => {
  if (!jobId) return;
  convertBtn.disabled = true;
  try {
    const edit = selection.map((sel) => sel.map((v, i) => (v ? i : null)).filter((v) => v !== null));
    const res = await fetch("/api/convert", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId, edit }),
    });
    if (!res.ok) {
      const e = await res.json().catch(() => ({ detail: "生成失败" }));
      throw new Error(e.detail || "生成失败");
    }
    show(progressCard);
    resetProgress();
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => pollStatus(), 800);
    pollStatus();
  } catch (e) {
    convertBtn.disabled = false;
    errorMsg.textContent = e.message || "生成失败";
    errorMsg.hidden = false;
    show(progressCard);
  }
});

function resetProgress() {
  pptBlob = null;
  if (pptObjectUrl) URL.revokeObjectURL(pptObjectUrl);
  pptObjectUrl = null;
  downloadBtn.hidden = true;
  errorMsg.hidden = true;
  progressFill.style.width = "0%";
  progressText.textContent = "0 / 0";
  progressStage.textContent = "处理中...";
}

async function pollStatus() {
  try {
    const res = await fetch(`/api/status/${jobId}`);
    if (!res.ok) throw new Error("状态查询失败");
    const d = await res.json();
    if (d.status === "error") {
      progressStage.textContent = "处理失败";
      errorMsg.textContent = d.error || "处理失败";
      errorMsg.hidden = false;
      clearInterval(pollTimer);
      convertBtn.disabled = false;
      return;
    }
    if (d.status === "done") {
      progressFill.style.width = "100%";
      progressText.textContent = `${d.total} / ${d.total}`;
      progressStage.textContent = "处理完成！";
      clearInterval(pollTimer);
      await fetchResult();
      convertBtn.disabled = false;
      return;
    }
    const pct = d.total ? Math.round((d.current / d.total) * 100) : 0;
    progressFill.style.width = pct + "%";
    progressFill.setAttribute("aria-valuenow", pct);
    progressText.textContent = `${d.current} / ${d.total}`;
    const names = { OCR: "OCR 识别", inpaint: "擦除文字", PPTX: "生成 PPT" };
    progressStage.textContent = names[d.stage] || d.stage;
  } catch {
    progressStage.textContent = "连接中断";
    clearInterval(pollTimer);
    convertBtn.disabled = false;
  }
}

async function fetchResult() {
  try {
    const res = await fetch(`/api/download/${jobId}`);
    if (!res.ok) throw new Error("下载失败");
    pptBlob = await res.blob();
    downloadBtn.hidden = false;
    document.querySelector("#doneTip").hidden = false;
    downloadBtn.focus();
  } catch (e) {
    errorMsg.textContent = e.message;
    errorMsg.hidden = false;
  }
}

downloadBtn.addEventListener("click", async () => {
  if (!pptBlob) return;
  if ("showSaveFilePicker" in window) {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: "editable-images.pptx",
        types: [{ description: "PowerPoint", accept: { "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"] } }],
      });
      const w = await handle.createWritable();
      await w.write(pptBlob);
      await w.close();
      progressStage.textContent = "已保存";
      return;
    } catch (e) {
      if (e.name === "AbortError") return;
    }
  }
  if (!pptObjectUrl) pptObjectUrl = URL.createObjectURL(pptBlob);
  const a = document.createElement("a");
  a.href = pptObjectUrl;
  a.download = "editable-images.pptx";
  a.click();
});
