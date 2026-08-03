const fileInput = document.querySelector("#files");
const fileList = document.querySelector("#fileList");
const convertBtn = document.querySelector("#convertBtn");
const downloadBtn = document.querySelector("#downloadBtn");
const statusText = document.querySelector("#status");
const uploadCard = document.querySelector("#uploadCard");
const progressCard = document.querySelector("#progressCard");
const progressFill = document.querySelector("#progressFill");
const progressText = document.querySelector("#progressText");
const progressStage = document.querySelector("#progressStage");
const errorMsg = document.querySelector("#errorMsg");
const stageEls = document.querySelectorAll(".stage");

let pptBlob = null;
let pptObjectUrl = null;
let pollTimer = null;

function resetDownload() {
  pptBlob = null;
  if (pptObjectUrl) {
    URL.revokeObjectURL(pptObjectUrl);
    pptObjectUrl = null;
  }
  downloadBtn.hidden = true;
}

function renderFiles() {
  const files = Array.from(fileInput.files || []);
  const items = files.map((file, index) => {
    const item = document.createElement("div");
    item.textContent = `${index + 1}. ${file.name}`;
    return item;
  });
  fileList.replaceChildren(...items);
  convertBtn.disabled = files.length === 0;
  statusText.textContent = "";
  resetDownload();
}

function showProcessing() {
  uploadCard.hidden = true;
  progressCard.hidden = false;
  errorMsg.hidden = true;
  downloadBtn.hidden = true;
  progressFill.style.width = "0%";
  progressFill.setAttribute("aria-valuenow", "0");
  progressText.textContent = "0 / 0";
  progressStage.textContent = "上传完成，开始处理...";
  stageEls.forEach((el) => el.classList.remove("active", "done"));
}

function showUpload() {
  uploadCard.hidden = false;
  progressCard.hidden = true;
  if (pollTimer) clearInterval(pollTimer);
}

function updateProgress(data) {
  const { stage, current, total } = data;
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;

  progressFill.style.width = pct + "%";
  progressFill.setAttribute("aria-valuenow", pct);
  progressText.textContent = `${current} / ${total}`;

  const stageNames = { OCR: "OCR 文字识别", inpaint: "擦除原文文字", PPTX: "生成 PPT 文件" };
  progressStage.textContent = stageNames[stage] || stage;

  stageEls.forEach((el) => {
    const s = el.dataset.stage;
    el.classList.remove("active", "done");
    if (s === stage) el.classList.add("active");
    if (stage === "PPTX" && s === "PPTX") el.classList.add("active");
    if (stage === "complete" || (stage === "PPTX" && (s === "OCR" || s === "inpaint"))) {
      el.classList.add("done");
    }
  });
}

async function pollStatus(jobId) {
  try {
    const res = await fetch(`/api/status/${jobId}`);
    if (!res.ok) throw new Error("状态查询失败");
    const data = await res.json();

    if (data.status === "error") {
      errorMsg.textContent = data.error || "处理失败";
      errorMsg.hidden = false;
      progressStage.textContent = "处理失败";
      if (pollTimer) clearInterval(pollTimer);
      return;
    }

    if (data.status === "done") {
      updateProgress({ stage: "complete", current: data.total, total: data.total });
      progressStage.textContent = "处理完成！";
      if (pollTimer) clearInterval(pollTimer);
      await fetchResult(jobId);
      return;
    }

    updateProgress(data);
  } catch {
    progressStage.textContent = "连接中断";
    if (pollTimer) clearInterval(pollTimer);
  }
}

async function fetchResult(jobId) {
  try {
    const res = await fetch(`/api/download/${jobId}`);
    if (!res.ok) throw new Error("下载失败");
    pptBlob = await res.blob();
    downloadBtn.hidden = false;
    downloadBtn.focus();
  } catch (e) {
    errorMsg.textContent = e.message;
    errorMsg.hidden = false;
  }
}

async function savePpt() {
  if (!pptBlob) return;
  if ("showSaveFilePicker" in window) {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: "editable-images.pptx",
        types: [{ description: "PowerPoint", accept: { "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"] } }],
      });
      const writable = await handle.createWritable();
      await writable.write(pptBlob);
      await writable.close();
      progressStage.textContent = "已保存";
      return;
    } catch (e) {
      if (e.name === "AbortError") throw e;
    }
  }
  if (!pptObjectUrl) pptObjectUrl = URL.createObjectURL(pptBlob);
  const link = document.createElement("a");
  link.href = pptObjectUrl;
  link.download = "editable-images.pptx";
  link.click();
}

fileInput.addEventListener("change", renderFiles);
renderFiles();

downloadBtn.addEventListener("click", async () => {
  try { await savePpt(); }
  catch (e) { if (e.name !== "AbortError") progressStage.textContent = "保存失败"; }
});

convertBtn.addEventListener("click", async () => {
  const files = Array.from(fileInput.files || []);
  if (!files.length) return;

  const formData = new FormData();
  files.forEach((f) => formData.append("files", f));

  resetDownload();
  convertBtn.disabled = true;
  statusText.textContent = "正在上传...";

  try {
    const res = await fetch("/api/convert", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "上传失败" }));
      throw new Error(err.detail || "上传失败");
    }
    const { job_id } = await res.json();
    showProcessing();
    pollTimer = setInterval(() => pollStatus(job_id), 800);
    pollStatus(job_id);
  } catch (e) {
    statusText.textContent = e.message;
    convertBtn.disabled = false;
  }
});