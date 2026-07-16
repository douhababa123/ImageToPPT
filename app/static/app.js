const fileInput = document.querySelector("#files");
const fileList = document.querySelector("#fileList");
const convertBtn = document.querySelector("#convertBtn");
const downloadBtn = document.querySelector("#downloadBtn");
const statusText = document.querySelector("#status");

let pptBlob = null;
let pptObjectUrl = null;

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

async function savePpt() {
  if (!pptBlob) return;

  if ("showSaveFilePicker" in window) {
    try {
      const fileHandle = await window.showSaveFilePicker({
        suggestedName: "editable-images.pptx",
        types: [
          {
            description: "PowerPoint 演示文稿",
            accept: {
              "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
            },
          },
        ],
      });
      const writable = await fileHandle.createWritable();
      await writable.write(pptBlob);
      await writable.close();
      statusText.textContent = "已保存 PPT。";
      return;
    } catch (error) {
      if (error.name === "AbortError") {
        throw error;
      }
      console.warn("File picker save failed, falling back to browser download.", error);
    }
  }

  if (!pptObjectUrl) {
    pptObjectUrl = URL.createObjectURL(pptBlob);
  }
  const link = document.createElement("a");
  link.href = pptObjectUrl;
  link.download = "editable-images.pptx";
  link.click();
  statusText.textContent = "当前浏览器不允许选择保存位置，已改用默认下载。";
}

fileInput.addEventListener("change", renderFiles);
renderFiles();

downloadBtn.addEventListener("click", async () => {
  try {
    await savePpt();
  } catch (error) {
    if (error.name === "AbortError") {
      statusText.textContent = "已取消保存。";
      return;
    }
    statusText.textContent = error.message || "保存失败。";
  }
});

convertBtn.addEventListener("click", async () => {
  const files = Array.from(fileInput.files || []);
  if (!files.length) return;

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  resetDownload();
  convertBtn.disabled = true;
  statusText.textContent = "正在识别文字、擦除原文并生成 PPT，请稍等...";

  try {
    const response = await fetch("/api/convert", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "生成失败" }));
      throw new Error(error.detail || "生成失败");
    }

    pptBlob = await response.blob();
    downloadBtn.hidden = false;
    downloadBtn.focus();
    statusText.textContent = "已完成，请点击“保存 PPT”选择保存位置。";
  } catch (error) {
    statusText.textContent = error.message || "生成失败。";
  } finally {
    convertBtn.disabled = files.length === 0;
  }
});
