const form = document.getElementById("predict-form");
const results = document.getElementById("result-list");
const panel = document.getElementById("results");
const clearButton = document.getElementById("clear-form");
const quickDemo = document.getElementById("quick-demo");
const gradcamToggle = document.getElementById("use-gradcam");
const gradcamPanel = document.getElementById("gradcam");
const gradcamImage = document.getElementById("gradcam-image");

const aiConsultationPanel = document.getElementById("ai-consultation");
const aiAdviceText = document.getElementById("ai-advice-text");

const API_URL = "http://127.0.0.1:8000/predict";
const GRADCAM_URL = "http://127.0.0.1:8000/gradcam";
const CHAT_URL = "http://127.0.0.1:8000/chat";

const LABEL_INFO = {
  akiec: {
    name: "Dày sừng ánh sáng / ung thư biểu mô tế bào vảy tại chỗ",
    desc: "Tổn thương do tia UV; nguy cơ tiến triển nếu không xử lý.",
  },
  bcc: {
    name: "Ung thư biểu mô tế bào đáy",
    desc: "Thường tiến triển chậm, ít di căn nhưng cần điều trị.",
  },
  bkl: {
    name: "Dày sừng tiết bã / tổn thương lành tính",
    desc: "Tổn thương lành tính thường gặp ở người lớn tuổi.",
  },
  df: {
    name: "U xơ da",
    desc: "Nốt lành tính, chắc, thường ở chi dưới.",
  },
  mel: {
    name: "U hắc tố ác tính",
    desc: "Tổn thương ác tính cần chẩn đoán và can thiệp sớm.",
  },
  nv: {
    name: "Nốt ruồi sắc tố",
    desc: "Tổn thương lành tính phổ biến, theo dõi thay đổi bất thường.",
  },
  vasc: {
    name: "Tổn thương mạch máu",
    desc: "Liên quan mạch máu, đa số lành tính.",
  },
};

const showMessage = (message) => {
  results.innerHTML = `<p>${message}</p>`;
};

const renderResults = (predictions) => {
  results.innerHTML = "";
  predictions.forEach((item) => {
    const card = document.createElement("div");
    card.className = "result-card";
    const percent = Math.round(item.probability * 100);
    const info = LABEL_INFO[item.label] || {
      name: item.label.toUpperCase(),
      desc: "",
    };
    card.innerHTML = `
      <strong>${info.name}</strong>
      ${info.desc ? `<small>${info.desc}</small>` : ""}
      <div class="progress"><span style="width: ${percent}%"></span></div>
      <small>${percent}% độ tin cậy</small>
    `;
    results.appendChild(card);
  });
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.getElementById("image");
  if (!fileInput.files.length) {
    showMessage("Vui lòng chọn một ảnh.");
    return;
  }

  const formData = new FormData(form);
  const symptomsText = formData.get("symptoms") || "";
  panel.querySelector("p").textContent = "Đang suy luận...";
  showMessage("Đang xử lý...");
  gradcamPanel.style.display = "none";
  aiConsultationPanel.style.display = "none";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Request failed");
    }

    const data = await response.json();
    renderResults(data.predictions);
    panel.querySelector("p").textContent = "Đã có kết quả";

    // Phân tích với Bác sĩ AI (Gemini)
    if (data.predictions && data.predictions.length > 0) {
      aiConsultationPanel.style.display = "block";
      aiAdviceText.innerHTML = "<em>Đang kết nối Bác sĩ AI (Gemini) để phân tích...</em>";
      
      const topPrediction = data.predictions[0];
      const topName = LABEL_INFO[topPrediction.label]?.name || topPrediction.label;
      const confidencePercent = parseFloat((topPrediction.probability * 100).toFixed(2));

      fetch(CHAT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          diagnosis: topName,
          confidence: confidencePercent,
          symptoms: symptomsText
        })
      })
      .then(res => res.json())
      .then(chatData => {
         // Format the text specifically for HTML (convert newlines to <br>)
         aiAdviceText.innerHTML = (chatData.advice || "Không nhận được phản hồi").replace(/\n/g, '<br>');
      })
      .catch(chatErr => {
         aiAdviceText.innerHTML = "<em>Xin lỗi, kết nối tới Bác sĩ AI đang bị gián đoạn. Vui lòng tham khảo kết quả ở trên.</em>";
         console.error("Chat API error:", chatErr);
      });
    }

    if (gradcamToggle.checked) {
      const gradcamData = new FormData();
      gradcamData.append("image", fileInput.files[0]);
      const gradcamResponse = await fetch(GRADCAM_URL, {
        method: "POST",
        body: gradcamData,
      });

      if (gradcamResponse.ok) {
        const gradcamResult = await gradcamResponse.json();
        gradcamImage.src = `data:image/png;base64,${gradcamResult.overlay_base64}`;
        gradcamPanel.style.display = "grid";
      }
    }
  } catch (error) {
    showMessage(`Error: ${error.message}`);
    panel.querySelector("p").textContent = "Dự đoán thất bại";
  }
});

if (clearButton) {
  clearButton.addEventListener("click", () => {
    form.reset();
    panel.querySelector("p").textContent = "Đang chờ suy luận...";
    results.innerHTML = "";
    gradcamPanel.style.display = "none";
    aiConsultationPanel.style.display = "none";
  });
}

if (quickDemo) {
  quickDemo.addEventListener("click", () => {
    showMessage("Hãy tải ảnh da soi thật để chạy suy luận.");
  });
}
