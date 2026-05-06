from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from dotenv import load_dotenv
import google.generativeai as genai

from .model import SkinModel

# --- Cấu hình AI Gemini ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel("models/gemini-2.5-flash")

APP_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = APP_ROOT / "model" / "best_model.pth"
LABELS_PATH = APP_ROOT / "model" / "labels.json"

app = FastAPI(title="Skin Diagnosis API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SkinModel(model_path=MODEL_PATH, labels_path=LABELS_PATH)


@app.get("/health")
def health() -> dict:
    status = "ready" if model.is_ready else "model-missing"
    return {"status": status}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    symptoms: Optional[str] = Form(None),
) -> JSONResponse:
    if not model.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model and place it at model/best_model.pth.",
        )

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    try:
        predictions = model.predict(pil_image, top_k=3)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    response = {
        "predictions": predictions,
        "symptoms_used": bool(symptoms and symptoms.strip()),
        "note": "Symptoms are collected for future multimodal fusion.",
    }
    return JSONResponse(content=response)


class ChatRequest(BaseModel):
    diagnosis: str
    confidence: float
    symptoms: str = ""

@app.post("/chat")
async def chat_consultation(request: ChatRequest) -> JSONResponse:
    try:
        symptoms_text = request.symptoms if request.symptoms.strip() else "Không có ghi nhận"
        prompt = f"""
        Bạn là một Bác sĩ da liễu tận tâm, chuyên nghiệp, ngôn từ nhẹ nhàng, an ủi.
        Hệ thống AI phân tích hình ảnh da của bệnh nhân vừa cho ra kết quả dự đoán với:
        - Bệnh lý nghi ngờ: {request.diagnosis}
        - Trọng số tin cậy: {request.confidence}%
        - Triệu chứng của bệnh nhân: {symptoms_text}
        
        Hãy viết một đoạn tư vấn y tế ngắn gọn (khoảng 100-150 chữ) bằng tiếng Việt gửi trực tiếp cho bệnh nhân (xưng hô: Bác sĩ - Bạn).
        Nội dung cần có:
        1. Nhẹ nhàng thông báo kết quả từ AI (nhấn mạnh đây chỉ là kết quả sàng lọc ảnh, không phải là kết luận cuối cùng).
        2. Giải thích cực kỳ ngắn gọn và dễ hiểu bệnh lý này là gì.
        3. Đưa ra lời khuyên chăm sóc/điều trị sơ bộ dựa vào triệu chứng.
        4. Khuyên bệnh nhân nhanh chóng sắp xếp đi khám chuyên khoa trực tiếp để có phác đồ chính xác.
        """
        response = llm_model.generate_content(prompt)
        return JSONResponse(content={"advice": response.text})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM Error: {exc}") from exc

@app.post("/gradcam")
async def gradcam(
    image: UploadFile = File(...),
    class_index: Optional[int] = Form(None),
) -> JSONResponse:
    if not model.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model and place it at model/best_model.pth.",
        )

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    try:
        result = model.gradcam(pil_image, class_index=class_index)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {exc}") from exc

    return JSONResponse(content=result)
