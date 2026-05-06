# SmartskinAI - Hệ Thống Chẩn Đoán Bệnh Da Liễu AI

<div align="center">

![Python](https://img.shields.io/badge/PYTHON-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PYTORCH-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FASTAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Vanilla JS](https://img.shields.io/badge/VANILLA%20JS-ES6-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
<br>
![EfficientNet-B0](https://img.shields.io/badge/EFFICIENTNET--B0-TRANSFER%20LEARNING-8BC34A?style=for-the-badge)
![Grad-CAM](https://img.shields.io/badge/XAI-GRAD--CAM-FF9800?style=for-the-badge)

</div>

> Hệ thống chẩn đoán và sàng lọc bệnh da liễu thông minh, kết hợp hình ảnh da soi và mô tả triệu chứng lâm sàng. Tích hợp công nghệ giải thích mô hình (Explainable AI - XAI) với Grad-CAM.

---

## Mục lục
- [1. Tổng quan Dự án](#1-tổng-quan-dự-án)
- [2. Công nghệ Áp dụng](#2-công-nghệ-áp-dụng)
- [3. Cấu trúc Dự án](#3-cấu-trúc-dự-án)
- [4. Kiến trúc Hệ thống](#4-kiến-trúc-hệ-thống)
- [5. Luồng Hoạt Động](#5-luồng-hoạt-động)
- [6. Mô tả Dữ liệu Căn bản](#6-mô-tả-dữ-liệu-căn-bản)
- [7. Kiến trúc Mô hình Giải thích](#7-kiến-trúc-mô-hình-giải-thích)
- [8. Đánh giá và Kết quả](#8-đánh-giá-và-kết-quả)
- [9. Hướng dẫn Cài đặt & Vận hành](#9-hướng-dẫn-cài-đặt--vận-hành)
- [10. Khai thác API Dịch vụ](#10-khai-thác-api-dịch-vụ)
- [11. Thông tin Tác giả](#11-thông-tin-tác-giả)

---

## 1. Tổng quan Dự án

Bệnh ung thư da và các cấu trúc tổn thương thường rất dễ nhầm lẫn. Hệ thống này được phát triển để hỗ trợ các bác sĩ và bệnh nhân sàng lọc, phân tích 7 loại bệnh da liễu phổ biến nhất với độ tin cậy được bóc tách bởi vùng giải thích **Grad-CAM**.

**Tính năng nổi bật:**
- **Nhận diện Tự động:** Phân tích độ chính xác trên 7 nhóm tổn thương (HAM10000 dataset).
- **Hỗ trợ Triệu chứng:** Bổ sung dữ liệu lâm sàng giúp quá trình phán đoán có cơ sở.
- **Explainable AI (Grad-CAM):** Trực quan hóa vùng tập trung của mô hình (Heatmap) trên da mô bệnh học.
- **UI Y tế Tối ưu:** Giao diện người dùng trực quan, Side-by-Side thân thiện.

---

## 2. Công nghệ Áp dụng

Dự án là sự kết hợp giữa các framework xử lý Deep Learning tiên tiến và công nghệ web API hiện đại:

- **Ngôn ngữ Lập trình:** Python (Backend/AI), JavaScript, HTML5, CSS3 (Frontend).
- **Trí tuệ Nhân tạo & Xử lý Dữ liệu:**
  - **PyTorch**: Framework Deep Learning chính vận hành và huấn luyện mô hình.
  - **Torchvision**: Khởi tạo và kế thừa trọng số cấu trúc EfficientNet-B0.
  - **Scikit-learn**: Đánh giá và phân tách bộ dữ liệu (	rain_test_split).
  - **Pandas / Pillow**: Kiểm soát luồng Metadata (CSV) và thao tác với ảnh ma trận.
  - **Grad-CAM**: Giải thuật Explainable AI ứng dụng làm rõ vùng quyết định của nơ-ron sâu nhất.
- **Web Backend & Deployment:**
  - **FastAPI**: Kiến tạo RESTful API server tốc độ cao, gen doc tự động (Swagger).
  - **Uvicorn**: ASGI server phục vụ endpoint cho dự án.
- **Phát triển & Huấn luyện (Môi trường):**
  - **Jupyter Notebook / Google Colab**: Đào tạo mô hình và thực nghiệm (EDA).

---

## 3. Cấu trúc Dự án

```text
BTL-Xulyanh/
├── app/                      # Source code hệ thống
│   ├── backend/              # Mã nguồn API Gateway & xử lý mô hình
│   │   ├── main.py           # FastAPI config & endpoints
│   │   ├── model.py          # Kiến trúc và helper load PyTorch model
│   │   └── requirements.txt  # Dependencies cho Server
│   ├── frontend/             # Giao diện chẩn đoán web
│   │   ├── index.html        # HTML Layout
│   │   ├── app.js            # Xử lý logic gọi API, render Grad-CAM
│   │   └── styles.css        # CSS Grid
│   ├── train.py              # Script huấn luyện mô hình chuẩn
│   ├── Untitled0.ipynb       # Notebook dùng để phân tích EDA/Thử nghiệm
│   └── requirements-train.txt# Dependencies (PyTorch, Pandas, v.v) cho quá trình Train
├── Data/                     # Dữ liệu HAM10000 (Ảnh + metadata)
├── DataTest/                 # Dữ liệu ảnh để test độc lập
└── model/                    # Chứa trọng số và label sau khi train
    ├── best_model.pth        # Model Weights
    └── labels.json           # File ánh xạ Index - Label
```

---

## 4. Kiến trúc Hệ thống

- **Frontend:** Thiết kế dạng Single Page Application thuần túy với CSS3 Grid Layout, Vanilla JavaScript. 
- **Backend API:** FastAPI được sử dụng làm API Gateway (cung cấp khả năng xử lý nhanh, gen tài liệu Swagger UI/ReDoc tự động).
- **Inference Engine:** PyTorch (Sử dụng pre-trained weights từ `torchvision`).
- **Môi trường Huấn luyện:** Google Colab.

---

## 5. Luồng Hoạt Động

1. **Người Dùng / Bác Sĩ (Client):** 
   - Tải hình ảnh vùng da tổn thương lên giao diện Web Frontend.
   - (Tùy chọn) Chọn các cấu hình mong muốn để chẩn đoán.
   
2. **Frontend Xử Lý:**
   - Gửi file ảnh thông qua giao thức HTTP (POST) đến Endpoint /predict của Backend FastAPI.

3. **Backend Xử Lý (Server):**
   - Nhận ảnh, kiểm thử định dạng và tiền xử lý (Resize, Normalize).
   - Truyền dữ liệu vào Core Model Inference (PyTorch + EfficientNet B0).
   
4. **Mô Hình Phân Tích (Inference + Explainability):**
   - Đưa ra danh sách top các xác suất bệnh (Classification).
   - Kích hoạt tính năng Explainable AI: tính toán Grad-CAM Heatmap tại lớp Convolution cuối cùng.
   
5. **Trả Kết Quả & Hiển Thị:**
   - Backend trả về dữ liệu chuẩn JSON.
   - Frontend hiển thị kết quả chẩn đoán cùng ảnh Overlay Map (bản đồ nhiệt Grad-CAM sinh từ API /gradcam) trực quan ngay trên trình duyệt.

---

## 6. Mô tả Dữ liệu Căn bản

Hệ thống được huấn luyện và kiểm định dựa trên bộ dữ liệu **HAM10000**, bao gồm các hình ảnh da soi (Dermoscopy):

| Viết tắt | Tên bệnh lý (Tiếng Anh / Tiếng Việt)                | Nhóm Phân loại |
| :------- | :------------------------------------------------ | :-------- |
| `akiec`  | Actinic keratoses / Dày sừng quang hóa            | Tiền U    |
| `bcc`    | Basal cell carcinoma / Ung thư biểu mô tế bào đáy | Ác tính   |
| `bkl`    | Benign keratosis-like / Tổn thương sừng hóa lành  | Lành tính |
| `df`     | Dermatofibroma / U xơ da                          | Lành tính |
| `mel`    | Melanoma / Khối u hắc tố                          | Ác tính   |
| `nv`     | Melanocytic nevi / Nốt ruồi hắc tố               | Lành tính |
| `vasc`   | Vascular lesions / Tổn thương mạch máu            | Lành tính |

---

## 7. Kiến trúc Mô hình Giải thích

- **Kiến trúc Trích xuất Đặc trưng:** `EfficientNet-B0`.
- **Kích thước Ảnh đầu vào (Input Resolution):** `224x224 RGB`.
- **Huấn luyện:** Optimizer `AdamW` cùng Loss `CrossEntropyLoss`.

**Cơ chế Grad-CAM (Saliency Map):** 
Can thiệp (hook) vào tầng chập (convolutional layer) sâu nhất để đo lường mức độ kích hoạt dựa trên feature maps. Heatmap sau đó được chồng đè (overlay) trực tiếp lên ảnh y tế gốc giúp bác sĩ hiểu lý do mô hình ra quyết định.

---

## 8. Đánh giá và Kết quả

*(Sinh viên có thể cập nhật các thông số thực tế vào mục này)*

- **Độ chính xác (Validation Accuracy):** `88.38 %`
- **ROC AUC (Macro):** `~ 0.94`
- **ROC AUC (Micro):** `~ 0.97`

**Ma trận nhầm lẫn (Confusion Matrix):** 
```markdown
![Confusion Matrix](./assets/confusion_matrix.png)
```

---

## 9. Hướng dẫn Cài đặt & Vận hành

**Yêu cầu hệ thống:** Python 3.8 trở lên.

### a. Khai báo Trọng số Mô hình
Lưu lại tệp mô hình đã train vào cấu trúc sau:
- `model/best_model.pth`
- `model/labels.json`

### b. Cài đặt Dependencies (Backend)
```bash
# Di chuyển vào thư mục dự án
python -m pip install -r app/backend/requirements.txt
```

### c. Khởi chạy Backend Server
```bash
python -m uvicorn app.backend.main:app --reload --port 8000
```
- API Docs: `http://127.0.0.1:8000/docs`
- Healthcheck: `http://127.0.0.1:8000/health`

### d. Khởi chạy Giao diện (Frontend)
Khởi động máy chủ tĩnh tại thư mục `app/frontend`:
```bash
python -m http.server 5500 -d app/frontend
```
- Mở trình duyệt Web tại: `http://127.0.0.1:5500`

### e. Huấn luyện lại Mô hình (Tùy chọn)
Nếu bạn muốn tự train lại mô hình trên dữ liệu HAM10000:
```bash
# Cài đặt thư viện hỗ trợ train
python -m pip install -r app/requirements-train.txt

# Khởi chạy script training
python app/train.py --epochs 5 --batch-size 32 --lr 0.001
```
Danh sách tham số cấu hình khi train (xem thêm): `--image-size`, `--num-workers`, `--val-split`, `--freeze-backbone`. Cấu hình model mới sẽ được lưu thẳng vào thư mục `model/`.

---

## 10. Khai thác API Dịch vụ
Hệ thống cung cấp Swagger giúp tích hợp dễ dàng:
1. **`POST /predict`**: Gửi hình ảnh + thông tin mô tả. Nhận về danh sách Top 3 khả năng lớn nhất.
2. **`POST /gradcam`**: Tính toán và gửi về Base64 String của bản đồ nhiệt (Heatmap).

---

## 11. Thông tin Tác giả

| Tiêu chí | Thông tin chi tiết |
| :--- | :--- |
| **Họ và tên** | [Điền tên của bạn] |
| **Trường** | [Điền tên trường Đại học] |
| **Khoa / Viện** | [Điền tên Khoa hoặc Chuyên ngành] |
| **Email** | [Điền email liên hệ] |

---
🌟 *Lưu ý: Hệ thống này phục vụ mục đích nghiên cứu học thuật. Tuyệt đối không thay thế hoàn toàn chẩn đoán y khoa do Bác sĩ thực hiện.*
