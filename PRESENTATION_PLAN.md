# Kế hoạch thuyết trình cuối kỳ V2 (đầy đủ 4 model + kiến trúc)

Plan 16 slide cho thuyết trình 18-22 phút, 4 thành viên, trọng tâm EfficientNet với compound scaling, mỗi model có slide kiến trúc riêng + sơ đồ block, demo `compare_app.py` mở rộng 4 model.

---

## Thay đổi so với V1

- ✅ Thêm slide **kiến trúc chi tiết** cho từng model (sơ đồ block + đặc trưng)
- ✅ Xception được đưa vào demo 4-model
- ✅ Slide tổng số: 15 → **16** (thêm 1 slide visualization architecture)
- ✅ Cập nhật `compare_app.py` cần thêm Xception (TODO implementation sau khi duyệt plan)

---

## Bảng so sánh tổng hợp (slide 13)

| Model | Block đặc trưng | Params | Img Size | Test Acc | AUROC | F1 | Optimizer | LR |
|---|---|---|---|---|---|---|---|---|
| **EfficientNet-B0** | MBConv (Inverted Residual + SE) | **5.3M** | 224 | **94.37%** | **0.9808** | **0.9434** | AdamW | 3e-4 |
| ResNet50 | Residual Bottleneck | 25.6M | 224 | 87.00% | 0.9000 | 0.87 | Adam | 1e-3→1e-5 |
| MobileNetV2 | Inverted Residual + Linear Bottleneck | **3.5M** | 224 | 91.17% | — | — | Adam | 1e-4 |
| Xception | Depthwise Separable Conv | 22.9M | 299 | 90.83% | 0.9542 | 0.9047 | AdamW | 3e-4 |

---

## Cấu trúc 16 Slide

### Slide 1 — Trang bìa (30s) — *Trưởng nhóm*
- Tiêu đề, thành viên, lớp, GVHD, ngày

---

### Slide 2 — Bài toán & Động lực (1 phút) — *TV1*
- Bối cảnh: bùng nổ ảnh AI (Midjourney, SD, DALL-E) → fake news, deepfake
- Bài toán: binary classification real vs fake
- Mục tiêu: so sánh 4 CNN tìm model tối ưu (Acc ↔ Params ↔ Speed)

---

### Slide 3 — Dataset & Pipeline (1 phút) — *TV1*
- 27.000 ảnh, 2 class (real/fake), split train/val/test
- Preprocessing: Resize 224 (Xception 299), Normalize ImageNet
- Augmentation: `RandomResizedCrop`, `HFlip`, `ColorJitter` (`src/dataset.py`)
- Robust loading: skip corrupted images (`RobustImageFolder`)

---

### Slide 4 — Phương pháp luận chung (1 phút) — *TV1*
- **Transfer Learning** từ ImageNet pretrained
- **2-phase fine-tune:** freeze 1 epoch → unfreeze toàn bộ
- **Mixed Precision (AMP):** `torch.amp.autocast` + `GradScaler` → tăng tốc 2×
- **Early stopping** theo `val_auroc` (patience=3)

---

### Slide 5 — 🌟 EfficientNet (1): Compound Scaling — lý thuyết (1.5 phút) — *TV2*

**Sơ đồ:** 3 trục scale + biểu đồ B0→B7

- **Vấn đề:** Scale model truyền thống chỉ tăng 1 trong 3 chiều
  - Depth → ResNet | Width → WideResNet | Resolution → input lớn
- **Compound Scaling (Tan & Le, 2019):**
  - `depth = α^φ`, `width = β^φ`, `resolution = γ^φ`
  - Ràng buộc: `α · β² · γ² ≈ 2`, `α≥1, β≥1, γ≥1`
  - `φ` (compound coefficient) → B0, B1, ..., B7
- **Insight:** Cân bằng 3 chiều cho AUC tốt hơn so với scale 1 chiều với cùng FLOPs

---

### Slide 6 — 🌟 EfficientNet (2): Kiến trúc & MBConv block (1.5 phút) — *TV2*

**Sơ đồ block:** MBConv (Mobile Inverted Bottleneck Conv) + Squeeze-Excitation

- **Cấu trúc B0:** Stem → 7 stage MBConv (1-1-2-2-3-3-1 layer) → Head
- **MBConv block:**
  1. **Expand:** 1×1 conv (tăng channel × `t`)
  2. **Depthwise:** 3×3 hoặc 5×5 DWConv
  3. **SE block:** Squeeze-Excitation (channel attention)
  4. **Project:** 1×1 conv (giảm channel) + Linear (không activation)
  5. **Skip connection** nếu cùng shape
- **Implementation:** `timm.create_model('efficientnet_b0', pretrained=True, drop_rate=0.2)`
- **Hyperparams (`configs/default.yaml`):**
  ```yaml
  AdamW(lr_head=3e-4, lr_backbone=1e-4, wd=1e-4)
  Cosine scheduler (min_lr=1e-6)
  label_smoothing=0.05, grad_clip=1.0
  ```
- **Kết quả:** **Acc 94.37% | AUROC 0.9808 | F1 0.9434**

---

### Slide 7 — ResNet50: Residual Learning + Bottleneck (1.5 phút) — *TV3*

**Sơ đồ block:** Residual Bottleneck (1×1 → 3×3 → 1×1) + skip connection

- **Ý tưởng:** `y = F(x) + x` → giải vanishing gradient ở mạng sâu (He et al., 2016)
- **Cấu trúc:** 50 layer, ~25.6M params
  - Stem 7×7 + maxpool → 4 stage [3, 4, 6, 3] residual block → Avgpool + FC
- **Bottleneck:** 1×1 (giảm dim) → 3×3 (conv chính) → 1×1 (tăng dim) → Add(input)
- **Implementation (`traindv.py`):**
  - Framework: TensorFlow/Keras (`tf.keras.applications.ResNet50`)
  - Mixed Precision `mixed_float16`
  - Phase 1: freeze base, Adam lr=1e-3, 5 epochs
  - Phase 2: unfreeze 5 layer cuối, lr=1e-5, 10 epochs
  - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- **Kết quả:** _TODO: nhóm bổ sung sau eval_

---

### Slide 8 — MobileNetV2: Depthwise Separable + Inverted Residual (1.5 phút) — *TV4*

**Sơ đồ block:** Inverted Residual + Linear Bottleneck

- **Ý tưởng cốt lõi:**
  - **Depthwise Separable Conv** = Depthwise (3×3 per-channel) + Pointwise (1×1) → giảm 8-9× FLOPs vs conv thường
  - **Inverted Residual:** expand → DWConv → project (đảo so với ResNet bottleneck)
  - **Linear Bottleneck:** bỏ activation ở project layer (tránh mất info ở low-dim)
- **Cấu trúc:** Stem → 17 inverted residual block → 1×1 conv → Pool → FC, ~3.5M params
- **Implementation (`mobilenetv2_ai_detector/traincd.py`):**
  - Framework: PyTorch torchvision (`models.mobilenet_v2(weights=DEFAULT)`)
  - Adam lr=1e-4, batch=8, 3 epochs
  - Augmentation: HFlip + RandomRotation(10) (đơn giản hơn EfficientNet)
- **Kết quả:** Test acc **91.17%** | Train 91.57% | Val 92.20%

---

### Slide 9 — Xception: Extreme Inception (1.5 phút) — *TV2 hoặc TV3*

**Sơ đồ block:** Entry/Middle/Exit Flow + Separable Conv

- **Ý tưởng:** "Inception → cực đoan" — thay block Inception bằng Depthwise Separable Conv (Chollet, 2017)
- **Giả thuyết:** Cross-channel correlation và spatial correlation **có thể tách rời hoàn toàn**
- **Cấu trúc 3 flow:**
  - **Entry Flow:** 4 module Separable Conv + downsample
  - **Middle Flow:** 8 module separable (lặp) — phần chính
  - **Exit Flow:** 2 module + GlobalAvgPool + FC
- **Đặc trưng:**
  - Input 299×299 (không phải 224)
  - ~22.9M params, sâu hơn ResNet50 về tổng layer
  - Skip connection giữa các block (residual-style)
- **Implementation (`trainmt.py`):**
  - Framework: PyTorch (timm `xception`)
  - Dùng chung `src/engine.py` với EfficientNet
  - AdamW lr=3e-4 head/1e-4 backbone, Cosine, 2-phase fine-tune
- **Kết quả:** Acc **90.83%** | AUROC 0.9542 | F1 0.9047
- **Lưu ý:** 25-40 phút/epoch (chậm hơn EfficientNet ~3-4×)

---

### Slide 10 — So sánh các loại block (1 phút) — *TV2*

**Hình minh họa:** 4 block side-by-side

| Block | Model | Đặc trưng |
|---|---|---|
| Residual Bottleneck | ResNet50 | 1×1 → 3×3 → 1×1 + skip |
| Inverted Residual | MobileNetV2 | Expand → DWConv → Project (linear) |
| MBConv + SE | EfficientNet | Inverted residual + Squeeze-Excitation |
| Separable Conv | Xception | Depthwise + Pointwise (no expand) |

- **Tiến hóa:** ResNet (residual) → MobileNet (DWConv + inverted) → EfficientNet (+ SE attention + compound scaling)

---

### Slide 11 — Quy trình Training (technical) (1 phút) — *TV1*
- **Pipeline (`train.py`, `trainmt.py`):**
  1. Load config YAML → resolve paths
  2. Build dataset + augmentation
  3. Build model (timm/torchvision/Keras) + transfer learning
  4. Loop: freeze → unfreeze, AMP, log TB + CSV
  5. Early stop + save best/last checkpoint
- **Logging:** TensorBoard (loss, acc, AUROC, LR, time)
- **Export views:** `export_views.py` xuất 13 PNG (ROC, PR, CM, calibration, threshold sweep, worst predictions, training curves)

---

### Slide 12 — Kết quả & Visualization (1.5 phút) — *TV2*

**Hiển thị 3-4 ảnh:**
- ROC curve overlay (EfficientNet vs Xception)
- Confusion matrix EfficientNet
- Training accuracy/loss curves
- Worst predictions (FP/FN)

**Quan sát:**
- EfficientNet hội tụ nhanh nhất (~3-4 epoch đạt val_acc>90%)
- Xception có dấu hiệu overfit nhẹ ở epoch 7-8 (train 98% / val 92%)
- MobileNetV2 chỉ 3 epoch — có thể tăng để cạnh tranh hơn
- ResNet50 ổn định nhờ 2-phase + ReduceLROnPlateau

---

### Slide 13 — 🎯 BẢNG SO SÁNH TỔNG HỢP (slide cốt lõi) (1.5 phút) — *Trưởng nhóm*

Bảng đầy đủ (xem mục "Bảng so sánh tổng hợp" ở trên) + biểu đồ Accuracy vs Params.

**Phân tích:**
- **Acc/Params ratio:** EfficientNet-B0 vượt trội — 94.4% với 5.3M (so với Xception 90.8% / 22.9M = chỉ bằng 1/4 params)
- **Speed:** EfficientNet 12 min/epoch | Xception 25-40 min/epoch | ResNet phase 1 ~5 min/epoch
- **VRAM (RTX 3050 4GB):** EfficientNet batch=32 | Xception/MobileNet/ResNet batch=8
- **Verdict:** EfficientNet-B0 là **lựa chọn tối ưu** cho bài toán này

---

### Slide 14 — 🖥️ DEMO LIVE — `compare_app.py` (4 model) (2.5-3 phút) — *TV4 thao tác + cả nhóm bình luận*

- **App Tkinter** load đồng thời 4 model:
  - EfficientNet (`best.pt`) — PyTorch
  - ResNet50 (`resnet50_real_vs_ai.h5`) — TensorFlow
  - MobileNetV2 (`.pth`) — PyTorch torchvision
  - **Xception** (`runs/xception_*/best.pt`) — PyTorch timm ← **cần update code**
- **Demo flow (3 ảnh):**
  1. Ảnh real (chụp thực tế) → 4 model cùng vote
  2. Ảnh fake rõ (Midjourney) → 4 model cùng vote
  3. Ảnh fake "khó" (AI photorealistic) → xem model nào còn dự đoán đúng
- **Slide:** screenshot UI mới (4 box prediction)

> **TODO sau khi duyệt plan:** Sửa `compare_app.py` thêm Xception (load timm xception, transform 299×299)

---

### Slide 15 — Kết luận & Khuyến nghị (1 phút) — *Trưởng nhóm*
- **EfficientNet-B0 thắng cuộc:**
  1. Acc cao nhất (94.37%) trong 4 model
  2. Params nhỏ (5.3M) — dễ deploy mobile/edge
  3. Training nhanh trên VRAM 4GB
  4. Minh chứng thực tế cho compound scaling (Tan & Le, 2019)
- **Khi nào dùng model khác:**
  - MobileNetV2: real-time mobile, IoT
  - ResNet50: baseline mạnh, hệ sinh thái lớn
  - Xception: texture-heavy dataset, có resource lớn
- **Hướng phát triển:** EfficientNet-B2/B3, ensemble 4-model, test trên GenImage benchmark

---

### Slide 16 — Q&A / Cảm ơn (1 phút) — *Cả nhóm*
- Cảm ơn GVHD, lớp
- Link repo: `github.com/Cogbao/efficientnet-ai-detector`
- Nhận câu hỏi

---

## Phân chia 4 thành viên (cân bằng)

| Thành viên | Slide phụ trách | Thời gian phát biểu | Vai trò |
|---|---|---|---|
| **TV1 (Trưởng nhóm)** | 1, 2, 3, 4, 11, 13, 15, 16 | ~5-6 phút | Mở đầu, dataset, pipeline, tổng kết |
| **TV2 (EfficientNet lead)** | 5, 6, 9, 10, 12 | ~5-6 phút | EfficientNet + Xception + so sánh block + viz |
| **TV3 (ResNet)** | 7 + hỗ trợ slide 10 | ~3 phút | ResNet50 chi tiết |
| **TV4 (MobileNetV2 + Demo)** | 8, 14 | ~4 phút | MobileNetV2 + thao tác demo 4-model |

---

## Checklist chuẩn bị

- [ ] **Bổ sung số liệu ResNet50** — chạy eval, lấy test acc/F1/AUROC
- [ ] **Update `compare_app.py`** thêm Xception (sau khi duyệt plan)
- [ ] **Chuẩn bị sơ đồ block** (4 ảnh): Residual / Inverted Residual / MBConv+SE / Separable Conv
- [ ] **Chuẩn bị sơ đồ compound scaling** (3 trục depth/width/resolution)
- [ ] **Export 4-5 visualization** từ `runs/effb0_*/views/` & `runs/xception_*/views/`
- [ ] **Chuẩn bị 3 ảnh demo:** real / fake rõ / fake khó
- [ ] **Test app trên máy thuyết trình** (TF + PyTorch + Tkinter + Xception ckpt)
- [ ] **Record backup video demo** phòng lỗi live

---

## Văn phong khuyến nghị

- **Slide kiến trúc (5-9):** Sơ đồ block to + 3-4 bullet ngắn — đừng đọc slide, giải thích flow
- **Slide so sánh block (10):** Tốc độ nhanh, làm rõ "tiến hóa" qua các năm
- **Slide kết quả (12-13):** Để **số liệu nói**, biểu đồ phải đọc được từ xa
- **Demo (14):** Tương tác app, comment kết quả thay vì đọc slide
- **Kết luận (15):** Văn phong tư vấn — "Khuyến nghị" + use-case phù hợp
