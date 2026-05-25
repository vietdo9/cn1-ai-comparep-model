# EfficientNet AI Image Detector

Pipeline PyTorch để phân loại ảnh **fake** (AI-generated) và **real** sử dụng EfficientNet-B0.

## Kết quả (WSL2 + RTX 3050 4GB)

| Metric | Giá trị |
|--------|---------|
| Accuracy | **94.37%** |
| Precision | **94.75%** |
| Recall | **93.93%** |
| F1 | **94.34%** |
| AUROC | **98.08%** |

- Dataset: 27.000 ảnh (fake/real)
- Model: EfficientNet-B0 (timm)
- Training: ~2 giờ / 10 epochs
- Batch: 32 | Mixed Precision (AMP) | Multi-worker DataLoader

## Cấu trúc project

```
.
├── configs/
│   └── default.yaml          # Hyperparameters
├── src/
│   ├── dataset.py            # Dataset + transforms
│   ├── model.py              # EfficientNet builder
│   ├── engine.py             # Train/eval loops
│   ├── utils.py              # Metrics, checkpoint, seed
│   └── viz.py                # Visualization helpers
├── train.py                  # Training script
├── eval.py                   # Evaluation + confusion matrix
├── predict.py                # Inference 1 ảnh / 1 thư mục
├── visualize.py              # Generate TensorBoard views
├── requirements.txt
└── README.md
```

## Dataset

```
train/
  fake/
  real/
val/
  fake/
  real/
test/
  fake/
  real/
```

Đảm bảo thư mục dataset đặt ở cùng cấp với `train.py` (hoặc chỉnh `configs/default.yaml`).

## Setup Windows

### 1. Tạo virtualenv

Khuyến nghị dùng Python 3.10 trên Windows để chạy ổn cả PyTorch và TensorFlow/DirectML.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 2. Cài thư viện

```powershell
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir "numpy<2" tensorflow-cpu==2.10.0 tensorflow-directml-plugin
```

Nếu chỉ cần chạy EfficientNet/PyTorch:

```powershell
pip install --no-cache-dir torch torchvision timm pyyaml pillow
```

### 3. Kiểm tra GPU

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Kiểm tra TensorFlow/DirectML:

```powershell
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

## Huấn luyện

### Train EfficientNet bằng PyTorch

```powershell
python train.py --config configs/default.yaml
```

Tùy chọn override:
```powershell
python train.py --config configs/default.yaml --epochs 15 --batch-size 32 --lr 1e-4
```

**Smoke test** (1 epoch, 5 batches):
```powershell
python train.py --config configs/default.yaml --epochs 1 --limit-batches 5
```

**Resume** từ checkpoint:
```powershell
python train.py --config configs/default.yaml --resume runs/effb0_xxx/last.pt
```

### Train ResNet50 bằng TensorFlow trên Windows

File train ResNet là:

```text
traindv.py
```

Cấu hình hiện tại trong `traindv.py`:

```text
IMG_SIZE = 224x224
BATCH_SIZE = 8
EPOCHS = 10
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 10
train_dir = Data/train
val_dir = Data/val
test_dir = Data/test a
RUN_DIR = runs/effb0_20260521_183925
```

Script đang ép TensorFlow DirectML dùng GPU 1:

```text
TF_DIRECTML_DEVICE = GPU:1
```

Nếu máy chỉ có 1 GPU hoặc muốn đổi GPU, chỉnh dòng này trong `traindv.py`:

```python
os.environ['TF_DIRECTML_DEVICE'] = 'GPU:1'
```

Chạy train:

```powershell
python traindv.py
```

Khi train xong, model được lưu tại:

```text
runs/effb0_20260521_183925/resnet50_real_vs_ai.h5
```

Checkpoint tốt nhất được lưu tại:

```text
runs/effb0_20260521_183925/best_model_resnet50.h5
```

Nếu đã có model cũ, `traindv.py` sẽ tự load theo thứ tự:

```text
runs/effb0_20260521_183925/best_model_resnet50.h5
runs/effb0_20260521_183925/resnet50_real_vs_ai.h5
```

rồi train tiếp kiểu incremental learning.

## Theo dõi training

```powershell
tensorboard --logdir runs
```

Mở browser tại `http://localhost:6006`.

## Đánh giá trên test set

```powershell
python eval.py --checkpoint runs/effb0_xxx/best.pt
```

Kết quả metrics và confusion matrix sẽ log vào cùng thư mục TensorBoard.

## Generate tất cả views (visualization nâng cao)

```powershell
python visualize.py --checkpoint runs/effb0_xxx/best.pt
```

Các view bao gồm:
- ROC curve, PR curve, Calibration plot
- Confusion matrix (raw + normalized)
- Threshold sweep (F1/Precision/Recall vs threshold)
- Probability distribution histogram
- Top worst predictions (FP/FN grid)
- Embedding projector (t-SNE/UMAP trong TensorBoard)

## Dự đoán ảnh mới

1 ảnh:
```powershell
python predict.py --image path/to/image.jpg --checkpoint runs/effb0_xxx/best.pt
```

Cả thư mục:
```powershell
python predict.py --folder path/to/folder --checkpoint runs/effb0_xxx/best.pt
```

## App so sánh EfficientNet và ResNet

App giao diện:

```text
compare_app.py
```

Chạy:

```powershell
python compare_app.py
```

Hoặc chạy trực tiếp bằng môi trường ảo:

```powershell
.\venv\Scripts\python.exe compare_app.py
```

App mặc định dùng:

```text
runs/effb0_20260521_183925/best.pt
runs/effb0_20260521_183925/resnet50_real_vs_ai.h5
```

Sau khi mở app, bấm `Chọn ảnh` để test và so sánh xác suất `fake` / `real` của 2 model.

## Cấu hình

Chỉnh sửa `configs/default.yaml` để thay đổi:
- `model.name`: đổi sang EfficientNet-B3, B4, v.v.
- `data.batch_size`: giảm xuống 16 nếu OOM
- `train.epochs`: số epoch
- `augmentation`: bật/tắt augmentation

## Lưu ý

- **Windows:** nên dùng `venv` trong project để tránh xung đột package.
- **TensorFlow:** nếu lỗi `pywrap_tensorflow`, thường là TensorFlow cài dở hoặc thiếu dung lượng ổ C; hãy dọn cache bằng `pip cache purge` rồi cài lại trong `venv`.
- **traindv.py:** đang dùng `BATCH_SIZE = 8` để an toàn RAM/VRAM.
- **OOM:** giảm batch size xuống 4 hoặc 2 nếu máy thiếu VRAM/RAM.
- **PyTorch trên Windows:** nếu dùng bản CPU thì EfficientNet vẫn chạy được nhưng sẽ chậm hơn GPU.
