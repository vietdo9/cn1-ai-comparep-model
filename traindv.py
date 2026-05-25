import os
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Ép hệ thống dùng GPU 1 (RTX 4050)
os.environ['TF_DIRECTML_DEVICE'] = 'GPU:1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import layers, models, callbacks

# 2. Cấu hình bộ nhớ để không bị tràn RAM
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) >= 2:
    # Chọn GPU 1 (RTX 4050)
    tf.config.set_visible_devices(gpus[1], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[1], True)
    print("\n[OK] ĐÃ KÍCH HOẠT RTX 4050 - SẴN SÀNG TRAIN!")
elif gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"\n[OK] ĐANG DÙNG GPU: {gpus[0].name}")

# 3. Sử dụng Mixed Precision (Cực kỳ quan trọng cho card RTX)
# Nó giúp card đồ họa tính toán nhanh gấp đôi và tiết kiệm VRAM
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --- CONFIG ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 8  # An toàn hơn cho 30.000 ảnh và GPU/RAM tầm trung
EPOCHS = 10       # Incremental learning epochs
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 10

train_dir = "Data/train"
val_dir   = "Data/val"
test_dir  = "Data/test a"
RUN_DIR = "runs/effb0_20260521_183925"
MODEL_PATH = os.path.join(RUN_DIR, "resnet50_real_vs_ai.h5")

# =========================
# LOAD DATA 
# =========================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary' 
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

AUTOTUNE = tf.data.AUTOTUNE

test_ds = None
if os.path.exists(test_dir):
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    # Use smaller prefetch buffer to save system RAM
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors()).prefetch(buffer_size=2)

# Use repeated datasets only for model.fit(), because steps_per_epoch needs enough batches across epochs.
# Keep non-repeated validation/test datasets for final reports so evaluation can finish normally.
train_ds = train_ds.apply(tf.data.experimental.ignore_errors()).shuffle(500).repeat().prefetch(buffer_size=2)
val_eval_ds = val_ds.apply(tf.data.experimental.ignore_errors()).prefetch(buffer_size=2)
val_ds = val_eval_ds.repeat()

# =========================
# DATASET VERIFICATION
# =========================
def count_images(folder):
    if not os.path.exists(folder):
        return 0
    return sum(1 for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))

# Calculate steps so .repeat() knows when an epoch ends
train_count = sum(count_images(os.path.join(train_dir, c)) for c in ['real', 'fake'] if os.path.exists(os.path.join(train_dir, c)))
val_count = sum(count_images(os.path.join(val_dir, c)) for c in ['real', 'fake'] if os.path.exists(os.path.join(val_dir, c)))
STEPS_PER_EPOCH = train_count // BATCH_SIZE
VALIDATION_STEPS = val_count // BATCH_SIZE

print("\n=== DATASET COUNTS ===")
print(f"Train  real: {count_images(os.path.join(train_dir, 'real'))}")
print(f"Train  fake: {count_images(os.path.join(train_dir, 'fake'))}")
print(f"Val    real: {count_images(os.path.join(val_dir,   'real'))}")
print(f"Val    fake: {count_images(os.path.join(val_dir,   'fake'))}")
if test_ds:
    print(f"Test   real: {count_images(os.path.join(test_dir,  'real'))}")
    print(f"Test   fake: {count_images(os.path.join(test_dir,  'fake'))}")
print("======================\n")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2), 
    layers.RandomZoom(0.2),     
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
])

# =========================
# BO CUU HO VA TU DONG THONG MINH
# =========================
os.makedirs(RUN_DIR, exist_ok=True)

# Tu dong xuat file best_model.h5 tranh chay loi tat mat bai tap
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(RUN_DIR, "best_model_resnet50.h5"), save_best_only=True, monitor="val_accuracy", mode="max", verbose=1
)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

my_callbacks = [checkpoint_cb, early_stopping_cb, reduce_lr_cb]

# =========================
# LOI AI: TU DONG NHAN DIEN HOC MOI HAY HOC BOI (INCREMENTAL LEARNING)
# =========================
CHECKPOINT_PATH = os.path.join(RUN_DIR, "best_model_resnet50.h5")
RESUME_PATH = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else MODEL_PATH

if os.path.exists(RESUME_PATH):
    print(f"\n[+] DA TIM THAY BO NAO CU '{RESUME_PATH}'")
    print("    -> AI se bo qua viec di hoc mau giao, truc tiep 'HOC BOI' tiep!")
    
    # Load model cu cua vong truoc (or checkpoint)
    model = tf.keras.models.load_model(RESUME_PATH)
    
    # Chinh toc do hoc cham cuc thap (2e-5) de ngam kien thuc moi ma KHONG QUEN kien thuc dot truoc
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),  
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n--- BAT DAU: Huan Luyen Noi Tiep (Continual Learning) ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS, 
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        callbacks=my_callbacks
    )
    
    # Ve luon bieu do pha boi
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title("Tien trinh Hoc Boi (Incremental Phase)")
    plt.savefig("Figure_Incremental_resnet50.png")
    print("Saved training plot to Figure_Incremental_resnet50.png")

else:
    print(f"\n[-] KHÔNG TIM THAY '{MODEL_PATH}', XAY DUNG TU DAU VOI RESNET50")
    
    # Xay lai nao neu tren may chua co nao nao, su dung ResNet50
    base_model = tf.keras.applications.ResNet50(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base_model.trainable = False# Pha 1: dong bang

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    
    # Preprocessing danh rieng cho ResNet50
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x) 
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\n--- PHA 1: Khoi dong ---")
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE1, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, callbacks=my_callbacks)
    
    print("\n--- PHA 2: Chuyen gia Boc me AI (Fine-tuning) ---")
    base_model.trainable = True
    for layer in base_model.layers[:-5]: # Chỉ mở 5 lớp cuối
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    history2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE2, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, callbacks=my_callbacks)
    
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    plt.plot(acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend()
    plt.title("Tien trinh Xay dung tu dau (Phase 1 & 2)")
    plt.savefig("Figure_Phase1_2_resnet50.png")
    print("Saved training plot to Figure_Phase1_2_resnet50.png")

# Buoc cuoi cung chung: Cu chay xong se chep de tien trinh vao h5
model.save(MODEL_PATH)
print(f"\n Da luu ket qua thanh tuu moi nhat vao '{MODEL_PATH}'")


# ========================================================
import numpy as np
try:
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix
    
    def evaluate_and_report(dataset, title_name, save_path):
        print(f"\n\n=======================================================")
        print(f" DANG KET XUAT BAO CAO KHOA HOC - {title_name}")
        print("=======================================================")

        y_true = []
        y_pred = []

        print(f"[+] AI đang quét qua danh sách ảnh {title_name}...")
        
        # Duyệt qua từng batch để đảm bảo Ảnh và Nhãn luôn khớp nhau 100%
        for images, labels in dataset:
            try:
                preds = model.predict(images, verbose=0)
                y_true.extend(labels.numpy().flatten())
                y_pred.extend((preds >= 0.5).astype(int).flatten())
            except Exception as e:
                # Nếu gặp ảnh hỏng (Corrupt JPEG), bỏ qua tấm đó và chạy tiếp
                print(f" [!] Bỏ qua một số ảnh lỗi trong quá trình đánh giá.")
                continue

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 1. Tạo báo cáo F1-Score, Precision, Recall
        print(f"\n[BANG CHI SO KHOA HOC - {title_name}]")
        # Tránh lỗi nếu y_true rỗng do ảnh hỏng hết
        if len(y_true) > 0:
            print(classification_report(y_true, y_pred, target_names=["Anh Fake (0)", "Anh That (1)"]))

            # 2. Vẽ Ma Trận Nhầm Lẫn (Confusion Matrix)
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(7,5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=["Dự đoán FAKE", "Dự đoán REAL"], 
                        yticklabels=["Thực tế FAKE", "Thực tế REAL"])
            plt.title(f"Ma Tran Nham Lan ({title_name})")
            plt.ylabel('DAP AN THUC TE')
            plt.xlabel('AI DU DOAN')
            plt.savefig(save_path)
            print(f" [OK] Đã xuất biểu đồ ra file '{save_path}'")
        else:
            print(" [!] Không có dữ liệu để đánh giá.")
        plt.close()

    # Chạy đánh giá cho tập Validation
    evaluate_and_report(val_eval_ds, "VALIDATION", "Confusion_Matrix_Val_resnet50.png")

    # Chạy đánh giá cho tập Test (nếu có)
    if test_ds:
        evaluate_and_report(test_ds.apply(tf.data.experimental.ignore_errors()), "TEST", "Confusion_Matrix_Test_resnet50.png")
    
except ImportError:
    print("\n[!] Thiếu thư viện. Chạy: pip install scikit-learn seaborn")
except Exception as e:
    print("\n[!] Lỗi phát sinh:", e)
