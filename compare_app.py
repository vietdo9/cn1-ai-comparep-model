import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
import yaml
from PIL import Image, ImageTk

from src.dataset import get_transforms
from src.model import build_model
from src.utils import load_checkpoint


RUN_DIR = Path("runs/effb0_20260521_183925")
XCEPTION_RUN_DIR = Path("runs/xception_20260522_161854")
EFFICIENTNET_PATH = RUN_DIR / "best.pt"
RESNET_PATH = RUN_DIR / "resnet50_real_vs_ai.h5"
XCEPTION_PATH = XCEPTION_RUN_DIR / "best.pt"
CONFIG_PATH = RUN_DIR / "config.yaml"
XCEPTION_CONFIG_PATH = XCEPTION_RUN_DIR / "config.yaml"
FALLBACK_CONFIG_PATH = Path("configs/default.yaml")
IMG_SIZE = 224
THRESHOLD = 0.5


class ModelComparatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Compare EfficientNet vs ResNet vs Xception")
        self.root.geometry("1080x720")
        self.root.minsize(980, 650)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tf = None
        self.efficientnet_model = None
        self.efficientnet_transform = None
        self.resnet_model = None
        self.xception_model = None
        self.xception_transform = None
        self.current_image = None
        self.preview_image = None

        self.status_var = tk.StringVar(value="Loading models...")
        self.image_path_var = tk.StringVar(value="No image selected")
        self.eff_result_var = tk.StringVar(value="No result")
        self.resnet_result_var = tk.StringVar(value="No result")
        self.xception_result_var = tk.StringVar(value="No result")
        self.compare_var = tk.StringVar(value="Select an image to start compare")

        self.build_ui()
        threading.Thread(target=self.load_models, daemon=True).start()

    def build_ui(self):
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(container, text="App test and compare 3 models", font=("Segoe UI", 20, "bold"))
        title.pack(anchor=tk.W)

        subtitle = ttk.Label(
            container,
            text=f"EfficientNet: {EFFICIENTNET_PATH}    |    ResNet: {RESNET_PATH}    |    Xception: {XCEPTION_PATH}",
            font=("Segoe UI", 10),
        )
        subtitle.pack(anchor=tk.W, pady=(4, 12))

        top = ttk.Frame(container)
        top.pack(fill=tk.X, pady=(0, 12))

        ttk.Button(top, text="Select image", command=self.select_image).pack(side=tk.LEFT)
        ttk.Button(top, text="Run again", command=self.predict_current_image).pack(side=tk.LEFT, padx=8)
        ttk.Label(top, textvariable=self.image_path_var).pack(side=tk.LEFT, padx=12)

        main = ttk.Frame(container)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(main, text="Test image", padding=12)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.image_label = ttk.Label(left, text="No image selected", anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        eff_box = ttk.LabelFrame(right, text="EfficientNet best.pt", padding=12)
        eff_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        ttk.Label(eff_box, textvariable=self.eff_result_var, font=("Consolas", 11), justify=tk.LEFT).pack(anchor=tk.W)

        resnet_box = ttk.LabelFrame(right, text="ResNet resnet50_real_vs_ai.h5", padding=12)
        resnet_box.pack(fill=tk.BOTH, expand=True, pady=(8, 8))
        ttk.Label(resnet_box, textvariable=self.resnet_result_var, font=("Consolas", 11), justify=tk.LEFT).pack(anchor=tk.W)

        xception_box = ttk.LabelFrame(right, text="Xception best.pt", padding=12)
        xception_box.pack(fill=tk.BOTH, expand=True, pady=(8, 8))
        ttk.Label(xception_box, textvariable=self.xception_result_var, font=("Consolas", 11), justify=tk.LEFT).pack(anchor=tk.W)

        compare_box = ttk.LabelFrame(right, text="Compare", padding=12)
        compare_box.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        ttk.Label(compare_box, textvariable=self.compare_var, font=("Segoe UI", 11, "bold"), justify=tk.LEFT).pack(anchor=tk.W)

        status = ttk.Label(container, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status.pack(fill=tk.X, pady=(12, 0))

    def load_config(self):
        path = CONFIG_PATH if CONFIG_PATH.exists() else FALLBACK_CONFIG_PATH
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_xception_config(self):
        with open(XCEPTION_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_models(self):
        try:
            config = self.load_config()
            xception_config = self.load_xception_config()

            checkpoint_path = str(EFFICIENTNET_PATH)
            if not EFFICIENTNET_PATH.exists():
                raise FileNotFoundError(f"EfficientNet checkpoint not found: {EFFICIENTNET_PATH}")
            if not RESNET_PATH.exists():
                raise FileNotFoundError(f"ResNet model not found: {RESNET_PATH}")
            if not XCEPTION_PATH.exists():
                raise FileNotFoundError(f"Xception checkpoint not found: {XCEPTION_PATH}")

            self.efficientnet_model = build_model(
                model_name=config["model"]["name"],
                num_classes=config["model"].get("num_classes", 2),
                pretrained=False,
                dropout=config["model"].get("dropout", 0.0),
            )
            self.efficientnet_model, _, _, _, _ = load_checkpoint(
                checkpoint_path,
                self.efficientnet_model,
                device=self.device,
            )
            self.efficientnet_model = self.efficientnet_model.to(self.device)
            self.efficientnet_model.eval()
            self.efficientnet_transform = get_transforms(
                img_size=config.get("data", {}).get("img_size", IMG_SIZE),
                is_train=False,
            )

            import tensorflow as tf
            self.tf = tf
            self.resnet_model = self.tf.keras.models.load_model(str(RESNET_PATH))

            self.xception_model = build_model(
                model_name=xception_config["model"]["name"],
                num_classes=xception_config["model"].get("num_classes", 2),
                pretrained=False,
                dropout=xception_config["model"].get("dropout", 0.0),
            )
            self.xception_model, _, _, _, _ = load_checkpoint(
                str(XCEPTION_PATH),
                self.xception_model,
                device=self.device,
            )
            self.xception_model = self.xception_model.to(self.device)
            self.xception_model.eval()
            self.xception_transform = get_transforms(
                img_size=xception_config.get("data", {}).get("img_size", 299),
                is_train=False,
            )
            self.root.after(0, lambda: self.status_var.set(f"Models loaded. PyTorch device: {self.device}"))
        except Exception as exc:
            self.root.after(0, lambda: self.show_error(str(exc)))

    def show_error(self, message):
        self.status_var.set("Model load/run error")
        messagebox.showerror("Error", message)

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select image to test",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        self.current_image = Path(path)
        self.image_path_var.set(str(self.current_image))
        self.show_preview(self.current_image)
        self.predict_current_image()

    def show_preview(self, path):
        image = Image.open(path).convert("RGB")
        image.thumbnail((430, 430))
        self.preview_image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.preview_image, text="")

    def predict_current_image(self):
        if self.current_image is None:
            messagebox.showinfo("Thông báo", "Bạn hãy chọn ảnh trước.")
            return
        if self.efficientnet_model is None or self.resnet_model is None or self.xception_model is None:
            messagebox.showinfo("Thông báo", "Model chưa tải xong, chờ một chút rồi thử lại.")
            return
        self.status_var.set("Đang dự đoán...")
        self.eff_result_var.set("Đang chạy...")
        self.resnet_result_var.set("Đang chạy...")
        self.xception_result_var.set("Đang chạy...")
        self.compare_var.set("Đang compare...")
        threading.Thread(target=self.run_predictions, daemon=True).start()

    def run_predictions(self):
        try:
            eff = self.predict_efficientnet(self.current_image)
            resnet = self.predict_resnet(self.current_image)
            xception = self.predict_xception(self.current_image)
            self.root.after(0, lambda: self.update_results(eff, resnet, xception))
        except Exception as exc:
            self.root.after(0, lambda: self.show_error(str(exc)))

    def predict_efficientnet(self, image_path):
        image = Image.open(image_path).convert("RGB")
        x = self.efficientnet_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.efficientnet_model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = 1 if probs[1] >= THRESHOLD else 0
        return {
            "name": "EfficientNet",
            "prediction": "real" if pred == 1 else "fake",
            "prob_fake": float(probs[0]),
            "prob_real": float(probs[1]),
            "confidence": float(max(probs)),
        }

    def predict_resnet(self, image_path):
        image = self.tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        x = self.tf.keras.utils.img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = self.tf.keras.applications.resnet50.preprocess_input(x)
        prob_real = float(self.resnet_model.predict(x, verbose=0)[0][0])
        prob_fake = 1.0 - prob_real
        pred = 1 if prob_real >= THRESHOLD else 0
        return {
            "name": "ResNet",
            "prediction": "real" if pred == 1 else "fake",
            "prob_fake": prob_fake,
            "prob_real": prob_real,
            "confidence": max(prob_fake, prob_real),
        }

    def predict_xception(self, image_path):
        image = Image.open(image_path).convert("RGB")
        x = self.xception_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.xception_model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = 1 if probs[1] >= THRESHOLD else 0
        return {
            "name": "Xception",
            "prediction": "real" if pred == 1 else "fake",
            "prob_fake": float(probs[0]),
            "prob_real": float(probs[1]),
            "confidence": float(max(probs)),
        }

    def update_results(self, eff, resnet, xception):
        self.eff_result_var.set(self.format_result(eff))
        self.resnet_result_var.set(self.format_result(resnet))
        self.xception_result_var.set(self.format_result(xception))

        results = [eff, resnet, xception]
        predictions = [result["prediction"] for result in results]
        if len(set(predictions)) == 1:
            compare = f"Bốn model cùng dự đoán: {predictions[0].upper()}\n"
        else:
            compare = "Các model dự đoán KHÁC nhau\n"
            for result in results:
                compare += f"- {result['name']}: {result['prediction'].upper()}\n"

        better = max(results, key=lambda result: result["confidence"])
        compare += f"Model tự tin hơn: {better['name']} ({better['confidence']:.4f})"
        self.compare_var.set(compare)
        self.status_var.set("Hoàn tất dự đoán")

    def format_result(self, result):
        return (
            f"Prediction : {result['prediction'].upper()}\n"
            f"Confidence : {result['confidence']:.4f}\n"
            f"Prob fake  : {result['prob_fake']:.4f}\n"
            f"Prob real  : {result['prob_real']:.4f}"
        )


def main():
    root = tk.Tk()
    app = ModelComparatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
