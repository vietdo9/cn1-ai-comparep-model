import os
import time
import yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Import local modules
from src.dataset import get_transforms
from src.model import build_model
from src.utils import load_checkpoint

app = Flask(__name__)

# Configure directories
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
SAMPLES_FOLDER = BASE_DIR / "static" / "samples"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLES_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

# Paths to models
RUN_DIR = BASE_DIR / "runs" / "effb0_20260521_183925"
XCEPTION_RUN_DIR = BASE_DIR / "runs" / "xception_20260522_161854"
MOBILENET_RUN_DIR = BASE_DIR / "runs" / "mobilenetv2_20260521_201900"

EFFICIENTNET_PATH = RUN_DIR / "best.pt"
RESNET_PATH = BASE_DIR / "runs" / "resnet" / "resnet50_pytorch.pth"
XCEPTION_PATH = XCEPTION_RUN_DIR / "best.pt"
MOBILENET_PATH = MOBILENET_RUN_DIR / "mobilenetv2_ai_detector.pth"

CONFIG_PATH = RUN_DIR / "config.yaml"
XCEPTION_CONFIG_PATH = XCEPTION_RUN_DIR / "config.yaml"
MOBILENET_CONFIG_PATH = MOBILENET_RUN_DIR / "config.yaml"
FALLBACK_CONFIG_PATH = BASE_DIR / "configs" / "default.yaml"

IMG_SIZE = 224
THRESHOLD = 0.5

import torchvision.models as tv_models

class PyTorchResNet50Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = tv_models.resnet50(weights=None)
        self.base.fc = nn.Identity()
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.base(x)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Global model state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {
    "efficientnet": None,
    "resnet50": None,
    "xception": None,
    "mobilenetv2": None
}
transforms_dict = {
    "efficientnet": None,
    "xception": None,
    "mobilenetv2": None
}
models_meta = {}

def get_file_size_mb(path):
    if not path or not Path(path).exists():
        return 0.0
    return Path(path).stat().st_size / (1024 * 1024)

def count_pytorch_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_models():
    """Load all three models at startup."""
    global device, models, transforms_dict, models_meta
    print(f"[*] Initializing models. PyTorch using device: {device}")

    # --- 1. Load EfficientNet-B0 (PyTorch) ---
    try:
        cfg_path = CONFIG_PATH if CONFIG_PATH.exists() else FALLBACK_CONFIG_PATH
        with open(cfg_path, "r", encoding="utf-8") as f:
            eff_config = yaml.safe_load(f)
        
        print(f"[*] Loading EfficientNet from {EFFICIENTNET_PATH}...")
        eff_model = build_model(
            model_name=eff_config["model"]["name"],
            num_classes=eff_config["model"].get("num_classes", 2),
            pretrained=False,
            dropout=eff_config["model"].get("dropout", 0.0),
        )
        eff_model, _, _, _, _ = load_checkpoint(
            str(EFFICIENTNET_PATH),
            eff_model,
            device=device,
        )
        eff_model = eff_model.to(device)
        eff_model.eval()
        
        models["efficientnet"] = eff_model
        transforms_dict["efficientnet"] = get_transforms(
            img_size=eff_config.get("data", {}).get("img_size", IMG_SIZE),
            is_train=False,
        )
        models_meta["efficientnet"] = {
            "name": "EfficientNet-B0 (PyTorch)",
            "path": str(EFFICIENTNET_PATH.relative_to(BASE_DIR)),
            "size_mb": get_file_size_mb(EFFICIENTNET_PATH),
            "params": count_pytorch_params(eff_model),
            "status": "Loaded successfully",
            "device": str(device)
        }
        print("[+] EfficientNet loaded successfully.")
    except Exception as e:
        print(f"[!] Error loading EfficientNet: {e}")
        models_meta["efficientnet"] = {"status": f"Error: {str(e)}", "name": "EfficientNet-B0 (PyTorch)"}

    # --- 2. Load ResNet50 (PyTorch) ---
    try:
        print(f"[*] Loading ResNet50 from {RESNET_PATH}...")
        resnet_model = PyTorchResNet50Binary()
        if RESNET_PATH.exists():
            state_dict = torch.load(str(RESNET_PATH), map_location=device)
            resnet_model.load_state_dict(state_dict)
            resnet_model = resnet_model.to(device)
            resnet_model.eval()
            models["resnet50"] = resnet_model
            status = "Loaded successfully"
        else:
            status = "Error: Checkpoint file not found"
            print(f"[!] ResNet50 checkpoint file not found at: {RESNET_PATH}")
            
        models_meta["resnet50"] = {
            "name": "ResNet50 (PyTorch)",
            "path": str(RESNET_PATH.relative_to(BASE_DIR)) if RESNET_PATH.exists() else str(RESNET_PATH),
            "size_mb": get_file_size_mb(RESNET_PATH) if RESNET_PATH.exists() else 0.0,
            "params": count_pytorch_params(resnet_model) if models["resnet50"] else 23725057,
            "status": status,
            "device": str(device) if models["resnet50"] else "N/A"
        }
        if models["resnet50"]:
            print("[+] ResNet50 loaded successfully.")
    except Exception as e:
        print(f"[!] Error loading ResNet50: {e}")
        models_meta["resnet50"] = {"status": f"Error: {str(e)}", "name": "ResNet50 (PyTorch)"}

    # --- 3. Load Xception (PyTorch) ---
    try:
        with open(XCEPTION_CONFIG_PATH, "r", encoding="utf-8") as f:
            xc_config = yaml.safe_load(f)
            
        print(f"[*] Loading Xception from {XCEPTION_PATH}...")
        xc_model = build_model(
            model_name=xc_config["model"]["name"],
            num_classes=xc_config["model"].get("num_classes", 2),
            pretrained=False,
            dropout=xc_config["model"].get("dropout", 0.0),
        )
        xc_model, _, _, _, _ = load_checkpoint(
            str(XCEPTION_PATH),
            xc_model,
            device=device,
        )
        xc_model = xc_model.to(device)
        xc_model.eval()
        
        models["xception"] = xc_model
        transforms_dict["xception"] = get_transforms(
            img_size=xc_config.get("data", {}).get("img_size", 299),
            is_train=False,
        )
        models_meta["xception"] = {
            "name": "Xception (PyTorch)",
            "path": str(XCEPTION_PATH.relative_to(BASE_DIR)),
            "size_mb": get_file_size_mb(XCEPTION_PATH),
            "params": count_pytorch_params(xc_model),
            "status": "Loaded successfully",
            "device": str(device)
        }
        print("[+] Xception loaded successfully.")
    except Exception as e:
        print(f"[!] Error loading Xception: {e}")
        models_meta["xception"] = {"status": f"Error: {str(e)}", "name": "Xception (PyTorch)"}

    # --- 4. Load MobileNetV2 (PyTorch) ---
    try:
        cfg_path = MOBILENET_CONFIG_PATH if MOBILENET_CONFIG_PATH.exists() else FALLBACK_CONFIG_PATH
        with open(cfg_path, "r", encoding="utf-8") as f:
            mb_config = yaml.safe_load(f)
            
        print(f"[*] Loading MobileNetV2 from {MOBILENET_PATH}...")
        import torchvision.models as tv_models
        mb_model = tv_models.mobilenet_v2(weights=None)
        mb_model.classifier[1] = nn.Linear(mb_model.last_channel, 2)
        
        if MOBILENET_PATH.exists():
            checkpoint = torch.load(str(MOBILENET_PATH), map_location=device)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    mb_model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    mb_model.load_state_dict(checkpoint['state_dict'])
                else:
                    mb_model.load_state_dict(checkpoint)
            else:
                mb_model = checkpoint
            mb_model = mb_model.to(device)
            mb_model.eval()
            models["mobilenetv2"] = mb_model
            status = "Loaded successfully"
        else:
            status = "Error: Checkpoint file not found"
            print(f"[!] MobileNetV2 checkpoint file not found at: {MOBILENET_PATH}")
            
        transforms_dict["mobilenetv2"] = get_transforms(
            img_size=mb_config.get("image_size", IMG_SIZE),
            is_train=False,
        )
        models_meta["mobilenetv2"] = {
            "name": "MobileNetV2 (PyTorch)",
            "path": str(MOBILENET_PATH.relative_to(BASE_DIR)) if MOBILENET_PATH.exists() else str(MOBILENET_PATH),
            "size_mb": get_file_size_mb(MOBILENET_PATH) if MOBILENET_PATH.exists() else 0.0,
            "params": count_pytorch_params(mb_model) if models["mobilenetv2"] else 3504872,
            "status": status,
            "device": str(device) if models["mobilenetv2"] else "N/A"
        }
        if models["mobilenetv2"]:
            print("[+] MobileNetV2 loaded successfully.")
    except Exception as e:
        print(f"[!] Error loading MobileNetV2: {e}")
        models_meta["mobilenetv2"] = {"status": f"Error: {str(e)}", "name": "MobileNetV2 (PyTorch)"}


# Run model loader once
load_models()

def predict_effnet(image_path):
    """Predict using EfficientNet PyTorch model."""
    if not models["efficientnet"]:
        raise ValueError("EfficientNet model is not loaded.")
    
    img = Image.open(image_path).convert("RGB")
    x = transforms_dict["efficientnet"](img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = models["efficientnet"](x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
    return {
        "prob_fake": float(probs[0]),
        "prob_real": float(probs[1]),
    }

def predict_resnet(image_path):
    """Predict using ResNet50 PyTorch model with Keras preprocessing."""
    if not models["resnet50"]:
        raise ValueError("ResNet50 model is not loaded.")
        
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    x = np.array(img, dtype=np.float32)
    x = x[:, :, ::-1].copy() # RGB -> BGR with positive strides
    x[:, :, 0] -= 103.939 # B
    x[:, :, 1] -= 116.779 # G
    x[:, :, 2] -= 123.68  # R
    x = x.transpose(2, 0, 1) # HWC -> CHW
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prob_real = float(models["resnet50"](x).cpu().numpy()[0][0])
        prob_fake = 1.0 - prob_real
        
    return {
        "prob_fake": prob_fake,
        "prob_real": prob_real
    }

def predict_xception(image_path):
    """Predict using Xception PyTorch model."""
    if not models["xception"]:
        raise ValueError("Xception model is not loaded.")
        
    img = Image.open(image_path).convert("RGB")
    x = transforms_dict["xception"](img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = models["xception"](x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
    return {
        "prob_fake": float(probs[0]),
        "prob_real": float(probs[1])
    }

def predict_mobilenet(image_path):
    """Predict using MobileNetV2 PyTorch model."""
    if not models["mobilenetv2"]:
        raise ValueError("MobileNetV2 model is not loaded.")
        
    img = Image.open(image_path).convert("RGB")
    x = transforms_dict["mobilenetv2"](img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = models["mobilenetv2"](x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
    return {
        "prob_fake": float(probs[0]),
        "prob_real": float(probs[1])
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/models-info")
def get_models_info():
    return jsonify({
        "status": "success",
        "device": str(device),
        "models": models_meta
    })

@app.route("/api/predict", methods=["POST"])
def run_predict():
    # Retrieve dynamic threshold if provided, else use default 0.5
    threshold = float(request.form.get("threshold", THRESHOLD))
    
    filename = None
    file_path = None
    
    # Handle sample selection OR file upload
    is_sample = request.form.get("is_sample", "false") == "true"
    if is_sample:
        sample_name = request.form.get("sample_name")
        if not sample_name:
            return jsonify({"status": "error", "message": "No sample name provided"}), 400
        file_path = SAMPLES_FOLDER / sample_name
        if not file_path.exists():
            return jsonify({"status": "error", "message": "Sample file not found"}), 404
        filename = f"sample_{sample_name}"
        web_path = f"/static/samples/{sample_name}"
    else:
        if "image" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "Empty filename"}), 400
            
        filename = secure_filename(file.filename)
        # Append timestamp to avoid caching issues on frontend
        unique_name = f"{int(time.time())}_{filename}"
        file_path = UPLOAD_FOLDER / unique_name
        file.save(file_path)
        web_path = f"/static/uploads/{unique_name}"
        
    # Get image dimensional metadata
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            img_format = img.format
    except Exception:
        width, height = 0, 0
        img_format = "Unknown"

    results = {}
    
    # Helper to execute prediction safely and record latency
    def safe_predict(predict_func, model_key, label_display):
        if not models[model_key]:
            return {
                "name": label_display,
                "status": "Error: Model not loaded",
                "prediction": "N/A",
                "confidence": 0.0,
                "prob_fake": 0.0,
                "prob_real": 0.0,
                "latency_ms": 0.0
            }
        try:
            start = time.time()
            res = predict_func(file_path)
            latency = (time.time() - start) * 1000
            
            prob_real = res["prob_real"]
            prob_fake = res["prob_fake"]
            
            # Prediction category based on threshold
            prediction = "real" if prob_real >= threshold else "fake"
            confidence = prob_real if prediction == "real" else prob_fake
            
            return {
                "name": label_display,
                "status": "Success",
                "prediction": prediction,
                "confidence": confidence,
                "prob_fake": prob_fake,
                "prob_real": prob_real,
                "latency_ms": latency
            }
        except Exception as e:
            return {
                "name": label_display,
                "status": f"Error during inference: {str(e)}",
                "prediction": "N/A",
                "confidence": 0.0,
                "prob_fake": 0.0,
                "prob_real": 0.0,
                "latency_ms": 0.0
            }

    # Run predictions for each model
    results["efficientnet"] = safe_predict(predict_effnet, "efficientnet", "EfficientNet-B0")
    results["resnet50"] = safe_predict(predict_resnet, "resnet50", "ResNet50")
    results["xception"] = safe_predict(predict_xception, "xception", "Xception")
    results["mobilenetv2"] = safe_predict(predict_mobilenet, "mobilenetv2", "MobileNetV2")
    
    # Calculate consensus/aggregations
    valid_predictions = [r["prediction"] for r in results.values() if r["status"] == "Success"]
    
    consensus_summary = "No predictions made"
    consensus_class = "N/A"
    agreement_count = 0
    total_valid = len(valid_predictions)
    
    if total_valid > 0:
        fakes = sum(1 for p in valid_predictions if p == "fake")
        reals = sum(1 for p in valid_predictions if p == "real")
        
        if fakes > reals:
            consensus_class = "fake"
            agreement_count = fakes
        elif reals > fakes:
            consensus_class = "real"
            agreement_count = reals
        else:
            # 1 vs 1 or tie? Usually 3 models, so no ties unless 2 load.
            # In case of tie, use the one with higher average confidence
            consensus_class = "mixed"
            agreement_count = 0
            
        if consensus_class == "mixed":
            consensus_summary = f"Mixed (Tie: {fakes} FAKE, {reals} REAL)"
        else:
            percentage = (agreement_count / total_valid) * 100
            consensus_summary = f"{consensus_class.upper()} ({agreement_count}/{total_valid} models agree - {percentage:.0f}%)"
            
    # Find most confident model
    success_results = [r for r in results.values() if r["status"] == "Success"]
    most_confident = None
    if success_results:
        most_confident = max(success_results, key=lambda x: x["confidence"])

    return jsonify({
        "status": "success",
        "image_url": web_path,
        "filename": filename,
        "metadata": {
            "width": width,
            "height": height,
            "format": img_format,
            "size_kb": round(os.path.getsize(file_path) / 1024, 1)
        },
        "results": results,
        "consensus": {
            "class": consensus_class,
            "summary": consensus_summary,
            "agreement_count": agreement_count,
            "total_models": total_valid,
            "most_confident": most_confident["name"] if most_confident else "None",
            "most_confident_score": most_confident["confidence"] if most_confident else 0.0
        }
    })

if __name__ == "__main__":
    # Start web development server (debug=False for production)
    app.run(debug=False, host="0.0.0.0", port=5000)