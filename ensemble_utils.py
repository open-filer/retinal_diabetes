"""
Core engine for the Multi-Model Diabetic Retinopathy Detection System.
Handles loading models from two frameworks (Keras + PyTorch/timm), applying
each model's CORRECT preprocessing (each one is different -- this is the
exact bug history of this project, see README), and running ensemble
predictions across any subset of the 9 trained models.
"""

import os
import json
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# CRITICAL: import order matters here. TensorFlow and PyTorch/timm can
# segfault if TensorFlow is imported first and timm is imported afterward
# (a native OpenMP library conflict between the two frameworks). Confirmed
# by direct testing: importing timm (and torch) BEFORE tensorflow avoids the
# crash entirely. This import block must stay at the very top of this file,
# before any other module does `import tensorflow`.
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: F401  (imported here deliberately, before tensorflow)
import timm  # noqa: F401
import tensorflow as tf  # noqa: F401

# ---------------------------------------------------------------------------
# Model registry: every model's metadata, file name, and preprocessing rule.
# This single source of truth is what lets the UI offer "any combination of
# the 9 models" without hardcoding logic per model everywhere else.
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "EfficientNetB6": {
        "framework": "keras", "file": "EfficientNetB6_blindness_model.keras",
        "img_size": 528, "preprocessing": "raw_0_255",
        "accuracy": 0.98, "auc": 0.9972, "params_millions": 41.2,
        "description": "Largest EfficientNet variant used -- highest accuracy, slowest inference.",
    },
    "EfficientNetV2S": {
        "framework": "keras", "file": "EfficientNetV2S_blindness_model.keras",
        "img_size": 384, "preprocessing": "raw_0_255",
        "accuracy": 0.9745, "auc": 0.9953, "params_millions": 20.5,
        "description": "Modern EfficientNet successor -- strong accuracy, faster than B6.",
    },
    "EfficientNetB4": {
        "framework": "keras", "file": "EfficientNetB4_blindness_model.keras",
        "img_size": 380, "preprocessing": "raw_0_255",
        "accuracy": 0.96, "auc": 0.9925, "params_millions": 19.0,
        "description": "Mid-sized EfficientNet -- good balance of speed and accuracy.",
    },
    "DenseNet121": {
        "framework": "keras", "file": "DenseNet121_blindness_model.keras",
        "img_size": 224, "preprocessing": "torch_style",
        "accuracy": 0.9564, "auc": 0.9907, "params_millions": 7.2,
        "description": "Densely-connected CNN -- different architecture family from EfficientNet.",
    },
    "ResNet50": {
        "framework": "keras", "file": "ResNet50_blindness_model.keras",
        "img_size": 224, "preprocessing": "caffe_style",
        "accuracy": 0.9545, "auc": 0.9918, "params_millions": 23.8,
        "description": "Classic residual-connection CNN, a long-standing benchmark architecture.",
    },
    "EfficientNetB0": {
        "framework": "keras", "file": "EfficientNetB0_blindness_model.keras",
        "img_size": 224, "preprocessing": "raw_0_255",
        "accuracy": 0.9491, "auc": 0.9908, "params_millions": 5.3,
        "description": "Smallest EfficientNet variant -- fast, lightweight baseline.",
    },
    "MobileNetV3Large": {
        "framework": "keras", "file": "MobileNetV3Large_blindness_model.keras",
        "img_size": 224, "preprocessing": "raw_0_255",
        "accuracy": 0.9418, "auc": 0.9876, "params_millions": 3.1,
        "description": "Designed for mobile/edge devices -- smallest, fastest model here.",
    },
    "MobileViT_XXS": {
        "framework": "pytorch", "timm_name": "mobilevit_xxs",
        "file": "MobileViT_XXS_blindness_model.pt",
        "img_size": 256, "preprocessing": "scale_0_1_only",
        "accuracy": 0.9364, "auc": 0.9845, "params_millions": 0.95,
        "description": "Hybrid CNN+Transformer architecture -- a genuinely different design family.",
    },
    "EfficientFormerV2_S1": {
        "framework": "pytorch", "timm_name": "efficientformerv2_s1",
        "file": "EfficientFormerV2_S1_blindness_model.pt",
        "img_size": 224, "preprocessing": "imagenet_mean_std",
        "accuracy": 0.9364, "auc": 0.9891, "params_millions": 5.7,
        "description": "Another hybrid CNN+Transformer model -- resolution-locked to 224x224.",
    },
}

DEFAULT_ENSEMBLE = ["EfficientNetB6", "EfficientNetV2S", "EfficientNetB4", "DenseNet121", "MobileViT_XXS"]

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "all_results.json")


def load_all_results():
    """Load the pre-bundled real results for all 9 models (for the dashboard)."""
    with open(RESULTS_PATH) as f:
        return json.load(f)


def available_models():
    """Return list of model names whose weight file actually exists on disk."""
    available = []
    for name, meta in MODEL_REGISTRY.items():
        path = os.path.join(MODELS_DIR, meta["file"])
        if os.path.exists(path):
            available.append(name)
    return available


# ---------------------------------------------------------------------------
# Preprocessing: each model needs ITS OWN correct preprocessing.
# This dictionary mirrors the bug history of this project -- every single
# one of these was individually confirmed against each model's documentation
# or config during training, not assumed by analogy to another model.
# ---------------------------------------------------------------------------

def preprocess_image(pil_image, model_name):
    """Resize + preprocess a PIL image correctly for the given model."""
    meta = MODEL_REGISTRY[model_name]
    size = meta["img_size"]
    img = pil_image.convert("RGB").resize((size, size))
    arr = np.array(img).astype("float32")

    rule = meta["preprocessing"]

    if rule == "raw_0_255":
        # EfficientNet family + MobileNetV3: built-in Rescaling layer,
        # expects raw [0,255] -- do NOT divide by 255 here.
        processed = arr

    elif rule == "caffe_style":
        # ResNet50: RGB->BGR + ImageNet mean subtraction, no scaling.
        from tensorflow.keras.applications.resnet50 import preprocess_input
        processed = preprocess_input(arr.copy())

    elif rule == "torch_style":
        # DenseNet121: scale to [0,1] then ImageNet mean/std normalize.
        from tensorflow.keras.applications.densenet import preprocess_input
        processed = preprocess_input(arr.copy())

    elif rule == "scale_0_1_only":
        # MobileViT-XXS: confirmed via its own timm config (mean=0, std=1)
        # -- genuinely just /255, nothing further.
        processed = arr / 255.0

    elif rule == "imagenet_mean_std":
        # EfficientFormerV2-S1: confirmed via its own timm config --
        # standard ImageNet mean/std normalization.
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        processed = (arr / 255.0 - mean) / std

    else:
        raise ValueError(f"Unknown preprocessing rule: {rule}")

    return processed


# ---------------------------------------------------------------------------
# Model loading (cached by caller -- see app.py's st.cache_resource usage)
# ---------------------------------------------------------------------------

def load_model(model_name):
    meta = MODEL_REGISTRY[model_name]
    path = os.path.join(MODELS_DIR, meta["file"])

    if meta["framework"] == "keras":
        # tf already safely imported at module level (after torch/timm)
        return tf.keras.models.load_model(path)

    elif meta["framework"] == "pytorch":
        model = timm.create_model(meta["timm_name"], pretrained=False, num_classes=1)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model

    else:
        raise ValueError(f"Unknown framework for {model_name}")


def predict_single(model, model_name, pil_image):
    """Run one model on one image, return probability of "Has DR" (0-1)."""
    meta = MODEL_REGISTRY[model_name]
    processed = preprocess_image(pil_image, model_name)

    if meta["framework"] == "keras":
        batch = np.expand_dims(processed, axis=0)
        prob = float(model.predict(batch, verbose=0).ravel()[0])
        return prob

    elif meta["framework"] == "pytorch":
        # processed is HWC, model wants CHW
        tensor = torch.tensor(processed.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logit = model(tensor).squeeze(1)
            prob = torch.sigmoid(logit).item()
        return prob


def run_predictions(selected_models, pil_image, loaded_models):
    """
    Run every selected model on the image. loaded_models is a dict of
    {model_name: model_object}, expected to already be loaded (cached).
    Returns per-model results + the ensemble (average) verdict.
    """
    per_model = {}
    probs = []

    for name in selected_models:
        model = loaded_models[name]
        prob = predict_single(model, name, pil_image)
        probs.append(prob)
        per_model[name] = {
            "probability": prob,
            "prediction": "Has DR" if prob > 0.5 else "No DR",
            "accuracy": MODEL_REGISTRY[name]["accuracy"],
        }

    ensemble_prob = float(np.mean(probs)) if probs else 0.0
    ensemble_verdict = "Has DR" if ensemble_prob > 0.5 else "No DR"

    return {
        "per_model": per_model,
        "ensemble_probability": ensemble_prob,
        "ensemble_verdict": ensemble_verdict,
    }
