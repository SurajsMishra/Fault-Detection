from functools import lru_cache
from io import BytesIO
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
from torch import nn

from .features import extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RF_MODEL_PATH = PROJECT_ROOT / "models" / "rf_fan_id06.joblib"
CNN_MODEL_PATH = PROJECT_ROOT / "models" / "cnn_fan_id06.pth"


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.layers(x)


@lru_cache(maxsize=1)
def load_rf_bundle():
    if not RF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Random Forest model not found: {RF_MODEL_PATH}")
    return joblib.load(RF_MODEL_PATH)


@lru_cache(maxsize=1)
def load_cnn_bundle():
    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"CNN model not found: {CNN_MODEL_PATH}")

    checkpoint = torch.load(
        CNN_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model = SmallCNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, float(checkpoint["mean"]), float(checkpoint["std"])


def predict_wav(wav_bytes):
    """Predict with the existing Random Forest model."""
    audio, sample_rate = librosa.load(
        BytesIO(wav_bytes),
        sr=None,
        mono=True,
    )

    features = extract_features(audio, sample_rate).reshape(1, -1)
    bundle = load_rf_bundle()

    if features.shape[1] != len(bundle["feature_names"]):
        raise ValueError(
            f"Expected {len(bundle['feature_names'])} features, "
            f"but extracted {features.shape[1]}."
        )

    model = bundle["model"]
    prediction = int(model.predict(features)[0])
    anomaly_score = float(model.predict_proba(features)[0, 1])

    return {
        "label": "Anomaly" if prediction == 1 else "Normal",
        "anomaly_score": anomaly_score,
    }


def predict_cnn_wav(wav_bytes):
    """Predict with the CNN using the Colab training preprocessing."""
    checkpoint_sample_rate = 16_000
    audio, sample_rate = librosa.load(
        BytesIO(wav_bytes),
        sr=checkpoint_sample_rate,
        mono=True,
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
    )
    spectrogram = librosa.power_to_db(mel + 1e-10, ref=1.0)

    model, mean, std = load_cnn_bundle()
    model_input = ((spectrogram - mean) / std).astype(np.float32)
    model_input = torch.from_numpy(model_input[None, None, :, :])

    with torch.no_grad():
        probabilities = torch.softmax(model(model_input), dim=1)[0]
        prediction = int(torch.argmax(probabilities).item())

    return {
        "label": "Anomaly" if prediction == 1 else "Normal",
        "anomaly_score": float(probabilities[1].item()),
    }
def predict_rf_cnn_review(wav_bytes):
    rf_result = predict_wav(wav_bytes)
    cnn_result = predict_cnn_wav(wav_bytes)

    agreed = rf_result["label"] == cnn_result["label"]

    return {
        "label": rf_result["label"] if agreed else "Review required",
        "agreed": agreed,
        "rf": rf_result,
        "cnn": cnn_result,
    }