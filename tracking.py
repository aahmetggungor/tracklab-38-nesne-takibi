"""Etiketli takip deneyleri: aynı ViTTrack, farklı görüntü ön işlemleri."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL = ROOT / "assets" / "vittrack.onnx"
SEQUENCES = ("david", "dudek", "faceocc2")
CONDITIONS = ("Orijinal", "Karanlık", "Bulanık")
METHODS = ("İşlemsiz", "CLAHE", "Gamma 0.7")


@dataclass(frozen=True)
class Sequence:
    name: str
    video: Path
    ground_truth: np.ndarray
    first_frame: int


def load_sequence(name: str) -> Sequence:
    if name not in SEQUENCES:
        raise ValueError(f"Bilinmeyen video: {name}")
    video = DATA / f"{name}.webm"
    labels = DATA / f"{name}_gt.txt"
    config = DATA / f"{name}.yml"
    if not all(path.is_file() for path in (video, labels, config)):
        raise FileNotFoundError("Örnek videolar eksik; download_assets.py çalıştırın.")
    match = re.search(r"^start:\s*(\d+)", config.read_text(encoding="utf-8"), re.M)
    if not match:
        raise ValueError(f"Başlangıç karesi bulunamadı: {config}")
    rows = []
    for line in labels.read_text(encoding="utf-8").splitlines():
        if line.strip():
            values = [float(item) for item in re.split(r"[,\s]+", line.strip())]
            if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
                raise ValueError(f"Geçersiz gerçek kutu: {name}")
            rows.append(values)
    return Sequence(name, video, np.asarray(rows, dtype=np.float32), int(match.group(1)))


def processed(frame: np.ndarray, condition: str, method: str) -> np.ndarray:
    if condition not in CONDITIONS or method not in METHODS:
        raise ValueError("Bilinmeyen koşul veya ön işleme yöntemi.")
    if condition == "Orijinal":
        result = frame
    elif condition == "Karanlık":
        result = cv2.convertScaleAbs(frame, alpha=0.40, beta=0)
    else:
        result = cv2.GaussianBlur(frame, (9, 9), 2.0)
    if method == "CLAHE":
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif method == "Gamma 0.7":
        lut = np.clip(np.power(np.arange(256, dtype=np.float32) / 255.0, 0.7) * 255, 0, 255).astype(np.uint8)
        result = cv2.LUT(result, lut)
    return result


def tracker() -> object:
    if not MODEL.is_file():
        raise FileNotFoundError("ViTTrack modeli eksik; download_assets.py çalıştırın.")
    params = cv2.TrackerVit_Params()
    params.net = str(MODEL)
    params.backend = cv2.dnn.DNN_BACKEND_OPENCV
    params.target = cv2.dnn.DNN_TARGET_CPU
    return cv2.TrackerVit_create(params)


def iou(box: tuple[float, ...] | None, truth: np.ndarray) -> float:
    if box is None:
        return 0.0
    ax, ay, aw, ah = box
    bx, by, bw, bh = truth
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0 else 0.0


def _seek_to_start(capture: cv2.VideoCapture, first_frame: int) -> np.ndarray:
    for _ in range(first_frame):
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Video etiket başlangıcından önce bitti.")
    return frame


def run_one(name: str, condition: str, method: str) -> pd.DataFrame:
    sequence = load_sequence(name)
    capture = cv2.VideoCapture(str(sequence.video))
    if not capture.isOpened():
        raise ValueError(f"Video açılamadı: {name}")
    try:
        first = _seek_to_start(capture, sequence.first_frame)
        started = time.perf_counter()
        first = processed(first, condition, method)
        model = tracker()
        initial = tuple(int(round(value)) for value in sequence.ground_truth[0])
        model.init(first, initial)
        elapsed = time.perf_counter() - started
        rows = [{"video": name, "koşul": condition, "yöntem": method, "kare": 1,
                 "iou": 1.0, "merkez_hatası": 0.0, "bulundu": True,
                 "işlem_süresi_sn": elapsed, "değerlendir": False}]
        for index in range(1, len(sequence.ground_truth)):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Video etiketten erken bitti: {name}, kare {index + 1}")
            started = time.perf_counter()
            view = processed(frame, condition, method)
            found, box = model.update(view)
            elapsed = time.perf_counter() - started
            truth = sequence.ground_truth[index]
            if found:
                center = (box[0] + box[2] / 2, box[1] + box[3] / 2)
                actual = (truth[0] + truth[2] / 2, truth[1] + truth[3] / 2)
                error = float(np.hypot(center[0] - actual[0], center[1] - actual[1]))
            else:
                error = float("inf")
            rows.append({"video": name, "koşul": condition, "yöntem": method,
                         "kare": index + 1, "iou": iou(box, truth) if found else 0.0,
                         "merkez_hatası": error, "bulundu": bool(found),
                         "işlem_süresi_sn": elapsed, "değerlendir": True})
        return pd.DataFrame(rows)
    finally:
        capture.release()


def summarize(frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (name, condition, method), group in frames.loc[frames["değerlendir"]].groupby(
            ["video", "koşul", "yöntem"], sort=False):
        overlap = group["iou"].to_numpy()
        thresholds = np.linspace(0, 1, 101)
        success_curve = np.asarray([(overlap > value).mean() for value in thresholds])
        rows.append({"video": name, "koşul": condition, "yöntem": method,
                     "kare": len(group), "ortalama_IoU": float(overlap.mean()),
                     "başarı_AUC": float(np.trapezoid(success_curve, thresholds)),
                     "IoU_0.5_başarı": float((overlap >= .5).mean()),
                     "merkez_20px": float((group["merkez_hatası"] <= 20).mean()),
                     "FPS": float(len(group) / group["işlem_süresi_sn"].sum())})
    return pd.DataFrame(rows)
