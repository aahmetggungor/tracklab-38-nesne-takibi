"""Etiketsiz kullanıcı videosunda iki aynı takipçiyi yan yana çalıştır."""

from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np

from tracking import CONDITIONS, METHODS, processed, tracker


MAX_BYTES = 30 * 1024 * 1024
MAX_PIXELS = 4_000_000
MAX_SECONDS = 20
MAX_OUTPUT_FRAMES = 180


@dataclass(frozen=True)
class CustomInfo:
    preview: np.ndarray
    width: int
    height: int
    fps: float
    duration: float


def _source(data: bytes, suffix: str) -> Path:
    if not data or len(data) > MAX_BYTES or suffix not in (".mp4", ".mov", ".avi"):
        raise ValueError("MP4/MOV/AVI video gerekli; dosya en fazla 30 MB olmalı.")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(data)
        return Path(temporary.name)


def _metadata(capture: cv2.VideoCapture) -> tuple[int, int, float, float]:
    if not capture.isOpened():
        raise ValueError("Video açılamadı.")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 16 or height < 16 or width * height > MAX_PIXELS:
        raise ValueError("Video çözünürlüğü desteklenmiyor; en fazla 4 megapiksel kullanın.")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not 1 <= fps <= 240:
        fps = 25.0
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height, fps, count / fps if count > 0 else 0.0


def inspect_custom(data: bytes, suffix: str) -> CustomInfo:
    source = _source(data, suffix)
    try:
        capture = cv2.VideoCapture(str(source))
        try:
            width, height, fps, duration = _metadata(capture)
            ok, frame = capture.read()
            if not ok:
                raise ValueError("İlk video karesi okunamadı.")
            return CustomInfo(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width, height, fps, duration)
        finally:
            capture.release()
    finally:
        source.unlink(missing_ok=True)


def preview_roi(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    result = image.copy()
    x, y, w, h = box
    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 198, 48), 3)
    return result


def _panel(frame: np.ndarray, label: str, box: tuple[int, int, int, int], found: bool,
           frame_index: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(480 / width, 288 / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    view = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    canvas = np.full((320, 480, 3), (17, 31, 44), dtype=np.uint8)
    left, top = (480 - size[0]) // 2, 32 + (288 - size[1]) // 2
    canvas[top:top + size[1], left:left + size[0]] = view
    if found:
        x, y, w, h = box
        cv2.rectangle(canvas, (left + round(x * scale), top + round(y * scale)),
                      (left + round((x + w) * scale), top + round((y + h) * scale)),
                      (114, 218, 141), 2)
    cv2.putText(canvas, f"{label} | kare {frame_index}" + ("" if found else " | kayip"),
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def render_custom(data: bytes, suffix: str, box: tuple[int, int, int, int],
                  condition: str, method: str) -> tuple[bytes, dict]:
    if condition not in CONDITIONS or method not in METHODS[1:]:
        raise ValueError("Geçersiz koşul veya ön işleme yöntemi.")
    source = _source(data, suffix)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temporary:
        destination = Path(temporary.name)
    encoder = None
    try:
        capture = cv2.VideoCapture(str(source))
        try:
            width, height, fps, _ = _metadata(capture)
            x, y, w, h = box
            if w < 8 or h < 8 or x < 0 or y < 0 or x + w > width or y + h > height:
                raise ValueError("İlk hedef kutusu video sınırları içinde ve en az 8×8 piksel olmalı.")
            scale = min(1.0, 960 / max(width, height))
            out_width, out_height = round(width * scale), round(height * scale)
            sx, sy = round(x * scale), round(y * scale)
            sw = min(round(w * scale), out_width - sx)
            sh = min(round(h * scale), out_height - sy)
            if sw < 8 or sh < 8:
                raise ValueError("Hedef kutusu küçültülmüş videoda en az 8×8 piksel olmalı.")
            scaled_box = (sx, sy, sw, sh)
            step = max(1, math.ceil(fps / 12))
            out_fps = fps / step
            command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                       "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "960x320", "-r", f"{out_fps:.5f}",
                       "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                       "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE)
            baseline = treatment = None
            processed_count = lost_baseline = lost_treatment = source_index = 0
            while source_index < math.ceil(MAX_SECONDS * fps) and processed_count < MAX_OUTPUT_FRAMES:
                ok, frame = capture.read()
                if not ok:
                    break
                source_index += 1
                if (source_index - 1) % step:
                    continue
                if scale < 1:
                    frame = cv2.resize(frame, (out_width, out_height),
                                       interpolation=cv2.INTER_AREA)
                left_view = processed(frame, condition, "İşlemsiz")
                right_view = processed(frame, condition, method)
                if baseline is None:
                    baseline, treatment = tracker(), tracker()
                    baseline.init(left_view, scaled_box)
                    treatment.init(right_view, scaled_box)
                    left_found = right_found = True
                    left_box = right_box = scaled_box
                else:
                    left_found, left_box = baseline.update(left_view)
                    right_found, right_box = treatment.update(right_view)
                    lost_baseline += not bool(left_found)
                    lost_treatment += not bool(right_found)
                combined = cv2.hconcat([
                    _panel(left_view, "İşlemsiz", left_box, bool(left_found), source_index),
                    _panel(right_view, method, right_box, bool(right_found), source_index),
                ])
                encoder.stdin.write(combined.tobytes())
                processed_count += 1
            encoder.stdin.close()
            error = encoder.stderr.read().decode("utf-8", "replace")
            if encoder.wait() != 0 or not processed_count:
                raise RuntimeError(f"MP4 üretilemedi: {error[-300:]}")
            return destination.read_bytes(), {"kare": processed_count, "fps": round(out_fps, 1),
                                              "işlemsiz_kayıp": lost_baseline, "yöntem_kayıp": lost_treatment}
        finally:
            capture.release()
    finally:
        if encoder and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        source.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
