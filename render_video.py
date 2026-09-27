"""İki takip koşulunu aynı karelerde yan yana MP4 olarak göster."""

from pathlib import Path
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np

from tracking import _seek_to_start, iou, load_sequence, processed, tracker


PANEL_WIDTH = 480
PANEL_HEIGHT = 320


def _fit_geometry(width: int, height: int) -> tuple[int, int, int, int]:
    """Kaynak en-boy oranını koruyarak panelin ortasına yerleştir."""
    if width < 1 or height < 1:
        raise ValueError("Boş video karesi gösterilemez.")
    scale = min(PANEL_WIDTH / width, PANEL_HEIGHT / height)
    target_width = min(PANEL_WIDTH, max(1, round(width * scale)))
    target_height = min(PANEL_HEIGHT, max(1, round(height * scale)))
    return target_width, target_height, (PANEL_WIDTH - target_width) // 2, (PANEL_HEIGHT - target_height) // 2


def _panel(frame, name: str, truth, box, found: bool, frame_index: int):
    height, width = frame.shape[:2]
    target_width, target_height, left, top = _fit_geometry(width, height)
    image = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), (7, 22, 32), dtype=np.uint8)
    interpolation = cv2.INTER_AREA if target_width < width or target_height < height else cv2.INTER_LINEAR
    image[top:top + target_height, left:left + target_width] = cv2.resize(
        frame, (target_width, target_height), interpolation=interpolation)
    scale_x, scale_y = target_width / width, target_height / height

    def draw(bounds, color, thickness):
        x, y, w, h = bounds
        a = (left + round(x * scale_x), top + round(y * scale_y))
        b = (left + round((x + w) * scale_x), top + round((y + h) * scale_y))
        cv2.rectangle(image, a, b, color, thickness)

    draw(truth, (0, 220, 255), 2)
    if found:
        draw(box, (114, 218, 141), 2)
    cv2.rectangle(image, (0, 0), (PANEL_WIDTH, 31), (7, 22, 32), -1)
    overlap = iou(box, truth) if found else 0.0
    label = f"{name} | kare {frame_index} | IoU {overlap:.2f}" if found else f"{name} | kare {frame_index} | kayip"
    cv2.putText(image, label,
                (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .53, (245, 245, 245), 1, cv2.LINE_AA)
    return image


def render_comparison(name: str, condition: str, method: str, max_frames: int = 180,
                      start_frame: int = 1) -> bytes:
    if method == "İşlemsiz":
        raise ValueError("Karşılaştırma yöntemi İşlemsiz olamaz.")
    sequence = load_sequence(name)
    if not 1 <= start_frame <= len(sequence.ground_truth) or max_frames < 1:
        raise ValueError("Geçersiz kare aralığı.")
    capture = cv2.VideoCapture(str(sequence.video))
    if not capture.isOpened():
        raise ValueError(f"Video açılamadı: {name}")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temporary:
        destination = Path(temporary.name)
    encoder = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 25.0
        first = _seek_to_start(capture, sequence.first_frame)
        baseline, treatment = tracker(), tracker()
        init = tuple(int(round(value)) for value in sequence.ground_truth[0])
        left_view = processed(first, condition, "İşlemsiz")
        right_view = processed(first, condition, method)
        baseline.init(left_view, init)
        treatment.init(right_view, init)
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "960x320", "-r", str(fps),
                   "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE)
        last_index = min(start_frame - 1 + max_frames, len(sequence.ground_truth))
        for index in range(last_index):
            if index:
                ok, frame = capture.read()
                if not ok:
                    break
                left_view = processed(frame, condition, "İşlemsiz")
                right_view = processed(frame, condition, method)
                left_found, left_box = baseline.update(left_view)
                right_found, right_box = treatment.update(right_view)
            else:
                frame = first
                left_found = right_found = True
                left_box = right_box = init
            if index + 1 < start_frame:
                continue
            truth = sequence.ground_truth[index]
            left = _panel(left_view, "İşlemsiz", truth,
                          left_box, left_found, index + 1)
            right = _panel(right_view, method, truth,
                           right_box, right_found, index + 1)
            encoder.stdin.write(cv2.hconcat([left, right]).tobytes())
        encoder.stdin.close()
        error = encoder.stderr.read().decode("utf-8", "replace")
        if encoder.wait() != 0:
            raise RuntimeError(f"MP4 üretilemedi: {error[-300:]}")
        return destination.read_bytes()
    finally:
        capture.release()
        if encoder and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        destination.unlink(missing_ok=True)
