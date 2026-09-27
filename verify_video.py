"""Etiketli karşılaştırma klibinin en-boy oranını ve MP4 çıktısını doğrula."""

from pathlib import Path
import tempfile

import cv2
import numpy as np

from render_video import _fit_geometry, _panel, render_comparison


def main() -> None:
    assert _fit_geometry(320, 240) == (427, 320, 26, 0)
    assert _fit_geometry(720, 480) == (480, 320, 0, 0)
    frame = np.full((240, 320, 3), (40, 60, 80), dtype=np.uint8)
    panel = _panel(frame, "İşlemsiz", (80, 60, 80, 60), None, False, 2)
    assert panel.shape == (320, 480, 3)
    assert panel[100, 0].tolist() == [7, 22, 32]
    assert panel[100, 50].tolist() == [40, 60, 80]
    assert panel[80, 133].tolist() == [0, 220, 255]

    video = render_comparison("david", "Orijinal", "CLAHE", max_frames=30)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "karşılaştırma.mp4"
        output.write_bytes(video)
        capture = cv2.VideoCapture(str(output))
        try:
            assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
            ok, first = capture.read()
            assert ok and first.shape[:2] == (320, 960)
        finally:
            capture.release()
    print("OK: etiketli karşılaştırma en-boy oranı, kutu konumu ve 30 kare MP4.")


if __name__ == "__main__":
    main()
