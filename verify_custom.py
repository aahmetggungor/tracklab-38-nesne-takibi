"""Etiketsiz kullanıcı videosunu yerel sentetik görüntüyle uçtan uca denetle."""

from pathlib import Path
import tempfile

import cv2
import numpy as np

from custom_video import inspect_custom, render_custom


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "sentetik.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
        if not writer.isOpened():
            raise RuntimeError("Sentetik doğrulama videosu oluşturulamadı.")
        for index in range(12):
            frame = np.full((120, 160, 3), 65, dtype=np.uint8)
            cv2.rectangle(frame, (42 + index, 30), (82 + index, 70), (230, 230, 230), -1)
            writer.write(frame)
        writer.release()
        data = source.read_bytes()
        info = inspect_custom(data, ".mp4")
        assert (info.width, info.height) == (160, 120)
        result, stats = render_custom(data, ".mp4", (42, 30, 40, 40), "Bulanık", "CLAHE")
        assert stats["kare"] == 12 and result
        output = Path(directory) / "sonuc.mp4"
        output.write_bytes(result)
        capture = cv2.VideoCapture(str(output))
        try:
            assert capture.isOpened()
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
            ok, frame = capture.read()
            assert ok and frame.shape[:2] == (320, 960)
        finally:
            capture.release()
        try:
            render_custom(data, ".mp4", (155, 30, 40, 40), "Orijinal", "CLAHE")
        except ValueError:
            pass
        else:
            raise AssertionError("Video dışına taşan kutu reddedilmeliydi.")
    print("OK: kullanıcı videosu önizlemesi, çift takip, MP4 ve kutu sınırı.")


if __name__ == "__main__":
    main()
