"""27 takip deneyini çalıştır; kesinti olursa doğrulanmış ara sonuçlardan devam et."""

import argparse
from hashlib import sha256
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd

from tracking import CONDITIONS, METHODS, ROOT, SEQUENCES, load_sequence, run_one, summarize


def build_manifest() -> dict:
    """Deney girdileri ve çekirdek kod için yeniden üretilebilir parmak izi."""
    inputs = [ROOT / "assets" / "vittrack.onnx", ROOT / "tracking.py",
              ROOT / "run_experiments.py"]
    for name in SEQUENCES:
        inputs.extend((ROOT / "data" / f"{name}.webm", ROOT / "data" / f"{name}_gt.txt",
                       ROOT / "data" / f"{name}.yml"))
    manifest = {"tracker": "OpenCV ViTTrack 2023sep", "opencv": cv2.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "python": platform.python_version(), "manifest_utc": datetime.now(timezone.utc).isoformat(),
            "sequences": list(SEQUENCES), "conditions": list(CONDITIONS), "methods": list(METHODS),
            "comparison": "ground truth initialization, no automatic reinitialization; first frame excluded",
            "conditions_detail": {"Karanlık": "pixel intensity x 0.40",
                                  "Bulanık": "GaussianBlur kernel 9x9, sigma 2.0"},
            "sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path.read_bytes()).hexdigest()
                       for path in inputs}}
    cache_input = {key: value for key, value in manifest.items() if key != "manifest_utc"}
    manifest["cache_key"] = sha256(json.dumps(cache_input, sort_keys=True,
                                              ensure_ascii=False).encode("utf-8")).hexdigest()[:20]
    return manifest


def _atomic_write(frame: pd.DataFrame, path: Path, kind: str) -> None:
    """Eksik yazılmış tabloyu geçerli sonuç olarak göstermemek için aynı dizinde değiştir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=f".{kind}", delete=False) as file:
        temporary = Path(file.name)
    try:
        if kind == "parquet":
            frame.to_parquet(temporary, index=False)
        else:
            frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint(path: Path, name: str, condition: str, method: str,
                     expected_rows: int) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        frame = pd.read_parquet(path)
        required = {"video", "koşul", "yöntem", "kare", "iou", "merkez_hatası",
                    "bulundu", "işlem_süresi_sn", "değerlendir"}
        if not required <= set(frame.columns) or len(frame) != expected_rows:
            return None
        if not (frame["video"].eq(name).all() and frame["koşul"].eq(condition).all() and
                frame["yöntem"].eq(method).all()):
            return None
        if not np.array_equal(frame["kare"].to_numpy(), np.arange(1, expected_rows + 1)):
            return None
        if frame["değerlendir"].to_numpy(dtype=bool).tolist() != [False] + [True] * (expected_rows - 1):
            return None
        if not frame["iou"].between(0, 1).all() or not (frame["işlem_süresi_sn"] >= 0).all():
            return None
        return frame
    except Exception:
        return None


def write_manifest(manifest: dict | None = None) -> None:
    path = ROOT / "results" / "manifest.json"
    path.parent.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".json", delete=False) as file:
        temporary = Path(file.name)
    try:
        temporary.write_text(json.dumps(manifest or build_manifest(), ensure_ascii=False, indent=2),
                             encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(force: bool = False) -> None:
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    manifest = build_manifest()
    checkpoint_dir = output / "checkpoints" / manifest["cache_key"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    reused = computed = 0
    run_number = 0
    for name in SEQUENCES:
        expected_rows = len(load_sequence(name).ground_truth)
        for condition in CONDITIONS:
            for method in METHODS:
                run_number += 1
                checkpoint = checkpoint_dir / f"run_{run_number:02d}.parquet"
                cached = None if force else _load_checkpoint(checkpoint, name, condition, method, expected_rows)
                if cached is not None:
                    print(f"[{run_number:02d}/27] Ara sonuç kullanıldı: {name} | {condition} | {method}", flush=True)
                    results.append(cached)
                    reused += 1
                    continue
                print(f"[{run_number:02d}/27] Hesaplanıyor: {name} | {condition} | {method}", flush=True)
                frame = run_one(name, condition, method)
                _atomic_write(frame, checkpoint, "parquet")
                results.append(frame)
                computed += 1
    frames = pd.concat(results, ignore_index=True)
    summary = summarize(frames)
    _atomic_write(frames, output / "frame_metrics.parquet", "parquet")
    _atomic_write(summary, output / "summary.csv", "csv")
    write_manifest(manifest)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Ara sonuç: {reused} kullanıldı, {computed} yeniden hesaplandı.")
    print(f"Kayıt: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Ara sonuçları kullanmadan tüm deneyleri yeniden çalıştır.")
    main(force=parser.parse_args().force)
