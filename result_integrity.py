"""Arayüzün yalnızca güncel ve birbiriyle tutarlı etiketli sonuçları göstermesini sağla."""

from itertools import product
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from tracking import CONDITIONS, METHODS, SEQUENCES, summarize


def validate_result_bundle(summary: pd.DataFrame, frames: pd.DataFrame,
                           stored: dict, current: dict) -> None:
    """Hata halinde eski veya bozuk sonuçları arayüze sunmadan durdur."""
    identity_keys = ("tracker", "sequences", "conditions", "methods", "comparison",
                     "conditions_detail", "sha256")
    if any(stored.get(key) != current.get(key) for key in identity_keys):
        raise ValueError("Model, veri veya deney kodu değişmiş; sonuçlar yeniden üretilmeli.")
    fingerprint_source = {key: value for key, value in stored.items()
                          if key not in ("manifest_utc", "cache_key")}
    expected_key = sha256(json.dumps(fingerprint_source, sort_keys=True,
                                     ensure_ascii=False).encode("utf-8")).hexdigest()[:20]
    if stored.get("cache_key") != expected_key:
        raise ValueError("Deney kaydının parmak izi bozuk.")

    keys = ["video", "koşul", "yöntem"]
    expected = set(product(SEQUENCES, CONDITIONS, METHODS))
    if (not set(keys).issubset(summary.columns) or not set(keys).issubset(frames.columns)
            or len(summary) != len(expected) or summary.duplicated(keys).any()
            or set(map(tuple, summary[keys].to_numpy())) != expected
            or set(map(tuple, frames[keys].drop_duplicates().to_numpy())) != expected):
        raise ValueError("27 koşulun sonuç tablosu eksik veya yinelenmiş.")
    required = {"kare", "iou", "merkez_hatası", "bulundu", "işlem_süresi_sn", "değerlendir"}
    metrics = ("ortalama_IoU", "başarı_AUC", "IoU_0.5_başarı", "merkez_20px", "FPS")
    if not required.issubset(frames.columns) or not set(metrics).issubset(summary.columns):
        raise ValueError("Kare veya özet ölçüm sütunları eksik.")
    if frames.duplicated(keys + ["kare"]).any() or not frames["iou"].between(0, 1).all():
        raise ValueError("Kare ölçümleri yinelenmiş veya IoU aralık dışında.")

    measured = frames.loc[frames["değerlendir"]]
    counts = measured.groupby(keys).size().rename("kare")
    reported = summary.set_index(keys)["kare"]
    if not counts.sort_index().equals(reported.sort_index()):
        raise ValueError("Kare sayıları özet tablosuyla uyuşmuyor.")
    recalculated = summarize(frames).set_index(keys).sort_index()
    reported_summary = summary.set_index(keys).sort_index()
    if any(not np.allclose(recalculated[column].to_numpy(),
                           reported_summary[column].to_numpy(), rtol=1e-7, atol=1e-9)
           for column in metrics):
        raise ValueError("Özet metrikler kare sonuçlarıyla uyuşmuyor; deneyleri yeniden çalıştırın.")
