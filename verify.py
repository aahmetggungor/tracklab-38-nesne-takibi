"""İndirilen veri, kare hizası ve rapor ölçümlerini denetle."""

import json

import numpy as np
import pandas as pd

from case_report import case_report_html
from download_assets import FILES, ROOT, valid
from failure_analysis import analyze_pair, evaluation_curves, paired_iou_interval, summarize_robustness
from result_integrity import validate_result_bundle
from run_experiments import build_manifest
from tracking import CONDITIONS, METHODS, SEQUENCES, iou, load_sequence, summarize


def main() -> None:
    manifest = json.loads((ROOT / "results" / "manifest.json").read_text(encoding="utf-8"))
    current = build_manifest()
    for key in ("tracker", "opencv", "numpy", "pandas", "python", "sequences", "conditions",
                "methods", "comparison", "conditions_detail", "sha256", "cache_key"):
        assert manifest[key] == current[key], f"Deney girdisi değişmiş: {key}"
    for relative, (_, expected_hash) in FILES.items():
        assert valid(ROOT / relative, expected_hash), relative
    assert {name: len(load_sequence(name).ground_truth) for name in SEQUENCES} == {
        "david": 471, "dudek": 1145, "faceocc2": 812}
    assert load_sequence("david").first_frame == 300
    assert iou((0, 0, 10, 10), np.asarray((0, 0, 10, 10))) == 1.0
    assert iou((0, 0, 10, 10), np.asarray((20, 20, 10, 10))) == 0.0

    frames = pd.read_parquet(ROOT / "results" / "frame_metrics.parquet")
    summary = pd.read_csv(ROOT / "results" / "summary.csv")
    validate_result_bundle(summary, frames, manifest, current)
    stale = dict(manifest, cache_key="eski-deney")
    try:
        validate_result_bundle(summary, frames, stale, current)
    except ValueError:
        pass
    else:
        raise AssertionError("Eski deney parmak izi fark edilmedi.")
    altered = summary.copy()
    altered.loc[0, "başarı_AUC"] += .1
    try:
        validate_result_bundle(altered, frames, manifest, current)
    except ValueError:
        pass
    else:
        raise AssertionError("Karelerle tutarsız AUC fark edilmedi.")
    assert len(summary) == len(SEQUENCES) * len(CONDITIONS) * len(METHODS) == 27
    assert not frames.duplicated(["video", "koşul", "yöntem", "kare"]).any()
    for row in summary.itertuples(index=False):
        assert row.kare == len(load_sequence(row.video).ground_truth) - 1
    recalculated = summarize(frames).sort_values(["video", "koşul", "yöntem"]).reset_index(drop=True)
    stored = summary.sort_values(["video", "koşul", "yöntem"]).reset_index(drop=True)
    for column in ("ortalama_IoU", "başarı_AUC", "IoU_0.5_başarı", "merkez_20px", "FPS"):
        assert np.allclose(recalculated[column], stored[column]), column
    assert frames["iou"].between(0, 1).all()
    pair = analyze_pair(frames, "david", "Orijinal", "CLAHE")
    assert pair.best.value > 0 and pair.window == 60
    assert 2 <= pair.best.start <= pair.best.end <= len(load_sequence("david").ground_truth)
    assert pair.baseline_loss is not None and pair.baseline_loss.value > 0
    assert pair.candidate_loss is not None and pair.candidate_loss.value > 0
    success, center = evaluation_curves(frames, "david", "Orijinal", ("İşlemsiz", "CLAHE"))
    clahe = summary.loc[(summary["video"] == "david") & (summary["koşul"] == "Orijinal") &
                        (summary["yöntem"] == "CLAHE")].iloc[0]
    assert np.isclose(np.trapezoid(success["CLAHE"], success["IoU eşiği"]), clahe["başarı_AUC"])
    assert np.isclose(center.loc[center["Merkez eşiği (px)"] == 20, "CLAHE"].iloc[0],
                      clahe["merkez_20px"])
    diagnostic = summarize_robustness(frames)
    assert len(diagnostic) == 27
    david = diagnostic.loc[(diagnostic["video"] == "david") &
                           (diagnostic["koşul"] == "Orijinal") &
                           (diagnostic["yöntem"] == "İşlemsiz")].iloc[0]
    assert david["en_uzun_başlangıç"] == pair.baseline_loss.start
    assert david["en_uzun_bitiş"] == pair.baseline_loss.end
    assert david["en_uzun_kare"] == pair.baseline_loss.value
    miniature = pd.DataFrame({"video": ["örnek"] * 8, "koşul": ["Orijinal"] * 8,
                              "yöntem": ["İşlemsiz"] * 8, "kare": range(1, 9),
                              "iou": [1, .8, .1, .05, .7, .1, .8, .1],
                              "bulundu": [True, True, True, True, True, False, True, True],
                              "değerlendir": [False] + [True] * 7})
    mini_result = summarize_robustness(miniature).iloc[0]
    assert (mini_result["düşük_kare"], mini_result["düşük_dönem"],
            mini_result["toparlanma"], mini_result["en_uzun_kare"]) == (4, 3, 2, 2)
    assert mini_result["kutu_var_ama_düşük_IoU"] == 3
    assert mini_result["model_kayıp_bildirimi"] == 1
    paired = pd.DataFrame({"video": ["örnek"] * 16, "koşul": ["Orijinal"] * 16,
                           "yöntem": ["İşlemsiz"] * 8 + ["CLAHE"] * 8,
                           "kare": list(range(1, 9)) * 2,
                           "iou": [.1, .2, .3, .4, .5, .6, .7, .1] +
                                  [.3, .4, .5, .6, .7, .8, .9, .3],
                           "değerlendir": [False] + [True] * 7 + [False] + [True] * 7})
    interval = paired_iou_interval(paired, "örnek", "Orijinal", "CLAHE", block=3, repeats=100)
    assert interval.frames == 7 and interval.block == 3
    assert np.allclose([interval.difference, interval.lower, interval.upper], [.2, .2, .2])
    assert interval == paired_iou_interval(paired, "örnek", "Orijinal", "CLAHE",
                                           block=3, repeats=100)
    missing = paired.loc[~((paired["yöntem"] == "CLAHE") & (paired["kare"] == 4))]
    try:
        paired_iou_interval(missing, "örnek", "Orijinal", "CLAHE")
    except ValueError:
        pass
    else:
        raise AssertionError("Eksik eş kare fark edilmedi.")
    card = case_report_html(summary, frames, manifest, "david", "Orijinal", "CLAHE").decode("utf-8")
    assert "<html lang=\"tr\">" in card and "david · Orijinal" in card
    assert "+44.9 puan" in card and "%71.3" in card
    assert manifest["cache_key"] in card and manifest["sha256"]["assets/vittrack.onnx"] in card
    assert "<script" not in card and "https://" not in card
    print("OK: model/veri hash, etiket hizası, 27 deney ve tüm ölçüm sonuçları tutarlı.")


if __name__ == "__main__":
    main()
