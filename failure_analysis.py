"""Kare düzeyi kayıtlardan iyileşme ve takip kaybı aralıkları çıkar."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Interval:
    start: int
    end: int
    value: float


@dataclass(frozen=True)
class PairAnalysis:
    best: Interval
    worst: Interval
    baseline_loss: Interval | None
    candidate_loss: Interval | None
    better_share: float
    worse_share: float
    window: int


@dataclass(frozen=True)
class PairedInterval:
    difference: float
    lower: float
    upper: float
    block: int
    frames: int


def paired_iou_interval(frames: pd.DataFrame, video: str, condition: str,
                        method: str, block: int = 60, repeats: int = 1000,
                        seed: int = 38) -> PairedInterval:
    """Eş kare IoU farkını ardışık kare bloklarıyla yeniden örnekle (keşifsel %95 GA)."""
    if method == "İşlemsiz" or block < 1 or repeats < 2:
        raise ValueError("Karşılaştırma yöntemi, blok ve tekrar sayısı geçerli olmalı.")
    selected = frames.loc[(frames["video"] == video) & (frames["koşul"] == condition) &
                          frames["yöntem"].isin(("İşlemsiz", method)) & frames["değerlendir"]]
    if selected.duplicated(["kare", "yöntem"]).any():
        raise ValueError("Aynı yöntem için yinelenen kare ölçümü var.")
    table = selected.pivot(index="kare", columns="yöntem", values="iou").sort_index()
    if not {"İşlemsiz", method}.issubset(table.columns) or table.empty or table.isna().any().any():
        raise ValueError("İki yöntemin aynı karelerde eksiksiz ölçülmesi gerekli.")
    differences = (table[method] - table["İşlemsiz"]).to_numpy(dtype=float)
    count = len(differences)
    size = min(block, count)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, count - size + 1, size=(repeats, (count + size - 1) // size))
    indices = (starts[:, :, None] + np.arange(size)).reshape(repeats, -1)[:, :count]
    draws = differences[indices].mean(axis=1)
    lower, upper = np.quantile(draws, [.025, .975])
    return PairedInterval(float(differences.mean()), float(lower), float(upper), size, count)


def summarize_robustness(frames: pd.DataFrame, threshold: float = .2) -> pd.DataFrame:
    """Düşük IoU dönemlerinin sıklığını, süresini ve modelin kayıp bildirimini ayır."""
    if not 0 < threshold < 1:
        raise ValueError("IoU eşiği 0 ile 1 arasında olmalı.")
    rows = []
    selected = frames.loc[frames["değerlendir"]]
    for (name, condition, method), group in selected.groupby(["video", "koşul", "yöntem"], sort=False):
        group = group.sort_values("kare")
        numbers = group["kare"].to_numpy(dtype=int)
        overlaps = group["iou"].to_numpy(dtype=float)
        found = group["bulundu"].to_numpy(dtype=bool)
        low = overlaps < threshold
        starts = np.flatnonzero(low & ~np.r_[False, low[:-1]])
        ends = np.flatnonzero(low & ~np.r_[low[1:], False])
        lengths = ends - starts + 1
        if len(lengths):
            longest = int(np.argmax(lengths))
            first_low = int(numbers[starts[0]])
            longest_start = int(numbers[starts[longest]])
            longest_end = int(numbers[ends[longest]])
            longest_count = int(lengths[longest])
        else:
            first_low = longest_start = longest_end = longest_count = 0
        rows.append({"video": name, "koşul": condition, "yöntem": method,
                     "düşük_IoU_oranı": float(low.mean()),
                     "düşük_kare": int(low.sum()), "ilk_düşük_kare": first_low,
                     "düşük_dönem": int(len(starts)),
                     "toparlanma": int(np.sum(ends < len(low) - 1)),
                     "en_uzun_başlangıç": longest_start, "en_uzun_bitiş": longest_end,
                     "en_uzun_kare": longest_count,
                     "model_kayıp_bildirimi": int((~found).sum()),
                     "kutu_var_ama_düşük_IoU": int((low & found).sum())})
    return pd.DataFrame(rows)


def evaluation_curves(frames: pd.DataFrame, video: str, condition: str,
                      methods: tuple[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IoU başarı ve merkez uzaklığı doğruluk eğrilerini aynı karelerden hesapla."""
    subset = frames.loc[(frames["video"] == video) & (frames["koşul"] == condition) &
                        frames["yöntem"].isin(methods) & frames["değerlendir"]]
    success_thresholds = np.linspace(0, 1, 101)
    center_thresholds = np.arange(0, 51)
    success = pd.DataFrame({"IoU eşiği": success_thresholds})
    precision = pd.DataFrame({"Merkez eşiği (px)": center_thresholds})
    for method in methods:
        group = subset.loc[subset["yöntem"] == method]
        if group.empty:
            raise ValueError(f"Kare sonuçları eksik: {method}")
        overlaps = group["iou"].to_numpy(dtype=float)
        distances = group["merkez_hatası"].to_numpy(dtype=float)
        success[method] = np.mean(overlaps[:, None] > success_thresholds[None, :], axis=0)
        precision[method] = np.mean(distances[:, None] <= center_thresholds[None, :], axis=0)
    return success, precision


def _longest_loss(numbers: np.ndarray, overlaps: np.ndarray, threshold: float = .2) -> Interval | None:
    """IoU eşiğinin altındaki en uzun ardışık bölüm (ilk kare zaten hariç)."""
    longest: Interval | None = None
    start = None
    for index in range(len(overlaps) + 1):
        low = index < len(overlaps) and overlaps[index] < threshold
        if low and start is None:
            start = index
        elif not low and start is not None:
            run = Interval(int(numbers[start]), int(numbers[index - 1]), float(index - start))
            if longest is None or run.value > longest.value:
                longest = run
            start = None
    return longest


def analyze_pair(frames: pd.DataFrame, video: str, condition: str,
                 method: str, window: int = 60) -> PairAnalysis:
    selected = frames.loc[(frames["video"] == video) & (frames["koşul"] == condition) &
                          (frames["yöntem"].isin(["İşlemsiz", method])) & frames["değerlendir"]]
    table = selected.pivot(index="kare", columns="yöntem", values="iou").sort_index()
    if method == "İşlemsiz" or len(table) == 0 or table.isna().any().any():
        raise ValueError("Karşılaştırma için iki yöntemin de kare ölçümleri gerekli.")
    baseline = table["İşlemsiz"].to_numpy(dtype=float)
    candidate = table[method].to_numpy(dtype=float)
    numbers = table.index.to_numpy(dtype=int)
    size = min(max(1, window), len(numbers))
    rolling = np.convolve(candidate - baseline, np.ones(size) / size, mode="valid")
    best_index, worst_index = int(np.argmax(rolling)), int(np.argmin(rolling))
    best = Interval(int(numbers[best_index]), int(numbers[best_index + size - 1]), float(rolling[best_index]))
    worst = Interval(int(numbers[worst_index]), int(numbers[worst_index + size - 1]), float(rolling[worst_index]))
    return PairAnalysis(best, worst, _longest_loss(numbers, baseline),
                        _longest_loss(numbers, candidate),
                        float(np.mean(candidate > baseline + 1e-9)),
                        float(np.mean(candidate < baseline - 1e-9)), size)
