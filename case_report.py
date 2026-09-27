"""Etiketli tek koşul için çevrimdışı, yazdırılabilir deney kartı üret."""

from html import escape

import pandas as pd

from failure_analysis import analyze_pair, paired_iou_interval, summarize_robustness
from tracking import METHODS


def _pct(value: float) -> str:
    return f"%{value * 100:.1f}"


def case_report_html(summary: pd.DataFrame, frames: pd.DataFrame, manifest: dict,
                     video: str, condition: str, method: str) -> bytes:
    """Kaydedilmiş etiketli deneyleri kullan; kullanıcı videosuna sonuç atama."""
    if method not in METHODS[1:]:
        raise ValueError("Karşılaştırma yöntemi gerekli.")
    selected = summary.loc[(summary["video"] == video) & (summary["koşul"] == condition)]
    if set(selected["yöntem"]) != set(METHODS) or len(selected) != len(METHODS):
        raise ValueError("Bu video/koşul için üç deneyin sonucu gerekli.")
    pair = analyze_pair(frames, video, condition, method)
    uncertainty = paired_iou_interval(frames, video, condition, method)
    robustness = summarize_robustness(frames)
    diagnostic = robustness.loc[(robustness["video"] == video) &
                                (robustness["koşul"] == condition)]
    if set(diagnostic["yöntem"]) != set(METHODS):
        raise ValueError("Sağlamlık analizi eksik.")

    metrics = []
    for name in METHODS:
        row = selected.loc[selected["yöntem"] == name].iloc[0]
        metrics.append("<tr><th scope='row'>" + escape(name) + "</th>" +
                       f"<td>{_pct(row['başarı_AUC'])}</td>" +
                       f"<td>{_pct(row['ortalama_IoU'])}</td>" +
                       f"<td>{_pct(row['IoU_0.5_başarı'])}</td>" +
                       f"<td>{_pct(row['merkez_20px'])}</td>" +
                       f"<td>{row['FPS']:.0f}</td></tr>")
    losses = []
    for name in METHODS:
        row = diagnostic.loc[diagnostic["yöntem"] == name].iloc[0]
        losses.append("<tr><th scope='row'>" + escape(name) + "</th>" +
                      f"<td>{_pct(row['düşük_IoU_oranı'])}</td>" +
                      f"<td>{int(row['en_uzun_kare'])} kare</td>" +
                      f"<td>{int(row['model_kayıp_bildirimi'])}</td>" +
                      f"<td>{int(row['kutu_var_ama_düşük_IoU'])}</td></tr>")
    candidate = selected.loc[selected["yöntem"] == method].iloc[0]
    baseline = selected.loc[selected["yöntem"] == "İşlemsiz"].iloc[0]
    model_hash = manifest.get("sha256", {}).get("assets/vittrack.onnx", "bilinmiyor")
    html = f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRACKLAB 38 · {escape(video)} / {escape(condition)}</title>
<style>
@page {{size:A4;margin:16mm}}
*{{box-sizing:border-box}}body{{font:14px/1.5 Arial,sans-serif;color:#19313c;max-width:880px;margin:28px auto;padding:0 22px}}
header{{background:#103746;color:#fff;padding:22px 26px;border-radius:12px}}header small{{color:#8ee6d6;letter-spacing:.14em;font-weight:bold}}
h1{{font-size:27px;margin:6px 0 2px}}h2{{font-size:17px;margin:22px 0 8px;color:#124255}}
.lead{{color:#d7e8eb;margin:0}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:17px 0}}
.card{{background:#eaf5f3;border:1px solid #bed9d6;padding:12px;border-radius:9px}}.card b{{font-size:19px;color:#095d57;display:block}}
.card span{{font-size:12px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:7px 8px;border-bottom:1px solid #d6e5e8;text-align:right}}
th:first-child,td:first-child{{text-align:left}}thead{{background:#e7f0f2}}p{{margin:7px 0}}.note{{background:#f3f7f8;border-left:3px solid #48a79c;padding:10px 13px}}
footer{{border-top:1px solid #d6e5e8;margin-top:24px;padding-top:10px;color:#526c75;font-size:11px;overflow-wrap:anywhere}}
@media print{{body{{margin:0;padding:0}}header,.card,.note{{print-color-adjust:exact;-webkit-print-color-adjust:exact}}}}
</style></head><body>
<header><small>TRACKLAB 38 / ETİKETLİ DENEY KARTI</small><h1>{escape(video)} · {escape(condition)}</h1>
<p class="lead">Aynı ViTTrack modeli, aynı ilk gerçek kutu, üç ön işleme yöntemi</p></header>
<div class="grid"><div class="card"><b>{_pct(candidate['başarı_AUC'])}</b><span>{escape(method)} başarı AUC</span></div>
<div class="card"><b>{(candidate['başarı_AUC']-baseline['başarı_AUC'])*100:+.1f} puan</b><span>İşlemsize göre AUC farkı</span></div>
<div class="card"><b>{int(candidate['kare'])}</b><span>Ölçülen etiketli kare</span></div></div>
<h2>Üç yöntemin tam video sonucu</h2><table><thead><tr><th>Yöntem</th><th>AUC</th><th>Ort. IoU</th>
<th>IoU ≥ 0,5</th><th>Merkez ≤ 20 px</th><th>İşlem FPS</th></tr></thead><tbody>{''.join(metrics)}</tbody></table>
<h2>İzleme kaybı ve toparlanma bağlamı</h2><table><thead><tr><th>Yöntem</th><th>IoU &lt; 0,2</th>
<th>En uzun dönem</th><th>Model kayıp dedi</th><th>Kutu var, IoU düşük</th></tr></thead>
<tbody>{''.join(losses)}</tbody></table>
<p><strong>{escape(method)} karşılaştırması:</strong> {pair.better_share*100:.1f}% karede daha yüksek IoU.
En iyi 60 karelik aralık {pair.best.start}–{pair.best.end} (ortalama fark {pair.best.value:+.3f});
en kötü aralık {pair.worst.start}–{pair.worst.end} ({pair.worst.value:+.3f}).</p>
<div class="note"><strong>Kareler arası değişkenlik:</strong> Eş karelerde ortalama IoU farkı
{uncertainty.difference*100:+.1f} puan; 60 karelik bloklarla 1000 tekrarlı keşifsel %95 aralık
{uncertainty.lower*100:+.1f} ile {uncertainty.upper*100:+.1f} puan.</div>
<h2>Nasıl ölçüldü?</h2><p>İlk etiketli karede her yöntem aynı gerçek kutuyla başlatıldı; o kare ölçülmedi.
Takip kaybolduğunda gerçek kutudan yeniden başlatılmadı. Başarı AUC, IoU eşikleri 0–1 boyunca başarı eğrisinin alanıdır.
FPS sadece ön işleme ve takip işlemini kapsar; video çözme ve MP4 kodlama hariçtir.</p>
<p><strong>Sınır:</strong> Bu, tek etiketli örnek video ve seçili görüntü koşulunun sonucudur.
Yapay karanlık/bulanıklık gerçek çekim koşullarının tamamını temsil etmez. Keşifsel aralık başka videolara başarı garantisi vermez.</p>
<footer>Deney parmak izi: {escape(str(manifest.get('cache_key','bilinmiyor')))} ·
OpenCV {escape(str(manifest.get('opencv','?')))} · Model SHA-256: {escape(str(model_hash))}<br>
TRACKLAB 38 · Yerel kaydedilmiş deneylerden üretildi · Tarayıcıda Yazdır → PDF olarak kaydet</footer>
</body></html>"""
    return html.encode("utf-8")
