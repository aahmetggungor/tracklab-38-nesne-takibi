"""PDF 38: görüntü ön işlemenin derin takipçi performansına etkisi."""

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from case_report import case_report_html
from custom_video import inspect_custom, preview_roi, render_custom
from failure_analysis import analyze_pair, evaluation_curves, paired_iou_interval, summarize_robustness
from render_video import render_comparison
from result_integrity import validate_result_bundle
from run_experiments import build_manifest
from tracking import CONDITIONS, METHODS, ROOT, SEQUENCES


st.set_page_config(page_title="TRACKLAB 38 · Derin Takip Deneyi", page_icon="🎯", layout="wide")
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 85% -25%,#1b3f50 0,transparent 35%),#09141f;color:#edf5f6}
.block-container{max-width:1450px;padding-top:1.3rem;padding-bottom:4rem}
[data-testid="stSidebar"]{background:#0e2130;border-right:1px solid #2a4e5d}
[data-testid="stMetric"]{background:#122a38;border:1px solid #2b5766;border-radius:14px;padding:15px}
.hero{background:linear-gradient(115deg,#103746,#162838);border:1px solid #346b78;border-radius:20px;
padding:28px 32px;margin-bottom:24px;box-shadow:0 15px 45px #0003}
.hero small{color:#74ddcd;letter-spacing:.16em;font-weight:800}
.hero h1{font-size:clamp(28px,3.5vw,43px);line-height:1.16;margin:9px 0;color:#f4fcfb}
.hero p{color:#bdd4db;max-width:880px;line-height:1.6;margin:0}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero"><small>TRACKLAB 38 / GÖRÜNTÜ İŞLEME DENEYİ</small>
<h1>Görüntü değişince takip nasıl değişiyor?</h1>
<p>Aynı derin takipçi, aynı başlangıç kutusu ve aynı video. Yalnızca görüntü ön işlemesini değiştir;
etkisini gerçek konum etiketleriyle ve yan yana videoda gör.</p></div>
""", unsafe_allow_html=True)

summary_path = ROOT / "results" / "summary.csv"
frame_path = ROOT / "results" / "frame_metrics.parquet"
manifest_path = ROOT / "results" / "manifest.json"
if not summary_path.is_file() or not frame_path.is_file() or not manifest_path.is_file():
    st.error("Deney sonuçları bulunamadı. Proje klasöründe `python download_assets.py` ve `python run_experiments.py` çalıştırın.")
    st.stop()


@st.cache_data
def load_results(summary_mtime: float, frames_mtime: float, manifest_mtime: float):
    return (pd.read_csv(summary_path), pd.read_parquet(frame_path),
            json.loads(manifest_path.read_text(encoding="utf-8")))


@st.cache_data(show_spinner="Video önizlemesi hazırlanıyor…")
def custom_preview(data: bytes, suffix: str):
    return inspect_custom(data, suffix)


summary, frames, manifest = load_results(summary_path.stat().st_mtime,
                                         frame_path.stat().st_mtime,
                                         manifest_path.stat().st_mtime)
try:
    validate_result_bundle(summary, frames, manifest, build_manifest())
except (OSError, ValueError, KeyError) as error:
    st.error(f"Deney sonuçları doğrulanamadı: {error}")
    st.info("Proje klasöründe `python run_experiments.py` ve ardından `python verify.py` çalıştırın.")
    st.stop()
current_runtime = build_manifest()
version_keys = ("opencv", "numpy", "pandas", "python")
if any(manifest.get(key) != current_runtime.get(key) for key in version_keys):
    st.info("Bu arayüz kaydedilmiş deney sonuçlarını gösteriyor. Deneyin Python/paket sürümleri "
            "sunucudakilerden farklı olabilir; yeni deney çalıştırılırsa sonuçlar yeniden doğrulanmalıdır.")
if not st.context.url.startswith(("http://localhost", "http://127.0.0.1")):
    st.warning("Çevrimiçi demo: yüklediğiniz video sunucuda işlenir. Gerçek kişisel veya gizli "
               "video yerine sentetik örnek kullanın.")
with st.sidebar:
    st.title("◈ TRACKLAB 38")
    st.caption("Derin takip · ön işleme · ölçüm")
    st.caption(f"Deney kaydı doğrulandı · {manifest['cache_key']}")
    st.divider()
    video = st.selectbox("Etiketli video", SEQUENCES)
    condition = st.radio("Görüntü koşulu", CONDITIONS)
    method = st.selectbox("Karşılaştırma yöntemi", METHODS[1:])
    st.divider()
    st.caption("Sarı: gerçek kutu · yeşil: takip tahmini. Başlangıç karesi ölçüme katılmaz; takip kaybolursa gerçek kutudan yeniden başlatılmaz.")

selected = summary.loc[(summary["video"] == video) & (summary["koşul"] == condition)].copy()
baseline = selected.loc[selected["yöntem"] == "İşlemsiz"].iloc[0]
candidate = selected.loc[selected["yöntem"] == method].iloc[0]
st.subheader(f"{video} · {condition} görüntü")
st.caption(f"{int(candidate['kare']):,} etiketli kare · aynı ViTTrack modeli · başlangıç kutusu ortak")
a, b, c, d = st.columns(4)
a.metric("Takip başarısı (AUC)", f"%{candidate['başarı_AUC'] * 100:.1f}",
         f"{(candidate['başarı_AUC'] - baseline['başarı_AUC']) * 100:+.1f} puan")
b.metric("Ortalama kutu örtüşmesi", f"%{candidate['ortalama_IoU'] * 100:.1f}",
         f"{(candidate['ortalama_IoU'] - baseline['ortalama_IoU']) * 100:+.1f} puan")
c.metric("20 piksel içinde merkez", f"%{candidate['merkez_20px'] * 100:.1f}")
d.metric("İşleme hızı", f"{candidate['FPS']:.0f} FPS")
st.info("Bu ölçümler üç örnek video ve iki yapay görüntü bozulmasına aittir; tüm videolarda aynı yönde iyileşme beklenmez.")
with st.expander("Kareler arası değişkenlik · eşleştirilmiş güven aralığı"):
    interval = paired_iou_interval(frames, video, condition, method)
    st.metric("Ortalama IoU farkı", f"{interval.difference * 100:+.1f} puan")
    st.caption(f"Keşifsel %95 aralık: {interval.lower * 100:+.1f} ile "
               f"{interval.upper * 100:+.1f} puan · {interval.frames} eş kare · "
               f"{interval.block} karelik ardışık bloklar, 1000 tekrar.")
    st.caption("İki yöntem aynı karelerde eşleştirildi. Ardışık kareler birbirine benzediği için "
               "bloklar yeniden örneklendi. Bu aralık ortalama IoU farkına aittir; "
               "AUC farkının aralığı değildir. Tek videodaki keşifsel değişkenliği gösterir, "
               "başka videolara genellenen bir başarı garantisi vermez.")

left, right = st.columns([1.1, .9], gap="large")
with left:
    st.markdown("**Kare kare kutu örtüşmesi**")
    view = frames.loc[(frames["video"] == video) & (frames["koşul"] == condition) &
                      (frames["yöntem"].isin(["İşlemsiz", method])) & frames["değerlendir"]]
    chart = view.pivot(index="kare", columns="yöntem", values="iou")
    st.line_chart(chart, height=300)
with right:
    st.markdown("**Aynı videoda yöntemler**")
    table = selected[["yöntem", "ortalama_IoU", "başarı_AUC", "IoU_0.5_başarı", "merkez_20px", "FPS"]]
    st.dataframe(table.style.format({key: "{:.1%}" for key in table.columns if key not in ("yöntem", "FPS")}
                                    | {"FPS": "{:.0f}"}), width="stretch", hide_index=True)

with st.expander("Başarı ve merkez doğruluğu eğrileri"):
    success_curve, center_curve = evaluation_curves(frames, video, condition, ("İşlemsiz", method))
    curve_left, curve_right = st.columns(2)
    with curve_left:
        st.markdown("**IoU başarı eğrisi**")
        st.line_chart(success_curve.set_index("IoU eşiği"), height=250)
        st.caption("Eğrinin altındaki alan, yukarıdaki başarı AUC değeridir.")
    with curve_right:
        st.markdown("**Merkez doğruluğu eğrisi**")
        st.line_chart(center_curve.set_index("Merkez eşiği (px)"), height=250)
        st.caption("20 pikseldeki değer, yukarıdaki merkez doğruluğudur.")

st.divider()
st.subheader("Kritik kare aralıkları")
analysis = analyze_pair(frames, video, condition, method)
robustness = summarize_robustness(frames)
first, second, third = st.columns(3)
first.metric("Yöntemin daha iyi olduğu kareler", f"%{analysis.better_share * 100:.1f}")
second.metric("En yüksek yerel IoU farkı", f"{analysis.best.value:+.2f}",
              f"{analysis.best.start}–{analysis.best.end}. kare")
third.metric("En düşük yerel IoU farkı", f"{analysis.worst.value:+.2f}",
             f"{analysis.worst.start}–{analysis.worst.end}. kare")
loss_parts = []
for label, interval in (("İşlemsiz", analysis.baseline_loss), (method, analysis.candidate_loss)):
    if interval is not None:
        loss_parts.append(f"{label}: {interval.start}–{interval.end}. kare ({int(interval.value)} kare)")
st.caption("En uzun ardışık düşük örtüşme (IoU < 0,2): " +
           (" · ".join(loss_parts) if loss_parts else "Bu eşiğin altında kare yok."))
st.caption(f"Yerel farklar {analysis.window} karelik kayan ortalamadır. "
           "Pozitif fark seçilen yöntemin daha iyi olduğu anlamına gelir.")
with st.expander("Düşük örtüşme ve toparlanma ayrıntıları"):
    detail = robustness.loc[(robustness["video"] == video) & (robustness["koşul"] == condition),
                            ["yöntem", "düşük_IoU_oranı", "düşük_dönem", "toparlanma",
                             "en_uzun_kare", "model_kayıp_bildirimi", "kutu_var_ama_düşük_IoU"]]
    st.dataframe(detail.style.format({"düşük_IoU_oranı": "{:.1%}"}),
                 width="stretch", hide_index=True)
    st.caption("Düşük örtüşme: IoU < 0,2. Toparlanma, düşük örtüşme döneminden sonra "
               "yeniden eşiğin üstüne çıkılmasıdır. Modelin 'kutu bulundu' bildirimi doğru hedefi "
               "izlediği anlamına gelmez; son sütun bu farkı gösterir.")
st.download_button("Bu koşulun sunum kartını indir (HTML)",
                   case_report_html(summary, frames, manifest, video, condition, method),
                   file_name=f"tracklab38_{video}_{condition}_{method}_sunum.html", mime="text/html")
st.caption("Sunum kartı çevrimdışı açılır; tarayıcıdaki Yazdır menüsünden PDF olarak kaydedilebilir.")

st.divider()
st.subheader("Yan yana karşılaştırma videosu")
st.caption("İstediğin aralığı seç. Takipçi yine ilk gerçek kutudan başlar; önceki kareler oynatılmadan hesaplanır.")
intervals = {"Başlangıç": 1,
             "En çok iyileşen bölüm": analysis.best.start,
             "En çok kötüleşen bölüm": analysis.worst.start}
if analysis.baseline_loss is not None:
    intervals["İşlemsiz yöntemin en uzun kaybı"] = max(1, analysis.baseline_loss.start - 10)
if analysis.candidate_loss is not None:
    intervals[f"{method} yönteminin en uzun kaybı"] = max(1, analysis.candidate_loss.start - 10)
segment = st.selectbox("İncelenecek bölüm", list(intervals))
start_frame = st.number_input("İlk etiketli kare", min_value=1, max_value=int(candidate["kare"]) + 1,
                              value=intervals[segment], step=1,
                              key=f"video_start_{video}_{condition}_{method}_{segment}")
length = st.slider("Gösterilecek kare sayısı", 30, 180, 120, step=30)
key = (video, condition, method, int(start_frame), length)
if st.button("Karşılaştırma videosunu üret", type="primary"):
    with st.spinner("İki takip koşulu aynı karelerde çalıştırılıyor…"):
        st.session_state["comparison_video"] = (key, render_comparison(
            video, condition, method, max_frames=length, start_frame=int(start_frame)))
if st.session_state.get("comparison_video", (None,))[0] == key:
    result = st.session_state["comparison_video"][1]
    st.video(result)
    st.download_button("MP4 indir", result,
                       file_name=f"tracklab_{video}_{condition}_{method}_{start_frame}.mp4", mime="video/mp4")
st.download_button("Bu iki yöntemin kare verilerini indir", view.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"tracklab_{video}_{condition}_{method}_kareler.csv", mime="text/csv")

st.divider()
st.subheader("Kendi videonla dene")
st.caption("Kısa bir MP4/MOV/AVI yükle, ilk karede tek hedef için bir kutu belirle. "
           "İki yöntem aynı başlangıç kutusundan izler; etiket olmadığı için IoU/AUC hesaplanmaz. "
           "İşlem bu bilgisayarda yapılır; ilk 20 saniye, en fazla 180 kare ve 30 MB.")
uploaded = st.file_uploader("Kendi videonu yükle", type=["mp4", "mov", "avi"],
                            max_upload_size=30, key="custom_tracking_video")
if uploaded:
    custom_data = uploaded.getvalue()
    custom_suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower()
    custom_id = sha256(custom_data).hexdigest()[:12]
    try:
        info = custom_preview(custom_data, custom_suffix)
    except (ValueError, OSError) as error:
        st.error(f"Video okunamadı: {error}")
    else:
        st.caption(f"{info.width} × {info.height} · {info.fps:.1f} FPS"
                   + (f" · {info.duration:.1f} sn" if info.duration else ""))
        default_width = min(160, max(8, info.width // 4))
        default_height = min(160, max(8, info.height // 4))
        left, top, wide, tall = st.columns(4)
        with left:
            x = st.number_input("Sol x", 0, info.width - 8,
                                max(0, info.width // 2 - default_width // 2),
                                key=f"custom_x_{custom_id}")
        with top:
            y = st.number_input("Üst y", 0, info.height - 8,
                                max(0, info.height // 2 - default_height // 2),
                                key=f"custom_y_{custom_id}")
        with wide:
            box_width = st.number_input("Kutu genişliği", 8, info.width - int(x),
                                        min(default_width, info.width - int(x)),
                                        key=f"custom_w_{custom_id}")
        with tall:
            box_height = st.number_input("Kutu yüksekliği", 8, info.height - int(y),
                                         min(default_height, info.height - int(y)),
                                         key=f"custom_h_{custom_id}")
        roi = (int(x), int(y), int(box_width), int(box_height))
        st.image(preview_roi(info.preview, roi), caption="İlk kare ve başlangıç hedef kutusu", width="stretch")
        option_a, option_b = st.columns(2)
        with option_a:
            custom_condition = st.selectbox("Video koşulu", CONDITIONS, key="custom_condition")
        with option_b:
            custom_method = st.selectbox("Ön işleme yöntemi", METHODS[1:], key="custom_method")
        custom_key = (custom_id, roi, custom_condition, custom_method)
        if st.button("Kendi videomda takibi çalıştır", type="primary"):
            try:
                with st.spinner("İki takipçi aynı karelerde çalıştırılıyor…"):
                    st.session_state["custom_result"] = (
                        custom_key, render_custom(custom_data, custom_suffix, roi,
                                                  custom_condition, custom_method))
            except Exception as error:
                st.error(f"Takip videosu üretilemedi: {error}")
        if st.session_state.get("custom_result", (None,))[0] == custom_key:
            custom_mp4, custom_stats = st.session_state["custom_result"][1]
            st.success(f"{custom_stats['kare']} kare · {custom_stats['fps']:.1f} çıktı FPS")
            st.caption("Takipçinin kutu vermediği kareler: işlemsiz "
                       f"{custom_stats['işlemsiz_kayıp']}, {custom_method} "
                       f"{custom_stats['yöntem_kayıp']}. Kutu vermesi doğru hedefi izlediğini kanıtlamaz.")
            st.video(custom_mp4)
            st.download_button("Kendi videomun MP4 karşılaştırmasını indir", custom_mp4,
                               file_name="tracklab_kendi_video.mp4", mime="video/mp4")

st.divider()
st.subheader("Tüm deneyler")
st.caption("3 video × 3 koşul × 3 yöntem = 27 deney. Başarı AUC, IoU eşiklerinin 0–1 aralığındaki ortalama başarısıdır.")
st.markdown("**Ön işlemenin kazanç haritası · işlemsize göre AUC puanı**")
gain = summary.pivot_table(index=["video", "koşul"], columns="yöntem", values="başarı_AUC")
gain["CLAHE farkı"] = (gain["CLAHE"] - gain["İşlemsiz"]) * 100
gain["Gamma farkı"] = (gain["Gamma 0.7"] - gain["İşlemsiz"]) * 100
gain["En yüksek AUC"] = gain[["İşlemsiz", "CLAHE", "Gamma 0.7"]].idxmax(axis=1)
gain_view = gain[["CLAHE farkı", "Gamma farkı", "En yüksek AUC"]].reset_index()
st.dataframe(gain_view.style.format({"CLAHE farkı": "{:+.1f}", "Gamma farkı": "{:+.1f}"}),
             width="stretch", hide_index=True)
st.dataframe(summary.style.format({column: "{:.3f}" for column in summary.columns if column not in
                                   ("video", "koşul", "yöntem", "kare")}), width="stretch", hide_index=True)
st.download_button("27 deneyin CSV tablosunu indir", summary_path.read_bytes(), file_name="tracklab38_sonuclar.csv", mime="text/csv")
with st.expander("27 koşulun düşük örtüşme analizi"):
    st.dataframe(robustness.style.format({"düşük_IoU_oranı": "{:.1%}"}),
                 width="stretch", hide_index=True)
    st.download_button("Sağlamlık CSV dosyasını indir", robustness.to_csv(index=False).encode("utf-8-sig"),
                       file_name="tracklab38_saglamlik.csv", mime="text/csv")
with st.expander("Yöntem ve veri kaynağı"):
    st.markdown("""
- Takipçi: OpenCV ViTTrack 2023sep. Hazır model tüm deneylerde aynıdır.
- CLAHE: LAB parlaklık kanalında `clipLimit=2`, `8×8` ızgara. Gamma: üs `0.7`.
- Karanlık koşul: tüm kanallarda yoğunluk `0.40` ile çarpılır.
- Bulanık koşul: tüm kareye `9×9` Gaussian bulanıklık, `σ=2.0` uygulanır.
- Gerçek kutular OpenCV'nin David, Dudek ve FaceOcc2 etiketlerinden gelir. David etiketleri 300. karede başlar.
- FPS yalnızca ön işleme ve takip güncellemesini kapsar; video çözme ve MP4 kodlama hariçtir.
""")
