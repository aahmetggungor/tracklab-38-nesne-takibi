# TRACKLAB 38

## Çevrimiçi yayın

**Canlı demo:** https://tracklab-38-nesne-takibi.streamlit.app/  
**Kaynak kod:** https://github.com/aahmetggungor/tracklab-38-nesne-takibi

Streamlit Community Cloud için giriş dosyası `app.py`, Python sürümü 3.12'dir. Demo model/video dosyaları ve 27 etiketli deneyin doğrulanmış sonuçları depoda bulunur. Deneyin kaydedildiği paket sürümleri manifestte tutulur; sunucu sürümü farklı olsa da model, veri ve deney kodu SHA-256 değerleri ile sonuç tabloları kontrol edilir. Kullanıcının yüklediği video sunucuda işlenir; gerçek kişisel veya gizli video yüklemeyin.

SAYZEK PDF 38 için hazırlanmış ölçülebilir nesne takibi demosu. Aynı derin öğrenme tabanlı ViTTrack modeli, üç etiketli videoda üç görüntü ön işleme yöntemi ve üç görüntü koşulunda denenir. Ön işleme ile başarı/hız değişimi ve kritik kare aralıkları arayüzde görülebilir. Kısa bulgular [KISA_RAPOR.md](KISA_RAPOR.md) dosyasındadır.

## Hemen aç

Bu bilgisayarda örnek veri ve 27 deneyin sonuçları hazırdır. Proje klasöründen:

```powershell
.\start_pdf38.ps1
```

Uygulama `http://127.0.0.1:8503/` adresinde açılır. Örnek video, koşul ve ön işleme yöntemi seç; tam video ölçümlerini, IoU başarı/merkez doğruluğu eğrilerini, düşük örtüşme dönemlerini ve en fazla iyileşen/kötüleşen kare aralıklarını incele. Başlangıç karesini seçip **Karşılaştırma videosunu üret** ile 30–180 karelik bölümü yan yana izle ve MP4 indir. Takip, seçilen bölümden önce de ilk gerçek kutudan itibaren çalıştırılır. Karşılaştırma panelleri videonun özgün en boy oranını korur; kutular yerleştirilen görüntüye göre çizilir. İki hazır örnek: [David/orijinal, CLAHE kazancı](results/demo_david_clahe.mp4) (263–352. etiketli kare) ve [Dudek/bulanık, CLAHE'nin gerilediği bölüm](results/demo_dudek_bulanik_kayip.mp4) (269–358. etiketli kare).

Seçili karşılaştırmadaki **Kareler arası değişkenlik** bölümünde ortalama IoU farkı ve keşifsel `%95` aralık vardır. Aynı kareler eşleştirilir; zamansal bağımlılığı kısmen korumak için 60 karelik ardışık bloklarla 1000 tekrar yapılır. Aralık yalnızca bu videodaki ortalama IoU farkı içindir; AUC aralığı veya farklı videolara başarı garantisi olarak yorumlanmamalıdır.

**Bu koşulun sunum kartını indir (HTML)** düğmesi, seçilen etiketli video/koşul için üç yöntemin ölçümlerini, düşük IoU özetini, kritik 60 karelik aralıkları, keşifsel IoU farkı aralığını ve model/deney parmak izini tek bir çevrimdışı dosyada toplar. HTML dosyasını tarayıcıda açıp **Yazdır → PDF olarak kaydet** ile sunuma ekleyebilirsin. Kart yalnızca kaydedilmiş etiketli deneylerden üretilir; yüklenen kullanıcı videosuna doğruluk atamaz.

**Kendi videonla dene:** Arayüzün alt bölümüne en fazla 30 MB MP4/MOV/AVI yükle; ilk karede hedef kutusunun x, y, genişlik ve yüksekliğini gir. Önizlemede kutuyu kontrol et ve yan yana takip MP4'ü üret. İlk 20 saniyeden en fazla 180 kare, en fazla 12 çıktı FPS ile işlenir; ses alınmaz. Video yerel sunucuda işlenir. Etiket olmadığından bu akışta IoU/AUC gösterilmez; kutunun görünmesi hedefi doğru izlediği anlamına gelmez.

## Başka bilgisayarda kur

Python 3.12 sanal ortamı oluşturup bağımlılıkları kurun:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_assets.py
.\.venv\Scripts\python.exe run_experiments.py
.\.venv\Scripts\python.exe verify.py
.\.venv\Scripts\python.exe verify_video.py
.\.venv\Scripts\python.exe verify_custom.py
.\start_pdf38.ps1
```

İlk komutta bağımsız ortam oluşur. Bu bilgisayarda kök proje `.venv` ortamı da aynı gerekli paketleri içerir; başlatıcı yerel ortam yoksa onu kullanır. Model ve örnek videolar toplam yaklaşık 15 MB indirilir. Deneyler CPU'da çalışır; RTX 4060 şart değildir.

## Deney tanımı

| Değişken | Seçenekler |
| --- | --- |
| Video | David, Dudek, FaceOcc2 |
| Koşul | Orijinal, tüm piksel yoğunluğu × 0,40, Gaussian bulanıklık (9×9, σ=2,0) |
| Ön işleme | Yok, LAB parlaklık kanalında CLAHE, gamma 0,7 |
| Takipçi | Her deneyde aynı OpenCV ViTTrack 2023sep |
| Ölçümler | Ortalama IoU, başarı eğrisi AUC, IoU≥0,5, 20 piksel merkez doğruluğu, işlem FPS |

Her deney gerçek etiketin ilk kutusundan başlar; başlatma karesi ölçülmez. Sonraki karelerde gerçek kutuya bakılarak yeniden başlatma yapılmaz. David videosunda etiketler 300. kareden itibaren başladığı için video buna göre hizalanır. FPS yalnızca ön işleme ve takip adımlarını kapsar; video çözme/kodlama hariçtir.

`results/summary.csv` 27 deneyin tablosu; `results/frame_metrics.parquet` kare düzeyindeki sonuçlar; `results/manifest.json` yazılım sürümleri ile model, video, etiket ve deney kodu SHA-256 değerleri. `verify.py` bu parmak izlerini ve hesaplanan metrikleri kontrol eder. `failure_analysis.py` 60 karelik kayan pencerede en iyi/en kötü IoU farklarını; 27 koşul için IoU < 0,2 oranını, en uzun düşük örtüşme dönemini, toparlanmayı ve modelin kendi kayıp bildirimini hesaplar. Sağlamlık tablosu arayüzden CSV olarak indirilebilir. “Kutu bulundu” bildirimi gerçek hedefle örtüşme garantisi değildir. Yayın için gerekli model/veri ve sonuçlar GitHub'a eklenir; geçici ara sonuçlar hariç tutulur.

Arayüz açılırken `result_integrity.py` mevcut model/veri/deney kodu parmak izini kayıtla karşılaştırır, 27 koşulun tamlığını ve özet metriklerin kare sonuçlarından yeniden hesaplanan değerlerle uyumunu denetler. Kaydın kendi sürüm parmak izi ayrıca doğrulanır. Sunucunun Python/paket sürümleri deneyinkinden farklıysa kayıtlı ölçümler açıkça belirtilir. Dosyalar değişmiş veya sonuçlar tutarsızsa grafik ve sunum kartı yerine yeniden çalıştırma talimatı gösterilir. Bu kontrol, eski bir deneyin yeni model ya da veriyle yapılmış gibi sunulmasını önler.

Deney yarıda kesilirse aynı `python run_experiments.py` komutunu yeniden çalıştırın. Tamamlanan koşullar `results/checkpoints/<parmak-izi>/` içinden okunur, eksikler hesaplanır. Parmak izi model, veri, sürüm ve deney koduna bağlıdır; bunlar değişirse eski ara sonuçlar kullanılmaz. İşlem süresini baştan ölçmek için `python run_experiments.py --force` kullanın. Sonuç tabloları geçici dosyadan tamamlanmış dosyaya taşınarak yazılır.

## Kaynak ve sınırlar

- Takipçi: [OpenCV Zoo ViTTrack](https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack), Apache 2.0.
- Videolar ve etiketler: [OpenCV Extra takip test verileri](https://github.com/opencv/opencv_extra/tree/4.x/testdata/cv/tracking), orijinal OTB çalışması: Wu, Lim ve Yang, *Online Object Tracking: A Benchmark*, CVPR 2013.
- SHA-256 değerleri `download_assets.py` içindedir; yeniden indirme bu değerlerle denetlenir.

Üç video ve iki yapay bozulma, genel görüntü takip başarımını kanıtlamaz. Bazı yöntemler videoya göre faydalı veya zararlı olabilir. Örneğin gamma FaceOcc2'de başarıyı düşürdü. Kullanıcı videosunda gerçek konum etiketi olmadığı sürece ölçülmüş doğruluk gösterilmez.
