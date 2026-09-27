# PDF 38 kısa deney raporu

**Araştırma sorusu:** Görüntüye uygulanan basit ön işleme, aynı derin öğrenme tabanlı takipçinin başarısını ve hızını nasıl etkiler?

Üç etiketli videoda ViTTrack değişmeden tutuldu. İşlemsiz, CLAHE ve gamma 0,7 yöntemleri orijinal, yoğunluğu `%40`'a düşürülmüş ve Gaussian bulanıklık uygulanmış görüntüde çalıştırıldı. Her koşul aynı gerçek kutudan başlatıldı; ilk kare ölçülmedi ve kayıp hedef gerçek konumdan yeniden başlatılmadı. Toplam **27 deneyde** başarı AUC, IoU eşiği boyunca ölçüldü. Tüm sayılar `results/summary.csv` içindeki kayıttan gelir.

| Video / koşul | İşlemsiz AUC | CLAHE AUC | Gamma AUC |
| --- | ---: | ---: | ---: |
| David / orijinal | %17,9 | **%62,7** | %40,7 |
| David / karanlık | %8,8 | **%18,5** | %14,3 |
| David / bulanık | %17,1 | **%35,8** | %18,0 |
| Dudek / orijinal | %69,6 | **%72,7** | %71,9 |
| Dudek / karanlık | %32,0 | **%67,4** | %64,4 |
| Dudek / bulanık | %69,6 | %69,4 | **%69,8** |
| FaceOcc2 / orijinal | %49,3 | **%52,8** | %20,5 |
| FaceOcc2 / karanlık | %17,3 | **%43,8** | %14,6 |
| FaceOcc2 / bulanık | %49,7 | **%54,9** | %20,1 |

Üç videonun eşit ağırlıklı ortalaması, orijinal görüntüde işlemsiz `%45,6` ve CLAHE `%62,8`; karanlık görüntüde `%19,4` ve `%43,3`; bulanık görüntüde `%45,5` ve `%53,3` oldu. CLAHE dokuz koşulun sekizinde AUC'yi artırdı. **Bulanık Dudek'te CLAHE `%69,6 → %69,4` ile hafif geriledi**; gamma `%69,8` ile en yüksek değeri aldı. Gamma ise FaceOcc2'de belirgin düşüş yarattı. Yöntem seçimi görüntü koşuluna bağlıdır.

Kare düzeyi analiz, David/orijinal görüntüde işlemsiz yöntemin `143–471` arası 329 kare üst üste `IoU < 0,2` olduğunu; CLAHE'de en uzun böyle kesintinin iki kare olduğunu gösterdi. CLAHE bu videonun ölçülen karelerinin `%98,1`'inde daha yüksek IoU verdi. Bulanık Dudek'te ise CLAHE karelerin yalnız `%41,3`'ünde daha iyiydi; `279–338` arası 60 karelik bölümde ortalama IoU farkı `−0,051` oldu. Arayüz bu bölümlerin gerçek başlangıçtan yürütülen yan yana videosunu üretir.

Sağlamlık analizinde David/orijinal işlemsiz akışının 470 ölçülen karesinden 335'inde IoU `< 0,2` çıktı (`%71,3`). Takipçinin `update()` çağrısı bu karelerin hiçbirinde “hedef bulunamadı” demedi; bir kutu döndürse de hedefi yanlış yerde izleyebildi. CLAHE'de düşük IoU'lu kare sayısı 7 (`%1,5`). Arayüz, düşük örtüşme oranı, en uzun dönem, toparlanma ve modelin kendi kayıp bildirimini 27 koşul için ayrı gösterir.

**Kareler arası değişkenlik:** Aynı karelerin IoU farkı, 60 ardışık karelik bloklar 1000 kez yeniden örneklenerek keşifsel `%95` aralıkla gösterilir (sabit rastgele tohum). David/orijinal CLAHE−işlemsiz farkı ortalama `+44,9` IoU puanı; aralık yaklaşık `+35,6` ile `+56,5` puan. Bulanık Dudek'te fark `−0,2` puan; aralık yaklaşık `−1,6` ile `+1,4` puan ve sıfırı kapsıyor. Bu aralıklar **ortalama IoU farkına** aittir; AUC farkı için hesaplanmadı. Tek videonun zaman içi değişkenliğini keşifsel gösterir, başka videolara genellenmez ve istatistiksel anlamlılık iddiası değildir.

Hazır sunum klipleri: `results/demo_david_clahe.mp4` kazanç örneği (263–352. etiketli kare), `results/demo_dudek_bulanik_kayip.mp4` karşı örnek (269–358. etiketli kare). İkisinde de takipçi 1. karede başlatılıp seçilen bölüme kadar yürütülmüştür. Yan yana paneller kaynak en boy oranını korur; görüntü ve gerçek/tahmini kutular aynı ölçek ve boşlukla yerleştirilir.

Arayüzdeki **sunum kartı** indirmesi, seçilen etiketli koşulun üç yöntemlik ölçüm tablosunu, sağlamlık ve kareler arası değişkenlik bulgularını, yöntem sınırlarını ve deney parmak izini yazdırılabilir HTML olarak verir. Tarayıcıdan PDF'ye dönüştürülebilir.

Uygulama ayrıca kullanıcı videosunda elle seçilen ilk hedef kutusuyla iki akışı yan yana üretir. Böyle bir videoda gerçek konum etiketi olmadığından AUC, IoU veya doğruluk yüzdesi hesaplanmaz. Kaydedilmiş deneylerin başarı ve merkez doğruluğu eğrileri arayüzde ayrıca incelenebilir.

Hız ölçümü ön işleme ve takip güncellemesini kapsar. Son tam çalıştırmada CLAHE, karanlık Dudek videosunda yaklaşık `182 FPS`, işlemsiz yöntem `205 FPS` üretti. Bu rakamlar kullanıcının toplam video oynatma hızı değildir; çözme, arayüz ve kodlama maliyetleri hariçtir. Aynı bilgisayar ve yazılım sürümleri `results/manifest.json` ile kayıtlıdır; FPS tekrar çalıştırmalarda değişebilir. Yarıda kesilen deneyler, doğrulanmış ara koşul dosyalarından sürdürülür.

**Sınırlar:** Üç video, tek hazır takipçi ve iki yapay bozulma kullanıldı; sonuçlar büyük bir benchmark veya gerçek düşük ışık/bulanıklık çekimi yerine geçmez. Hazır ViTTrack modeli yeniden eğitilmedi. Yan yana videoda görünen IoU yalnızca etiketli örneklerde hesaplanabilir. Akademik genişletme için daha çok video, farklı takipçiler ve ayrı tekrarlar gerekir.
