# Türkçe OCR: 200k resmî belge ve genel metin kesiti

Bu sürüm, v3'teki 100.000 görüntüyü ve veri ayrımlarını koruyup 100.000 yeni
resmî belge tarzı **sentetik satır kesiti** ekler. Toplam ayrım 180.000 eğitim,
10.000 doğrulama ve 10.000 test görüntüsüdür. Tam sayfa tespit verisi değildir.
Elle doğrulanmış gerçek belge kesitleri bu sürüme dahil değildir.

## Metin ve görsel dağılımı

Yeni bölümde aşağıdaki on sınıfın her birine 10.000 görüntü ayrılır:

- Kurum başlığı ve imza/unvan satırı.
- Konu, ek, dağıtım ve ilgi alanları.
- Kısa resmî yazışma cümleleri.
- Telefon, faks, dahili numara, adres, e-posta ve KEP alt bilgileri.
- 150–252 karakterlik uzun resmî yazışma satırları.
- Elektronik imza açıklaması, URL ve belge doğrulama kodları.
- Kelime ve cümle içinde karışan karakterler.
- Tarih, sayı ve belge referansları.
- Türkçe binlik/ondalık ayırıcılarıyla tutar ve miktarlar.
- Kod alanlarında karışan karakterler.

Yeni bölümde Times New Roman ve Arial'den 20.000'er görüntü bulunur.
Diğer 18 aile kalan 60.000 görüntüyü paylaşır; ailelerin mevcut gerçek
normal/kalın/italik/kalın-italik dosyaları dengeli kullanılır. Aynı satırda
karışık stiller de vardır. Font dosyaları pakete konulmaz.

13 hedef grup: l/1/I/İ/ı/i, O/0, S/5, B/8, rn/m, cl/d, ş/s/Ş/S,
ğ/g/Ğ/G, ü/u/Ü/U, ç/c/Ç/C, ö/o/Ö/O, 2021/202I ve İl/İ1.
Kelime ve kod örneklerinde grup sayıları dengelenir; doğal cümlelerde her
karakterin toplam frekansının eşit olduğu iddia edilmez.

V3'ün 400 Vikipedi makalesi, 20 konusu, diğer fontları, eğrileri, perspektifleri,
ışık/gölge koşulları, arka planları, baskı kusurları ve kırpma kategorileri korunur.
Yeni resmî bölümde beyaz kâğıt, düz satır ve olağan pozlama daha sık kullanılır.
Denge, açıkça belirtilmiş kategori oranlarıdır; tüm kombinasyonların eşit sayıda
üretilmesi değildir. Kesin sayılar category-counts.json, official-balance.json
ve quality-report.json dosyalarındadır.

## Küçük yazı ve okunabilirlik

Yeni 30.000 örnek, ayrı bir fiziksel yakalama sürecinden geçer:
alan ortalamasıyla küçültme → optik bulanıklık → tarama gürültüsü →
JPEG sıkıştırması → bicubic büyütme. 5.000 small_clear, 10.000 small_scan,
10.000 small_jpeg ve 5.000 photocopy_scan örneği vardır.

Küçültme oranı, üreticinin dönüşümlerden sonraki harf yüksekliği tahminine
bağlıdır. Eğri satırın toplam mürekkep kutusunu harf yüksekliği saymak,
harfleri gereğinden fazla küçültebildiği için kullanılmaz. Metadata; yakalama
çözünürlüğünü, harf yüksekliğini, ölçülen mürekkep sınırlarını, bulanıklık ve
JPEG parametrelerini kaydeder. Punto ile piksel yüksekliği aynı şey değildir.

Yeni bölümde tam karakter bileşeni kaybettiren ağır kırpma üretilmez.
V3'teki zor örnekler korunur ve kalite bayrakları taşır:

- train.txt / val.txt / test.txt: bütün örnekler.
- train-recommended.txt ve karşılıkları: standart eğitim için önerilen filtre.
- train-challenge.txt ve karşılıkları: belirsiz veya fazla bozulmuş örnekler.
- Her grubun ayrıca yedi genişlik kovası listesi vardır.

Filtre; kırpılmış bileşenleri, çok küçük harfleri, küçük yazıda ağır blur/soluk
mürekkebi ve yüksek mürekkep kaybını ayırır. Bu otomatik ölçümler, bütün
görüntülerin insan tarafından okunabilirliğinin doğrulandığı anlamına gelmez.
Filtre sonrası kategori dağılımları ayrıca raporlanır.

## Resmî belge referansı

[Resmî Yazışmalar Yönetmeliği](https://www.mevzuat.gov.tr/MevzuatMetin/21.5.2646.pdf)
incelenmiştir. Madde 7, Times New Roman 12 veya Arial 11 puntoyu; gerektiğinde
gövde için 9, iletişim için 8 puntoyu belirtir. Başlık, tarih/sayı, imza,
doğrulama ve iletişim alanları da ilgili maddelerden kontrol edilmiştir.
URL, indirilen dosyanın SHA-256 özeti ve gözlemler
official_document_references.json dosyasındadır.

Resmî PDF'nin sayfaları veya gerçek kişilere ait belgeler eğitim verisi olarak
kopyalanmaz. Metinler özgün şablonlardan üretilir; adlar ve numaralar sentetiktir,
e-posta/URL alanlarında ayrılmış example.org alan adı kullanılır. Belge biçimi
referansı, gerçek tarama dağılımının ölçüldüğü anlamına gelmez.

## İndirme

[200k v4 sürümü](https://github.com/turgutguvercin/donut/releases/tag/turkish-ocr-200k-v4)
ve [indirme aracı](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/download_dataset.py):

```bash
python download_dataset.py \
  --manifest https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/download-manifest.json \
  --output turkish-ocr-v4
```

Araç tüm ZIP parçalarını SHA-256/CRC ile doğrular, büyük metadata dosyasını
parçalarından yeniden birleştirir ve görüntü/etiket sayısını kontrol eder.
Çıktı: turkish-ocr-v4/turkish-recognition-200k-v4/.
Tarayıcıdan bütün parçaları içeren etiket ZIP'ini indirirseniz, dış ZIP'i açtıktan
sonra içindeki download_dataset.py aracını --manifest download-manifest.json
--parts-dir . --output . seçenekleriyle çalıştırın. Metadata birleştirme
adımı nedeniyle yalnızca iç ZIP'leri açmak yeterli değildir.

## Yeniden üretme

Önce mevcut Türkçe SynthDoG kurulumunu, lisanslı font kataloğunu ve v3 veri setini
hazırlayın. V3 makale manifestiyle aynı korpusu yeniden oluşturmak için
[v3 kılavuzunu](turkish-balanced-vocabulary.md) kullanın. Komutlar synthdog/
dizininden çalıştırılır:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/expand_tr_official.py \
  --base-dataset /data/turkish-recognition-100k-v3 \
  --corpus /data/corpus-tr-v3/trwiki.txt \
  --font-manifest /data/fonts/recognition-fonts.json \
  --output /data/turkish-recognition-200k-v4 \
  --additional-count 100000 --seed 20261016 --workers 4

python scripts/finalize_tr_v5_buckets.py \
  --dataset /data/turkish-recognition-200k-v4 \
  --shape-report /data/reference/model-shape-audit.json

python scripts/prepare_paddle_tr_v5.py \
  --dataset /data/turkish-recognition-200k-v4 \
  --data-dir ./train_data/turkish-recognition-200k-v4
```

Aynı kaynaklar, font dosyaları, bağımlılıklar ve tohum gereklidir.
Çalışan sayısı görüntüleri değiştirmez. Eski v3 görüntüleri yeniden çizilmez.
V5 sonlandırma adımı, mevcut görüntü ve etiketleri değiştirmeden genişlik
kovalarını düzeltir. model-shape-audit.json yayımlanan pakette bulunur.
Bu rapor, gerçek v5 server ağının CPU çalıştırmasıyla ölçülmüştür. Yeniden ölçmek
için ayrı bir Python 3.11 ortamına paddlepaddle==3.2.2, PyYAML, einops,
packaging, requests ve tqdm kurup aşağıdaki komutu çalıştırın:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/check_paddle_v5_shapes.py \
  --paddleocr /data/PaddleOCR \
  --output /data/reference/model-shape-audit.json
```

PaddleOCR checkout'u aşağıdaki sabit commit'te olmalıdır. Bu kontrol rastgele
ağırlıklarla yalnızca çıktı boyutlarını ölçer; doğruluk/eğitim deneyi değildir.

## PP-OCRv5 ve uzun satırlar

PaddleOCR referansı: dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf.
paddle-v5/ içindeki ayarlar gerçek PP-OCRv5_server_rec mimarisini kullanır.
Global.max_text_length, NRTRHead.max_text_length ve etiket kodlayıcıları 256'dır.
Başlangıç/bitiş işaretleri nedeniyle gerçek karakter kapasitesi 254'tür.
Resmî varsayılan 25 ayarı 23 gerçek karakter kabul eder.

Görüntüler 48 piksel yüksekliğinde, en fazla 4096 piksel genişliğindedir.
Kovalar: 320, 640, 960, 1280, 1920, 2560, 4096. Yatay sıkıştırma yapılmaz.
CTC kapasitesi, ardışık tekrarlar için gerekli boş zaman adımları dahil kontrol
edilir. Etiketi keserek veya uzun satırı 320 piksele sıkarak çözüm üretilmez.
Gerçek PP-OCRv5 server çıktısı, kova genişlikleri için sırasıyla
40/80/120/160/240/320/512 zaman adımıdır: genişlik oranı **8**'dir.
V3'ün v4 tabanlı oran-4 varsayımı v5 için kullanılmaz. Gerekli satırlar daha geniş
kovaya taşınır; görüntünün yatay ölçeği değişmez.

rec_tr_recommended.yml bütün önerilen örnekleri 4096 genişliğe sağdan doldurur;
batch=1 bir başlangıç ayarıdır, GPU belleğine sığma garantisi değildir.
rec_tr_all.yml zorluk örneklerini de içerir. rec_tr_w*.yml dosyalarının her biri
yalnızca kendi kovasını kullanır; otomatik ortak eğitim yapmaz. Verimli Colab
eğitimi için bütün kovaları kapsayan bir sampler gerekir.

Yeni Türkçe sözlükle eski çıkış başlığını kontrolsüz yüklemeyin. Ortak karakter
ağırlıkları doğru eşlemeyle aktarılmalı; etkilenen CTC/NRTR sınıf ve embedding
katmanları uyarlanmalıdır. pretrained_model bu nedenle otomatik seçilmez.
RecAug zaten blur içerir; bu sette ölçülmüş bozulmalar bulunduğu için ek RecAug
ve RecConAug varsayılan olarak kapalıdır.

Vikipedi makaleleri ve tam normalize metinler veri ayrımları arasında korunur.
Sentetik şablon aileleri ise ortak olabilir. V3 ve v4 aynı eski örnekleri
içerdiğinden, sürümleri bağımsız veri setleriymiş gibi birleştirmeyin.
Tespit/kırpma başarısını ve gerçek OCR doğruluğunu ayrıca gerçek belgelerde ölçün.

Görüntü/etiket paketi CC BY-SA 4.0; Vikipedi kaynak manifesti ve font bildirimleri
korunmalıdır. Üretici kodu MIT, Paddle türevi ayarlar Apache-2.0 lisanslıdır.
Ham Vikipedi korpusu, resmî PDF ve font ikilileri yayımlanmaz.
