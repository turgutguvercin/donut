# 200k Türkçe veri setiyle PP-OCRv6_medium_rec eğitimi

Hedef **PP-OCRv6_medium_rec**: PPLCNetV4 `medium` omurgası, LightSVTR CTC
boynu ve NRTR yardımcı başlığı. Mevcut 200.000 görüntü, etiket, kategori oranı,
kaynak ayrımı ve kalite bayrakları korunur. V5 ayarları da yerinde kalır.
V6 için görüntüleri yeniden üretmek veya indirmek gerekmez; aşağıdaki küçük
ek paketi mevcut v4 veri dizinine açın.

- [V6 eğitim ek paketi](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-v6-medium-adapter-v1/turkish-v6-medium-adapter-v1.zip)
- [V6 sürümü ve doğrulama raporları](https://github.com/turgutguvercin/donut/releases/tag/turkish-ocr-v6-medium-adapter-v1)
- [Asıl 200k görüntü paketi](https://github.com/turgutguvercin/donut/releases/tag/turkish-ocr-200k-v4)
- [200k görüntüleri indirme ve üretme kılavuzu](https://github.com/turgutguvercin/donut/blob/feat/synthdog-turkish/synthdog/docs/turkish-official-200k.md)

## Boyutlar ve uzun cümleler

| Ayar | Değer |
|---|---|
| Görüntü | BGR, 3 kanal, yükseklik 48 |
| Genişlik kovaları | 320, 640, 960, 1280, 1920, 2560, 4096 |
| CTC zaman adımları | 40, 80, 120, 160, 240, 320, 512 |
| Global / NRTR / kodlayıcı `max_text_length` | 256 |
| NRTR gerçek karakter kapasitesi | 254; verideki en uzun etiket 252 |
| Sözlük | 220 satır + kodlayıcının eklediği boşluk |
| CTC çıkış sınıfları | 222; blank dahil |

Görüntü en boy oranı korunur; gereken kova sağdan sıfırla doldurulur. Ardışık
aynı karakterlerin ek CTC adımı ihtiyacı da hesaba katılır. Yedi kova **aynı
eğitim turunda** kullanılır. Son eksik batch atılmaz; `ratio_list=[1.0]` olur.
Mevcut v4 kova listeleri gerçek V6 ağıyla kontrol edilerek tekrar kullanılır.

**Eğitimde aşağıdaki `train_paddle_tr_v6.py` başlatıcısını kullanın.** Sabitlenen
PaddleOCR kodundaki PPLCNetV4, eğitim sırasında her genişliği 40 adıma havuzlar;
çıkarımda genişliği korur. Bu nedenle yalnızca YAML'de 4096/256 yazmak yeterli
değildir. Başlatıcı, eğitimde de omurganın mevcut çıkarım havuzlamasını
(`avg_pool2d([3,2])`) kullanır. Öğrenilen katmanlar, ağırlık adları, BatchNorm
eğitim davranışı ve çıkarım yolu korunur. Uzun etiketli CTC kaybının sessizce
sıfıra düşmesini önleyen bu düzeltme, tüm genişliklerde geri yayılımla sınandı.
Doğrudan `tools/train.py` bu düzeltmeyi uygulamaz.

## Hangi listeyle eğitim yapılmalı?

| Ayrım | Tam veri | Önerilen | Zor/belirsiz |
|---|---:|---:|---:|
| Eğitim | 180.000 | 151.402 | 28.598 |
| Doğrulama | 10.000 | 8.391 | 1.609 |
| Test | 10.000 | 8.410 | 1.590 |

`rec_tr_v6_medium.yml` önerilen listeleri kullanır. `rec_tr_v6_medium_all.yml`
tam listeleri kullanır; kırpılma yüzünden bazı etiketler görüntüden tekil olarak
çözülemez. Bu örnekleri normal okunabilir örnekler gibi değerlendirmeyin.
Kalite filtresi sonrası kategori dağılımları, temel paketin raporlarında
ayrıca verilmiştir; filtre sonrası her kategorinin eşit olduğu iddia edilmez.

`rec_tr_v6_medium_fixed.yml`, önerilen bütün örnekleri 4096'ya dolduran daha
yavaş bir seçenektir. Bu dosya için de aynı başlatıcı gerekir.
Başlangıç batch boyutları sırasıyla 32/16/8/8/4/2/1'dir. GPU belleğine göre
`TurkishWidthBuckets.batch_sizes` değerlerini azaltın. Bütün kovaları tutun.
Yükleyici tek GPU veya tek CPU süreci içindir; çoklu GPU/DDP desteklenmez.

## Kurulum ve indirme

Referans PaddleOCR commit'i:
`dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`. Başlatıcı farklı veya yerel olarak
değiştirilmiş PaddleOCR eğitim kodunu kabul etmez. SynthDoG görüntü üretme
ortamıyla ayrı bir Python 3.11 ortamı kullanın.

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git PaddleOCR
cd PaddleOCR
git checkout dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf
```

Mevcut veri dizinini `PaddleOCR/train_data/turkish-recognition-200k-v4/` konumuna
yerleştirin veya buradan gerçek dizine sembolik bağlantı kurun. Henüz veriniz
yoksa PaddleOCR dizininde:

```bash
curl -fLO https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/download_dataset.py
python download_dataset.py --output train_data
```

Sonra yalnızca küçük ek paketi indirin:

```bash
curl -fLO https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-v6-medium-adapter-v1/turkish-v6-medium-adapter-v1.zip
curl -fLO https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-v6-medium-adapter-v1/SHA256SUMS
sha256sum -c SHA256SUMS
python -m zipfile -e turkish-v6-medium-adapter-v1.zip train_data/turkish-recognition-200k-v4
```

Çıktı `train_data/turkish-recognition-200k-v4/paddle-v6-medium/` olmalıdır.
V6 ek paketi tek başına görüntü içermez. İçindeki bütün komutlar, aşağıdaki
eğitim komutları gibi PaddleOCR dizininden çalıştırılır.

Bulutta doğrulanan CPU kurulumu:

```bash
python3.11 -m venv /workspace/paddle-v6
. /workspace/paddle-v6/bin/activate
python -m pip install paddlepaddle==3.2.2
python -m pip install -r train_data/turkish-recognition-200k-v4/paddle-v6-medium/requirements-paddle-v6.txt
python -m pip check
```

Colab'da GPU çalışma zamanı seçin. Ayrı Python 3.11 ortamı için `uv` ile
kurulum yapılabilir; shell hücresinde:

```bash
python -m pip install uv
uv python install 3.11.17
uv venv --python 3.11.17 /content/paddle-v6
uv pip install --python /content/paddle-v6/bin/python paddlepaddle-gpu==3.2.2 \
  --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
uv pip install --python /content/paddle-v6/bin/python \
  -r train_data/turkish-recognition-200k-v4/paddle-v6-medium/requirements-paddle-v6.txt
/content/paddle-v6/bin/python -c "import paddle; paddle.utils.run_check(); print(paddle.device.cuda.device_count())"
```

Bu GPU tekerleği CUDA 12.6 içindir; Colab sürücüsünün uyumluluğunu `nvidia-smi`
ve Paddle `run_check()` ile kontrol edin. CPU ve GPU Paddle paketlerini aynı
ortama birlikte kurmayın. Sonraki hücrelerde `python` yerine
`/content/paddle-v6/bin/python` yazın; `cd`/aktivasyon ayrı Colab shell hücreleri
arasında korunmaz. GPU bellek sınırı ve eğitim süresi bu CPU ortamında ölçülmedi.

## Ön eğitim ağırlıklarını Türkçe sözlüğe aktarın

```bash
python train_data/turkish-recognition-200k-v4/paddle-v6-medium/scripts/prepare_paddle_v6_weights.py \
  --paddleocr . \
  --dictionary train_data/turkish-recognition-200k-v4/paddle-v6-medium/turkish_dict.txt \
  --output pretrained/PP-OCRv6_medium_rec_tr
```

Araç resmî `PP-OCRv6_medium_rec_pretrained.pdparams` dosyasını indirir ve
SHA-256 doğrular. 437 özellik tensörü eksiksiz aktarılır. CTC/NRTR başlıkları
ve embedding, karakter sırasına göre yeniden eşlenir; boşluk ve özel token'lar
korunur. 217 ortak sembol, bütün Türkçe harfler dahil, ön eğitimden alınır.
Vikipedi metnindeki dört ek sembol (`ḥ`, `ṓ`, `ṭ`, `⁄`) deterministik olarak
başlatılır. Sözlüğü sonradan sıralamayın veya doğrudan eski başlığı yüklemeyin.

NRTR kodlayıcısında 225 sınıf vardır; sabitlenen resmî Transformer bir ek çıkış
sınıfı eklediği için başlığın boyutu 226'dır. Bu son satır ayrıca korunur.
`.transfer.json` sözlük/ağırlık özetlerini kaydeder; başlatıcı bunları kontrol eder.

## Kontrol ve eğitim

Önce tüm veri etiketlerini gerçek V6 operatörleriyle doğrulayın:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python train_data/turkish-recognition-200k-v4/paddle-v6-medium/scripts/check_paddle_tr_v6.py data \
  --paddleocr . --dataset train_data/turkish-recognition-200k-v4 \
  --output native-data-audit.json

python train_data/turkish-recognition-200k-v4/paddle-v6-medium/scripts/train_paddle_tr_v6.py \
  -c train_data/turkish-recognition-200k-v4/paddle-v6-medium/rec_tr_v6_medium.yml
```

CPU'da kısa deneme için eğitim komutuna `-o Global.use_gpu=False` ekleyin.
Tam veri seçeneği için YAML dosyasını `rec_tr_v6_medium_all.yml` ile değiştirin.
Önerilen eğitim 30 epoch ve `1e-4` öğrenme hızıyla başlar; gerçek belge doğrulama
sonucuna göre ayarlayın. Ek RecAug/RecConAug varsayılan olarak kapalıdır.
Eğitime devam etmek için `-o Global.checkpoints=./output/PP-OCRv6_medium_rec_tr/latest`
kullanın; optimizer durumu da yüklenir. Google Drive'a model kaydetmek için
`Global.save_model_dir` yolunu değiştirin; görüntüleri Colab yerel diskinde tutmak
çok sayıda küçük dosyanın okunmasını hızlandırır.

Model seçimini `val-recommended.txt` ile yapın. Testi model seçimi için
kullanmayın. Eğitim bittikten sonra önerilen test ayrımını resmî değerlendiriciyle
ölçebilirsiniz; bu komut daha yavaş sabit 4096 ön işlemeyi kullanır:

```bash
python tools/eval.py \
  -c train_data/turkish-recognition-200k-v4/paddle-v6-medium/rec_tr_v6_medium_fixed.yml \
  -o Global.checkpoints=./output/PP-OCRv6_medium_rec_tr/best_accuracy \
     'Eval.dataset.label_file_list=[./train_data/turkish-recognition-200k-v4/test-recommended.txt]'
```

Zor örnekleri `test-challenge.txt` ile ayrı raporlayın. Değerlendirme ayarı
boşlukları, büyük/küçük harfi ve noktalama işaretlerini silmez.

## Dışa aktarma ve belge üzerinde kullanım

Eğitim tamamlandıktan sonra:

```bash
python tools/export_model.py \
  -c train_data/turkish-recognition-200k-v4/paddle-v6-medium/rec_tr_v6_medium.yml \
  -o Global.pretrained_model=./output/PP-OCRv6_medium_rec_tr/best_accuracy \
     Global.save_inference_dir=./inference/PP-OCRv6_medium_rec_tr

python tools/infer/predict_rec.py \
  --image_dir ./document_line_crops \
  --rec_model_dir ./inference/PP-OCRv6_medium_rec_tr \
  --rec_algorithm SVTR_LCNet --rec_image_shape '3,48,4096' --rec_batch_num 1 \
  --rec_char_dict_path ./train_data/turkish-recognition-200k-v4/paddle-v6-medium/turkish_dict.txt \
  --use_space_char True
```

Bu veri **satır tanıma** verisidir. Belge hattında tespit modelinizin ürettiği
kesitleri tanıyıcıya verin; algılayıcının kesip attığı nokta/aksanları geri
getirmez. Bu paket tespit modelini yeniden eğitmez. Çıkarımda yüksekliği 48'e
orantılı ölçekleyin ve uzun satırları 320'ye sıkıştırmayın. 4096'yı aşan gerçek
kesitleri ayrıca ele alın; bu genişlikler burada doğrulanmadı.

## Bulutta yapılan doğrulama ve kapsam

Ek paketteki raporlar; tüm 200.000 görüntünün gerçek DecodeImage,
MultiLabelEncode, RecResizeImg ve KeepKeys işlemlerinden geçmesini, CTC/NRTR
UTF-8 etiketlerinin birebir geri çözülmesini ve tüm kovaların iki turda da
eksiksiz örneklenmesini kaydeder. Gerçek ön eğitim ağırlıkları yüklenerek yedi
genişlikte CTC+NRTR geri yayılım/Adam adımları yapılmıştır. En uzun etiket
252 karakterdir. Resmî eğitim başlatıcısı da kısa bir eğitim/değerlendirme ve
checkpoint kaydetme denemesiyle kontrol edildi. Depodaki **98 test geçti**.
Kısa denemenin modeli dışa aktarılıp gerçek çıkarım aracıyla da çalıştırıldı.

Bunlar eğitim hattının çalıştığını doğrular; 200k üzerinde tamamlanmış bir
eğitim veya gerçek belgeler için doğruluk sonucu değildir. GPU performansı
ayrıca ölçülmelidir. Sentetik görüntüler CC BY-SA 4.0 kapsamındaki kaynak ve
font atıflarıyla kullanılmalıdır. Yardımcı kod MIT; Paddle'dan türetilen ayarlar
ve havuzlama uyarlaması Apache-2.0 lisanslıdır. Ham korpus, font ikilileri ve
model ağırlıkları bu ek pakette dağıtılmaz.
