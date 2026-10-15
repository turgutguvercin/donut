"""Original synthetic Turkish administrative text; no real personal records.

Field conventions follow the public official-correspondence regulation. Values,
names, identifiers and combinations below are generated, not copied documents.
The reserved example.org domain makes contact/verification examples non-live.
"""
import hashlib
import unicodedata

import generate_tr_dataset as dataset
import generate_tr_recognition as base

KINDS = {
    "ascii_word": "official_heading",
    "turkish_word": "official_subject",
    "short_line": "official_body_short",
    "medium_line": "official_footer_contact",
    "long_sentence": "official_body_long",
    "extra_long_sentence": "official_verification",
    "confusables": "official_confusables_words",
    "date": "official_date_reference",
    "price": "official_amount",
    "synthetic_id": "official_confusables_codes",
}
GROUPS = {
    "l_1_I_dotted_dotless": ("l", "1", "I", "İ", "ı", "i"),
    "O_0": ("O", "0"), "S_5": ("S", "5"), "B_8": ("B", "8"),
    "rn_m": ("rn", "m"), "cl_d": ("cl", "d"),
    "s_cedilla": ("ş", "s", "Ş", "S"),
    "g_breve": ("ğ", "g", "Ğ", "G"),
    "u_umlaut": ("ü", "u", "Ü", "U"),
    "c_cedilla": ("ç", "c", "Ç", "C"),
    "o_umlaut": ("ö", "o", "Ö", "O"),
    "year_2021": ("2021", "202I"),
    "province_Il": ("İl", "İ1"),
}
WORDS = {
    "l_1_I_dotted_dotless": "İl ve ilçe birimleri ılık ışık altında bilgi girişini inceledi.",
    "O_0": "Okul Onarım Birimi, 0 numaralı örnek kaydı kontrol etti.",
    "S_5": "Sosyal hizmet servisinin 5 sayfalık dosyası sunuldu.",
    "B_8": "Birim Başkanlığı, 8 ekin bilgisini bildirdi.",
    "rn_m": "Modern arşiv düzeninde rn ve m kodları ayrı tutulur.",
    "cl_d": "Belge sınıflamasında cl ve d kodları ayrı değerlendirilir.",
    "s_cedilla": "Şube sorumlusu iş, sayı ve sıra bilgilerini karşılaştırdı.",
    "g_breve": "Görevli, eğitim ve değerlendirme çizelgesini güncelledi.",
    "u_umlaut": "Üst kurul, kurum müdürlüğünün duyurusunu uygun buldu.",
    "c_cedilla": "Çalışma çizelgesi, çevre ve inceleme birimine gönderildi.",
    "o_umlaut": "Ön ödeme, özel ödenek ve onay dosyaları görüşüldü.",
    "year_2021": "2021 yılına ait kayıt ile 202I kodlu deney alanı karşılaştırıldı.",
    "province_Il": "İl Müdürlüğü, İ1 kodlu örnek alanın il bilgisini inceledi.",
}
CITIES = "Ankara İstanbul İzmir Eskişehir Çanakkale Şanlıurfa Iğdır Muğla Gümüşhane Kırşehir Kütahya Diyarbakır Erzurum Adıyaman Aydın Balıkesir Bilecik Bingöl Bolu Burdur Bursa Denizli Edirne Elazığ Erzincan Gaziantep Giresun Hatay Isparta Karabük Karaman Kars Kastamonu Kayseri Kırıkkale Kırklareli Kilis Kocaeli Konya Malatya Manisa Mardin Mersin Nevşehir Niğde Ordu Osmaniye Rize Sakarya Samsun Siirt Sinop Sivas Şırnak Tekirdağ Tokat Trabzon Tunceli Uşak Van Yalova Yozgat Zonguldak".split()
UNITS = ["İl Yazı İşleri Müdürlüğü", "İl Millî Eğitim Müdürlüğü", "İnsan Kaynakları Birimi",
         "Destek Hizmetleri Müdürlüğü", "Mali Hizmetler Birimi", "Strateji Geliştirme Birimi",
         "Bilgi İşlem Şube Müdürlüğü", "Özel Kalem Müdürlüğü", "Evrak Kayıt Birimi",
         "Arşiv ve Dokümantasyon Birimi", "Çevre Yönetimi Şube Müdürlüğü",
         "Kültür ve Sosyal İşler Müdürlüğü", "Hukuk İşleri Birimi", "Teknik İşler Müdürlüğü"]
SUBJECTS = ["Eğitim Programının Düzenlenmesi", "Yıllık Çalışma Planı", "Ödenek Talebi",
            "Arşiv Belgelerinin Tasnifi", "Personel Görevlendirmesi", "Hizmet İçi Eğitim",
            "Taşınır Kayıtlarının Güncellenmesi", "Bilgi Güvenliği Tedbirleri", "Bina Bakım ve Onarım İşleri",
            "Toplantı Gündemi", "Başvuruların Değerlendirilmesi", "İstatistik Bilgilerinin Gönderilmesi",
            "Çevre Düzenleme Çalışmaları", "Ulaşım Hizmetleri", "İş Sağlığı ve Güvenliği",
            "İletişim Bilgilerinin Güncellenmesi", "Elektronik Belge Yönetimi", "İhale Hazırlık İşlemleri",
            "Öğrenci İşleri", "Afet Hazırlık Çalışmaları", "Demirbaş Sayımı", "Bütçe Uygulamaları",
            "Kütüphane Hizmetleri", "Sosyal Yardım Başvuruları", "Denetim Sonuçları", "Faaliyet Raporu"]
FIRST = "İlker Işıl İpek İlknur Çağrı Çiğdem Şule Şükrü Gül Gökçe Özgür Özge Ümit Ülkü Esra Emre Derya Deniz Selin Eren".split()
LAST = "Örnek Işık Şen Güneş Çelik Özdemir Ünal Yıldız Güngör Çınar Şahin Aydın Yüce Sağlam Erdem Koç".split()
PURPOSES = ["bilgi ve belgelerin zamanında hazırlanabilmesi", "başvuruların usulüne uygun değerlendirilmesi",
            "kayıtların eksiksiz biçimde güncellenmesi", "birimler arasında eş güdümün sağlanması",
            "kaynakların verimli ve düzenli kullanılması", "işlemlerin elektronik ortamda izlenebilmesi",
            "ilgili personelin bilgilendirilmesi", "hizmetlerin aksamadan yürütülmesi"]
ENDINGS = ["gereğini rica ederim.", "bilgilerinize arz ederim.", "bilgi ve gereğini arz ve rica ederim.",
           "ilgili birimlere duyurulmasını rica ederim.", "sonucun yazılı olarak bildirilmesini rica ederim."]


def pick(rng, values):
    return values[int(rng.integers(len(values)))]


def fields(rng):
    year = int(rng.integers(2018, 2027))
    date = f"{int(rng.integers(1, 29)):02d}.{int(rng.integers(1, 13)):02d}.{year}"
    serial = int(rng.integers(10000, 99999999))
    number = f"{pick(rng, ['E', 'Z', 'O'])}-{int(rng.integers(10000000, 99999999))}-{int(rng.integers(100, 999))}.{int(rng.integers(1, 99)):02d}-{serial}"
    code = "-".join("".join(rng.choice(list("0123456789ABCDEF"), size=n)) for n in (8, 4, 4, 4, 12))
    return dict(city=pick(rng, CITIES), unit=pick(rng, UNITS), subject=pick(rng, SUBJECTS), date=date,
                year=year, serial=serial, number=number, code=code,
                person=pick(rng, FIRST) + " " + pick(rng, LAST), ending=pick(rng, ENDINGS),
                purpose=pick(rng, PURPOSES), extension=int(rng.integers(1000, 9999)))


def raw_text(kind, f, rng, group="none"):
    variant = int(rng.integers(6))
    if kind == "official_heading":
        return [f"T.C. {base.turkish_upper(f['city'])} VALİLİĞİ", f"{f['city']} {f['unit']}",
                f"{f['person']} / Vali a. / {f['unit']}",
                f"{f['city']} BELEDİYE BAŞKANLIĞI ({f['year']})",
                f"T.C. {f['city']} Kaymakamlığı — {f['unit']}",
                f"{base.turkish_upper(f['subject'])} / EK-{int(rng.integers(1, 30))}"][variant]
    if kind == "official_subject":
        return pick(rng, [f"Konu: {f['subject']}", f"Ek: {f['subject']} ({int(rng.integers(1, 25))} sayfa)",
                          f"Dağıtım: {f['unit']} / {f['city']}", f"İlgi: {f['date']} tarihli başvuru ({f['serial']})."])
    if kind == "official_body_short":
        return pick(rng, [f"{f['date']} tarihli başvurunuz işleme alınmıştır.",
                          f"{f['unit']} tarafından hazırlanan çizelge ekte sunulmuştur.",
                          f"{f['city']} ilindeki çalışmalar hakkında {f['ending']}",
                          f"{f['serial']} sayılı kaydın incelenerek sonucunun bildirilmesi rica olunur."])
    if kind == "official_body_long":
        return (f"{f['date']} tarihli ve {f['number']} sayılı yazı kapsamında, {f['purpose']} amacıyla "
                f"{f['unit']} tarafından hazırlanan belgelerin değerlendirilerek {f['ending']}")
    if kind == "official_footer_contact":
        phone = f"0 ({pick(rng, ['212', '216', '312', '232', '262'])}) {int(rng.integers(200, 999))} {int(rng.integers(10, 99))} {int(rng.integers(10, 99))}"
        return [f"Telefon: {phone} / Dahili: {f['extension']} / Faks: {phone}",
                f"E-posta: evrak.{f['serial']}@example.org / İnternet: https://kurum.example.org",
                f"Bilgi için: {f['person']} / {f['unit']} / Tel: {phone}",
                f"Adres: Örnek Mahallesi, Çınar Sokak No: {int(rng.integers(1, 180))}/{int(rng.integers(1, 60))}, {f['city']}",
                f"KEP: kurum.{f['serial']}@kep.example.org / E-posta: iletisim@example.org",
                f"İletişim: {f['unit']}; {phone}; bilgi.{f['serial']}@example.org"][variant]
    if kind == "official_verification":
        return pick(rng, [f"Belge Doğrulama Kodu: {f['code']}",
                          f"Belge Doğrulama Adresi: https://belge.example.org/dogrula?kod={f['code']}",
                          f"Bu belge, güvenli elektronik imza ile imzalanmıştır. Doğrulama: {f['code']}",
                          f"https://evrak.example.org/belge/{f['serial']} / Doğrulama kodu: {f['code']}"])
    if kind == "official_date_reference":
        return pick(rng, [f"Sayı: {f['number']} / Tarih: {f['date']}",
                          f"İlgi: a) {f['date']} tarihli ve {f['number']} sayılı yazı.",
                          f"Başvuru Tarihi: {f['date']} / Kayıt No: {f['serial']}",
                          f"{f['year']}/{int(rng.integers(1, 9999)):04d} — {f['date']} — Dosya: {f['number']}"])
    if kind == "official_amount":
        amount = f"{int(rng.integers(1, 9999999)):,}".replace(",", ".") + f",{int(rng.integers(100)):02d}"
        return pick(rng, [f"Toplam Tutar: {amount} TL (KDV dâhil) / Belge No: {f['serial']}",
                          f"Ödenek: {amount} TL; Dönem: {f['year']}; Birim: {f['unit']}",
                          f"Birim Fiyat: {amount} TL / Miktar: {int(rng.integers(1, 500))} adet / {f['date']}",
                          f"Tahakkuk No: {f['serial']} — Ödenecek tutar: {amount} TL"])
    members = GROUPS[group]
    if kind == "official_confusables_words":
        return f"{WORDS[group]} ({f['date']}, kayıt {f['serial']})"
    if kind == "official_confusables_codes":
        # Each member occurs once in the tagged target span, including digraphs.
        tokens = list(members)
        rng.shuffle(tokens)
        return "Kod alanları: " + " / ".join(tokens) + f" ; Belge: {f['number']}"
    raise ValueError(kind)


def select(row, font, rng):
    for _ in range(10000):
        source = raw_text(row["official_kind"], fields(rng), rng, row["targeted_group"])
        source = unicodedata.normalize("NFC", source)
        text = base.turkish_upper(source) if row["text_case"] == "uppercase" else source
        if row["mixed_typography"] in ("superscript", "subscript"):
            text += " " + str(int(rng.integers(2, 10)))
        if row["official_kind"] == "official_body_long" and not 150 <= len(text) <= 252:
            continue
        if dataset.safe_text(text) and len(text) <= 252 and set(map(ord, text)) <= font.codepoints:
            if dataset.text_split(source) == row["split"] and dataset.text_split(text) == row["split"]:
                return text, source
    raise ValueError(f"No synthetic official label for {row['index']}")


def source_id(source):
    return hashlib.sha256(dataset.canonical(source).encode()).hexdigest()
