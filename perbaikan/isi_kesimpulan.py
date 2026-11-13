# Bagian 6 laporan; dieksekusi dari isi_laporan.py. Angka diambil dari file hasil, bukan ditulis tangan.
P("6. Interpretasi, keterbatasan, dan langkah berikutnya", "h2")
sc = SC.set_index("Kandidat"); n_try = len(SC); n_ok = int((SC.Diadopsi == "ya").sum())
cap = SC[SC.Tahap == "S1 kapasitas"]["Perubahan (%)"]; nb_neg = int((BULAN["Selisih v4 − v3"] < 0).sum()); nb_sig = int((BULAN.Kesimpulan == "lebih baik").sum())
pv, ptt = PITA[PITA.phase == "Validation"].set_index("pita"), PITA[PITA.phase == "Test"].set_index("pita")
jt = JAM[JAM.phase == "test"]
def chg(k): return float(SC[SC.Kandidat.str.contains(k, regex=False)]["Perubahan (%)"].iloc[0])
P("6.1 Apa yang ternyata membantu dan apa yang tidak", "h3")
bullets([
    f"Dari {n_try} kandidat yang dicoba di validation, {n_ok} lolos aturan adopsi: <b>koreksi bias sekuensial</b> "
    f"(jendela {bc['jendela_hari']} hari, per pita horizon × jam target, shrink {bc['shrink']:g}). Model dasar v4 sama persis dengan model final v3.",
    f"<b>Model yang lebih besar justru lebih buruk.</b> Menambah pohon atau kedalaman mengubah MAE validation {cap.min():+.1f}% sampai {cap.max():+.1f}%, "
    "dan makin dalam makin buruk. Artinya model v3 tidak kekurangan kapasitas; model yang lebih besar hanya menghafal derau.",
    f"<b>Fitur tambahan dan half-life 90 hari membaik tipis</b> ({chg('fitur jam-sama'):+.2f}% dan {chg('half-life 90'):+.2f}%), di bawah ambang 0,5%, sehingga tidak dipakai. "
    f"Model per pita horizon ({chg('per pita horizon (4 pita)'):+.2f}%) dan refit mingguan ({chg('refit mingguan + koreksi'):+.2f}% terhadap putaran 1) lebih buruk.",
    "Kesimpulannya: informasi yang bisa diprediksi dari riwayat CPU, kalender, dan delapan telemetri sudah terpakai oleh v3. Yang masih bisa diambil adalah "
    "<b>pergeseran pola harian yang baru terjadi</b>, dan itu yang ditangkap oleh koreksi bias.",
])
P("6.2 Seberapa besar perbaikannya", "h3")
bullets([
    f"<b>Validation:</b> MAE {v3v:.2f} → {v4v:.2f} pp ({pct(v4v, v3v):+.1f}%), dan uji DM menyatakan {dv['Kesimpulan']} ({ci(dv)}). "
    "Namun parameter koreksi dipilih di periode ini, jadi angka ini cenderung optimis.",
    f"<b>Test:</b> MAE {v3t:.2f} → {v4t:.2f} pp ({pct(v4t, v3t):+.1f}%), dan uji DM menyatakan <b>{dt['Kesimpulan']}</b> ({ci(dt)}). "
    f"Terhadap rata-rata 4 minggu, v4 unggul {-d4t['Selisih MAE (A − B)']:.2f} pp (sebelumnya {-d4t_old['Selisih MAE (A − B)']:.2f} pp), tetapi masih {d4t['Kesimpulan']} (p = {d4t['p']:.3f}).",
    f"<b>Arah perbaikan konsisten, besarnya kecil.</b> MAE v4 lebih rendah dari v3 di {nb_neg} dari {len(BULAN)} bulan origin, tetapi nyata secara statistik hanya di {nb_sig} bulan (Tabel 15).",
    f"<b>Bias per jam mengecil.</b> Di test rentangnya menyempit dari {jt.bias_v3_final.min():+.1f}…{jt.bias_v3_final.max():+.1f} pp menjadi {jt.bias_v4_final.min():+.1f}…{jt.bias_v4_final.max():+.1f} pp (Gambar 2).",
    f"<b>Yang memburuk: jam beban tinggi.</b> MAE puncak naik dari {m('Validation', NB, 'MAE puncak'):.2f} ke {m('Validation', NA, 'MAE puncak'):.2f} pp (validation) dan dari "
    f"{m('Test', NB, 'MAE puncak'):.2f} ke {m('Test', NA, 'MAE puncak'):.2f} pp (test). Koreksi menarik prediksi ke median galat terkini, sehingga lonjakan langka sedikit lebih terlewat. "
    "Untuk perencanaan kapasitas tetap pakai kuantil atas (bagian 9 notebook), bukan prediksi titik.",
    f"<b>Horizon 1–6 jam tidak berubah</b> ({ptt.loc['1–6 jam', 'v3_final']:.2f} pp di test, sebelum dan sesudah). Pada pita ini hanya ada 14 galat matang per jam target dalam 14 hari, "
    "di bawah batas minimum 30 yang ditetapkan sebelum eksperimen, sehingga koreksinya tidak aktif. Batas itu sengaja tidak diubah setelah hasil test terlihat.",
])
P("6.3 Jawaban jujur atas pertanyaan 'apakah sekarang sudah bagus?'", "h3")
P(f"Belum banyak berubah. Perbaikan yang kredibel dan lolos seleksi menurunkan MAE test dari {v3t:.2f} ke {v4t:.2f} pp, dan selisih sekecil itu belum bisa dibedakan dari kebetulan. "
  "Bukti dari seluruh percobaan mengarah ke hal yang sama: tiga resolusi data memberi hasil setara, Ridge setara dengan pembanding sederhana, dan model yang lebih besar memburuk. "
  "Dengan data dan variabel yang ada, galat sekitar 7 pp tampaknya dekat dengan batas yang bisa dicapai; sisanya adalah lonjakan tidak teratur yang tidak punya tanda di data historis. "
  "Ini temuan yang sah untuk skripsi, asalkan dilaporkan apa adanya.")
P("6.4 Keterbatasan laporan ini", "h3")
bullets([
    "Periode test (September–Oktober 2026) sudah pernah dilihat di versi sebelumnya, sehingga semua angka test eksploratori.",
    f"{n_try} kandidat dibandingkan di validation yang sama. Aturan adopsi dua syarat mengurangi, tetapi tidak menghilangkan, risiko memilih kandidat yang kebetulan menang.",
    "Perbaikan hanya dikerjakan untuk model 1 jam. Model 30 menit dan 15 menit tidak diubah karena kinerjanya setara dengan model 1 jam.",
    "Prediksi kuantil, biaya kapasitas, dan SHAP (bagian 9–11 notebook) belum dihitung ulang untuk v4.",
    "v4 berada di luar seleksi beku dan pra-registrasi v3. Notebook v3, seleksi beku, dan pra-registrasinya tidak diubah oleh pekerjaan ini.",
])
P("6.5 Langkah berikutnya yang disarankan", "h3")
bullets([
    "<b>Tetap pakai v3 sebagai model final untuk uji konfirmasi</b> sesuai pra-registrasi. Bila v4 ingin ikut diuji, tambahkan sebagai amandemen pra-registrasi "
    "<b>sebelum</b> nilai CPU setelah 1 November 2026 dibaca.",
    "Laporkan koreksi bias sebagai analisis tambahan: arah perbaikannya konsisten, tetapi di test belum nyata secara statistik.",
    "Bila v4 dipakai untuk operasi, hitung ulang kuantil di atas prediksi yang sudah dikoreksi, dan periksa lagi cakupan intervalnya.",
    "Sumber perbaikan yang lebih menjanjikan ada di luar data ini: variabel penyebab beban (jadwal job atau batch, jumlah pengguna atau permintaan) dan riwayat yang lebih panjang. "
    "Ini dugaan yang belum diuji di sini.",
    "Tambahkan .gitattributes (<i>*.csv -text</i>) agar hash data tidak berubah saat repo di-clone di Windows.",
])
