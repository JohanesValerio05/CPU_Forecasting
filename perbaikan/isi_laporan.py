# Isi laporan; dieksekusi oleh buat_laporan.py (memakai P, bullets, table, image, dan data yang sudah dimuat di sana).
import subprocess
def git(*a):
    try: return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    except Exception: return "?"
pct = lambda a, b: 100 * (a / b - 1)
def ci(r): return f"{r['Selisih MAE (A − B)']:+.3f} pp, CI 95% [{r['CI 95% bawah']:+.3f}; {r['CI 95% atas']:+.3f}], p = {r['p']:.3f}"
B4 = "Rata-rata 4 minggu"
v3v, v3t, v4v, v4t = m("Validation", NB, "MAE"), m("Test", NB, "MAE"), m("Validation", NA, "MAE"), m("Test", NA, "MAE")
b4v, b4t = m("Validation", B4, "MAE"), m("Test", B4, "MAE")
dv, dt = dm("Validation", NA, NB), dm("Test", NA, NB); d4t, d4t_old = dm("Test", NA, B4), dm("Test", NB, B4)
bc = sel["koreksi_bias"]; fam_new = sel["spesifikasi_keluarga_baru"]; FAMN = {"hgb": "HistGradientBoosting", "lgbm": "LightGBM", "xgb": "XGBoost"}
ok_runs = all(rc == 0 for _, rc, _ in runs) and len(runs) >= 3

# ============================================================ halaman judul + ringkasan
P("Laporan forecasting penggunaan CPU 120 jam", "h1")
P("Hasil run tiga notebook, evaluasi, dan perbandingan sebelum vs sesudah perbaikan", "h3")
BLN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]; _n = time.localtime()
P(f"Disusun {_n.tm_mday} {BLN[_n.tm_mon - 1]} {_n.tm_year}, {time.strftime('%H:%M')} WIB · repo CPU_Forecasting, branch {git('rev-parse', '--abbrev-ref', 'HEAD')}, commit {git('rev-parse', '--short', 'HEAD')} · "
  f"GPU NVIDIA RTX 3060 Laptop (XGBoost CUDA; LightGBM, HistGradientBoosting di CPU)", "s")
P("1. Ringkasan", "h2")
bullets([
    (f"<b>Run.</b> Ketiga notebook forecasting (1 jam, 30 menit, 15 menit) selesai tanpa galat." if ok_runs else
     f"<b>Run.</b> Status eksekusi notebook ada di Tabel 1; tidak semuanya selesai tanpa galat.") +
    " Model 1 jam ikut dilatih ulang dari nol, dan semua tabel metrik ketiga notebook sama dengan yang sudah di-commit (selisih angka terbesar &lt; 0,00001).",
    f"<b>Sebelum (model final v3, 1 jam).</b> MAE validation {v3v:.2f} pp dan test {v3t:.2f} pp. Terhadap pembanding sederhana <i>rata-rata 4 minggu</i> "
    f"(validation {b4v:.2f}; test {b4t:.2f}) model hanya {-pct(v3t, b4t):.1f}% lebih baik di test, dan selisih itu <b>{d4t_old['Kesimpulan']}</b> "
    f"({ci(d4t_old)}). Jam beban tinggi (CPU ≥ 80%) diprediksi terlalu rendah {-m('Test', NB, 'Bias puncak'):.1f} pp di test. "
    "Resolusi 30 menit dan 15 menit tidak berbeda nyata dari 1 jam. Jadi kinerjanya memang masih lemah.",
    (f"<b>Sesudah (kandidat v4).</b> MAE validation {v4v:.2f} pp ({pct(v4v, v3v):+.1f}%) dan test {v4t:.2f} pp ({pct(v4t, v3t):+.1f}%). "
     f"Uji Diebold–Mariano v4 vs v3: validation <b>{dv['Kesimpulan']}</b> ({ci(dv)}); test <b>{dt['Kesimpulan']}</b> ({ci(dt)}). "
     f"Terhadap rata-rata 4 minggu di test, v4 kini <b>{d4t['Kesimpulan']}</b> ({ci(d4t)}).") if berubah else
    "<b>Sesudah.</b> Tidak ada perubahan yang lolos aturan adopsi di validation, sehingga model final tetap v3. Rincian percobaan ada di bagian 4.",
    "<b>Cara perbaikan.</b> Semua keputusan diambil hanya dari periode validation dengan aturan adopsi yang ditetapkan sebelum eksperimen, "
    "seleksi v4 dibekukan (sha256) sebelum test dihitung, dan test dijalankan sekali. Seleksi beku dan pra-registrasi v3 tidak diubah.",
    "<b>Batas klaim.</b> Periode test sudah pernah dilihat pada versi sebelumnya, jadi angka test bersifat eksploratori, sama seperti di notebook v3. "
    "Bukti konfirmatori tetap harus datang dari data setelah 1 November 2026, yang tidak dibaca di sini.",
])
t = M[M.Model.isin([B4, NB, NA])].copy()
table(t, "Tabel ringkas. Model 1 jam sebelum dan sesudah perbaikan (satuan pp CPU; MAE relatif = MAE dibagi MAE rata-rata 4 minggu)", F3,
      widths=[1.3, 5.2, 1, 1.2, 1, 1, 1.3, 1.3], hl=lambda r: r.Model == NA,
      note="Puncak = jam target dengan CPU aktual ≥ 80%. Bias negatif = prediksi terlalu rendah.")

# ============================================================ 2. eksekusi
P("2. Eksekusi notebook", "h2")
P("Ketiga notebook dijalankan berurutan dengan kernel <i>Python (skripsi GPU)</i> lewat nbclient. Notebook sumber tidak diubah; salinan yang sudah dieksekusi "
  "(lengkap dengan semua tabel dan gambar) disimpan di folder <b>hasil_eksekusi/</b>.")
if runs:
    table(pd.DataFrame([{"Notebook": a, "Status": "selesai" if rc == 0 else "GAGAL", "Durasi (menit)": d} for a, rc, d in runs]), "Tabel 1. Status eksekusi",
          {"Durasi (menit)": "{:.1f}"}, widths=[5, 2, 2],
          note="Notebook 30 menit dan 15 menit cepat karena memakai cache pelatihan yang sudah ada di repo; notebook 1 jam melatih ulang semuanya (lihat temuan di bawah).")
P("<b>Temuan saat run: akhir baris CSV berubah karena pengaturan git.</b> Repo ini di-clone dengan <i>core.autocrlf = true</i>, sehingga ketiga file data "
  "(cleaned_*.csv) di folder kerja memakai akhir baris CRLF, padahal file aslinya LF. Isi datanya sama, tetapi sha256-nya berbeda dari yang tercatat di seleksi beku. Akibatnya:")
bullets(["Cache model 1 jam tidak terpakai, sehingga notebook 1 jam melatih ulang semua konfigurasi dari nol (itu sebabnya durasinya paling lama). Seleksi yang terpilih "
         "sama dengan seleksi beku dan angkanya sama (Tabel 1b). Ini sekaligus bukti bahwa hasil v3 bisa direproduksi.",
         "Notebook 30 menit dan 15 menit akan berhenti di pengecekan hash data. Sebelum keduanya dijalankan, ketiga CSV dikembalikan ke byte aslinya dari git "
         "(sha256 kini sama dengan yang tercatat di seleksi beku), sehingga keduanya berjalan normal.",
         "Saran: tambahkan file .gitattributes berisi <i>*.csv -text</i> agar git tidak lagi mengubah akhir baris file data di komputer mana pun. "
         "Sampai itu dilakukan, <i>git status</i> menampilkan ketiga CSV sebagai berubah walaupun isinya sama dengan yang di-commit."])
RP = rd(f"{EX}/reproduksi_vs_commit.csv")
if RP is not None:
    table(RP, "Tabel 1b. Keluaran run ini dibanding versi yang sudah di-commit di repo", {"Baris": "{:.0f}", "Selisih angka terbesar": "{:.7f}"}, widths=[7.5, 1.2, 3, 3],
          note="Selisih sebesar ini berasal dari pembulatan saat menulis file dan urutan operasi di GPU; tidak ada angka laporan yang berubah.")

# ============================================================ 3. evaluasi sebelum
story.append(PageBreak())
P("3. Evaluasi hasil run (kondisi sebelum perbaikan)", "h2")
P("3.1 Model 1 jam (skripsi_forecast_v3)", "h3")
table(T10.drop(columns=["Final"]), "Tabel 2. Kinerja semua model terpilih, model 1 jam (dari outputs_v3_baru/metrik.csv)", F3, widths=[1.3, 5.6, 1, 1.2, 1, 1, 1.2, 1.2],
      hl=lambda r: r.Model == "Ensemble GBDT · CPU + kalender + telemetri terpilih", note="Baris berlatar = model final v3 (dipilih di validation, seleksi beku).")
t = DM3[DM3["Dibanding (B)"].isin([B4, "Naif musiman mingguan"]) | DM3["Model A"].eq(DM3["Model A"].iloc[0])].drop(columns=["m"])
table(t, "Tabel 3. Uji Diebold–Mariano fixed-m, model 1 jam (negatif = model A lebih akurat)", FD, widths=[1.5, 3.3, 3.1, 1.4, 1.35, 1.35, 1.15, 1.0, 2.2])
if STAB is not None:
    table(STAB.drop(columns=["MAE puncak final"]), "Tabel 4. Stabilitas per bulan origin: model final v3 vs rata-rata 4 minggu",
          {"MAE final": "{:.2f}", "MAE rata-rata 4 minggu": "{:.2f}", "MAE relatif": "{:.3f}", "Selisih (final − pembanding)": "{:+.2f}", "CI 95% bawah": "{:+.2f}", "CI 95% atas": "{:+.2f}"},
          widths=[1.4, 1.5, 1, 1.2, 1.5, 1.2, 1.7, 1.2, 1.2, 2.2])
if RIDGE is not None:
    table(RIDGE[RIDGE.Model.str.startswith("Ridge")].drop(columns=["Final"]), "Tabel 5. Pembanding linear Ridge (bagian 8.3 notebook)", F3, widths=[1.3, 5.6, 1, 1.2, 1, 1, 1.2, 1.2])
if WL is not None:
    table(WL[["Fase", "λ", "MAE keseluruhan", "Perubahan MAE keseluruhan (%)", "MAE puncak", "Perubahan MAE puncak (%)", "Bias puncak"]],
          "Tabel 6. Weighted loss untuk jam beban tinggi (bagian 8.4 notebook; λ = 0 adalah model final)",
          {"λ": "{:g}", "MAE keseluruhan": "{:.3f}", "Perubahan MAE keseluruhan (%)": "{:+.1f}", "MAE puncak": "{:.2f}", "Perubahan MAE puncak (%)": "{:+.1f}", "Bias puncak": "{:+.2f}"},
          widths=[1.3, .6, 1.5, 1.9, 1.3, 1.8, 1.3])
if PROB is not None:
    table(PROB[["Fase", "Metode", "Terpilih", "Skor kuantil", "Cakupan 80% (%)", "Lebar 80%", "Cakupan 90% (%)", "Puncak di bawah kuantil 0,95 (%)"]].fillna(""),
          "Tabel 7. Prediksi probabilistik (bagian 9 notebook)", {"Skor kuantil": "{:.3f}", "Cakupan 80% (%)": "{:.1f}", "Lebar 80%": "{:.1f}", "Cakupan 90% (%)": "{:.1f}",
                                                                 "Puncak di bawah kuantil 0,95 (%)": "{:.1f}"}, widths=[1.3, 5.4, .9, 1.1, 1.2, 1, 1.2, 1.7])
P("3.2 Resolusi 30 menit dan 15 menit", "h3")
fin = "Ensemble GBDT · CPU + kalender + telemetri terpilih"
rows = []
for nm, d, c in (("30 menit", M30, "MAE 30 menit"), ("15 menit", M15, "MAE 15 menit")):
    if d is None: continue
    for _, r in d[d.Model.isin([B4, fin])].iterrows():
        rows.append({"Resolusi": nm, "Fase": r.Fase, "Model": r.Model, "MAE model 1 jam": r["MAE 1 jam"], "MAE resolusi ini": r[c], "Selisih": r[c] - r["MAE 1 jam"]})
if rows: table(pd.DataFrame(rows), "Tabel 8. MAE per jam: model 30 menit dan 15 menit dibanding model 1 jam pada origin yang sama",
               {"MAE model 1 jam": "{:.3f}", "MAE resolusi ini": "{:.3f}", "Selisih": "{:+.3f}"}, widths=[1.3, 1.3, 5.6, 1.5, 1.5, 1.1])
if DMR3 is not None:
    t = DMR3.copy(); t["Model A"] = t["Model A"].str.split(" · ").str[0]; t["Dibanding (B)"] = t["Dibanding (B)"].str.split(" · ").str[0]
    table(t.drop(columns=["m"]), "Tabel 9. Uji Diebold–Mariano antar-resolusi (model final masing-masing)", FD, widths=[1.5, 1.5, 1.5, 1.6, 1.35, 1.35, 1.15, 1, 2.2])
P("Ketiga resolusi menghasilkan MAE per jam yang praktis sama (selisih di bawah 0,1 pp dan tidak berbeda nyata). Data yang lebih rapat tidak menambah akurasi "
  "prediksi rata-rata per jam, sehingga perbaikan di bagian berikut difokuskan ke model 1 jam.")
P("3.3 Diagnosis: di mana model masih lemah", "h3")
pt = PITA[PITA.phase == "Test"].set_index("pita"); jt = JAM[JAM.phase == "test"]
bullets([
    f"<b>Keunggulan atas pembanding sederhana tipis dan tidak stabil.</b> MAE relatif {m('Validation', NB, 'MAE relatif'):.3f} di validation tetapi {m('Test', NB, 'MAE relatif'):.3f} di test; "
    f"di test selisihnya terhadap rata-rata 4 minggu {d4t_old['Kesimpulan']}. Per bulan (Tabel 4), model final nyata lebih baik hanya di sebagian bulan.",
    f"<b>Galat hampir rata di semua horizon.</b> Di test, MAE {pt.loc['1–6 jam', 'v3_final']:.2f} pp untuk 1–6 jam ke depan dan {pt.loc['73–120 jam', 'v3_final']:.2f} pp untuk 73–120 jam. "
    "Prediksi beberapa jam ke depan seharusnya jauh lebih mudah; ini tanda model belum memanfaatkan informasi terkini dengan baik.",
    f"<b>Bias mengikuti jam.</b> Di test, bias per jam target berkisar {jt.bias_v3_final.min():+.1f} sampai {jt.bias_v3_final.max():+.1f} pp (Gambar 2): "
    "terlalu rendah pada dini hari sampai pagi, terlalu tinggi pada siang sampai sore. Pola harian beban bergeser dibanding data latih (concept drift).",
    f"<b>Jam beban tinggi diprediksi terlalu rendah.</b> Bias puncak {m('Validation', NB, 'Bias puncak'):+.1f} pp (validation) dan {m('Test', NB, 'Bias puncak'):+.1f} pp (test). "
    "Ini sifat prediksi yang meminimalkan MAE (median), bukan kesalahan kode; kebutuhan kapasitas ditangani lewat kuantil (Tabel 7).",
])

# ============================================================ 4. metode perbaikan
P("4. Metode perbaikan", "h2")
P("<b>Keterbatasan yang dihormati.</b> Data hanya sekitar satu tahun dari satu server; telemetri yang boleh dipakai tetap delapan metrik terpilih; "
  "data setelah 1 November 2026 tidak dibaca; seleksi beku dan pra-registrasi v3 tidak diubah; perangkat keras satu laptop. Karena itu yang dicoba hanya perubahan "
  "yang murah, kausal (memakai data sampai origin saja), dan punya dasar di literatur yang sudah dipakai notebook.")
P("<b>Aturan adopsi (ditetapkan sebelum eksperimen).</b> Sebuah perubahan dipakai hanya bila MAE validation turun minimal 0,5% relatif terhadap acuannya "
  "<b>dan</b> lebih baik di minimal 3 dari 4 fold bulanan validation (Mei–Agustus 2026). Syarat kedua mencegah perubahan yang hanya menang di satu bulan. "
  "Tahap dijalankan berurutan; tahap berikutnya memakai hasil tahap sebelumnya sebagai acuan.")
table(pd.DataFrame([
    ("S1", "Kapasitas model", "Jumlah pohon 300/600/1000 × kedalaman 4/5/6 (15/31/63 daun). v3 memakai 300 pohon dangkal untuk ±0,5 juta baris.", "Bentéjac dkk. (2021); Makridakis dkk. (2022a)"),
    ("S2", "Fitur tambahan", "CPU pada jam yang sama dengan target dalam 7 hari terakhir (nilai terakhir, rata-rata, SD, maksimum), selisihnya terhadap rata-rata 4 minggu, "
                             "dan SD/maks/min/median dari 4 nilai mingguan. Menangkap pergeseran pola harian yang baru terjadi.", "Januschowski dkk. (2022); Hyndman & Athanasopoulos (2021, §13.1)"),
    ("S3", "Half-life bobot waktu", "Tanpa bobot, 45, 90, 180, 365 hari.", "Bayram dkk. (2022); Hyndman & Athanasopoulos (2021, §8.1)"),
    ("S4", "Model per pita horizon", "Empat model terpisah (1–6, 7–24, 25–72, 73–120 jam) sebagai ganti satu model untuk semua horizon.", "Ben Taieb dkk. (2012)*"),
    ("S5", "Per keluarga", "Pengaturan yang lolos S1–S4 (diuji pada XGBoost) dipakai keluarga lain hanya bila keluarga itu sendiri membaik.", "Cerqueira dkk. (2020)"),
    ("S6", "Ensemble", "Rata-rata tiga keluarga, dibandingkan dengan model final v3.", "Wang dkk. (2023)"),
    ("S7", "Koreksi bias sekuensial", "Prediksi digeser sebesar median galat yang sudah matang (waktu target sudah lewat saat origin) dalam 14/28 hari terakhir, "
                                      "per pita horizon atau per pita × jam target, dengan shrink 0,5 atau 1. Mekanismenya sama dengan kalibrasi konformal di bagian 9 notebook.",
     "Xu & Xie (2023); Bayram dkk. (2022)"),
    ("P2", "Refit mingguan", "Putaran 2, ditambahkan setelah putaran 1 menunjukkan bahwa yang membantu adalah penyesuaian terhadap pola terkini: ensemble yang sama dilatih ulang "
                             "tiap 7 hari, bukan tiap bulan. Koreksi bias memakai parameter beku putaran 1 (tidak dipilih ulang). Diputuskan sebelum test dihitung.", "Bayram dkk. (2022)")],
    columns=["Tahap", "Perubahan", "Isi", "Dasar"]),
    "Tabel 10. Tahap percobaan", widths=[.8, 2.3, 8.6, 4],
    note="* Ben Taieb, Bontempi, Atiya & Sorjamaa (2012), Expert Systems with Applications 39(8): belum ada di daftar pustaka notebook; periksa dulu sebelum dikutip.")
if SC is not None:
    t = SC[["Tahap", "Kandidat", "MAE validation", "MAE acuan", "Perubahan (%)", "Fold lebih baik (dari 4)", "Diadopsi"]].rename(
        columns={"MAE validation": "MAE val.", "Perubahan (%)": "Beda (%)", "Fold lebih baik (dari 4)": "Fold menang (dari 4)"})
    table(t, "Tabel 11. Seluruh kandidat yang dicoba di validation dan keputusannya (tidak ada yang disembunyikan)",
          {"MAE val.": "{:.3f}", "MAE acuan": "{:.3f}", "Beda (%)": "{:+.2f}", "Fold menang (dari 4)": "{:.0f}"}, widths=[2.3, 6.2, 1.3, 1.3, 1.3, 1.6, 1.5],
          hl=lambda r: r.Diadopsi == "ya", note="Baris berlatar = diadopsi. Bila beberapa kandidat dalam satu tahap lolos, dipakai yang MAE validation-nya terendah.")
P("<b>Hasil seleksi (dibekukan sebelum test).</b>")
its = []
if fam_new:
    s0 = next(iter(fam_new.values()))
    its.append(f"Ensemble baru: {', '.join(FAMN[k] for k in fam_new)} memakai {s0['n_iter']} pohon, kedalaman {sel['kapasitas'][s0['cap']]['depth']} / {sel['kapasitas'][s0['cap']]['leaves']} daun"
               + (", fitur tambahan" if s0["extra"] else "") + (", model per pita horizon" if s0["bands"] else "") + "."
               + (f" Keluarga yang tetap memakai konfigurasi beku v3: {', '.join(FAMN[k] for k in sel['keluarga_tetap_v3'])}." if sel["keluarga_tetap_v3"] else ""))
else: its.append("Model dasar tidak berubah: ensemble tiga keluarga dengan konfigurasi beku v3.")
its.append(f"Koreksi bias sekuensial: jendela {bc['jendela_hari']} hari, per {'pita horizon × jam target' if bc['kelompok'] == 'pita_jam' else 'pita horizon'}, shrink {bc['shrink']:g}." if bc
           else "Koreksi bias sekuensial tidak diadopsi.")
mv2 = sel2["mae_validation"]
its.append((f"Refit mingguan <b>diadopsi</b>: MAE validation {mv2['putaran1_final']:.3f} → {mv2['refit_mingguan_final']:.3f} pp." if sel2["refit_mingguan"] else
            f"Refit mingguan <b>tidak diadopsi</b>: MAE validation {mv2['refit_mingguan_final']:.3f} pp dibanding {mv2['putaran1_final']:.3f} pp tanpa refit mingguan, "
            "tidak memenuhi aturan adopsi. Model tetap dilatih ulang tiap bulan."))
its.append(f"MAE validation: v3 {sel['mae_validation']['v3_final']:.3f} → v4 {mv2['v4_final']:.3f} pp. "
           f"Seleksi dibekukan {sel['dibekukan']} (putaran 1) dan {sel2['dibekukan']} (putaran 2), keduanya sebelum test dihitung "
           "(outputs_perbaikan/seleksi_v4_beku.json dan seleksi_v4_putaran2_beku.json, masing-masing dengan sha256).")
bullets(its)

# ============================================================ 5. sebelum vs sesudah
P("5. Hasil: sebelum vs sesudah", "h2")
table(M, "Tabel 12. Kinerja sebelum dan sesudah, model 1 jam, pada origin dan target yang sama", F3, widths=[1.3, 5.6, 1, 1.2, 1, 1, 1.2, 1.2],
      hl=lambda r: r.Model == NA, note=("Baris 'Ablasi' dihitung setelah seleksi dibekukan, hanya untuk menunjukkan sumbangan tiap komponen; tidak dipakai untuk memilih apa pun."
                                         if M.Model.str.startswith("Ablasi").any() else "Model dasar v4 sama dengan v3; satu-satunya perbedaan adalah koreksi bias sekuensial."))
t = DM.copy(); t["Model A"] = t["Model A"].str.replace(" (Ensemble GBDT + telemetri)", "", regex=False); t["Dibanding (B)"] = t["Dibanding (B)"].str.replace(" (Ensemble GBDT + telemetri)", "", regex=False)
table(t.drop(columns=["m"]), "Tabel 13. Uji Diebold–Mariano fixed-m sebelum vs sesudah (negatif = model A lebih akurat)", FD, widths=[1.5, 3.3, 3.1, 1.4, 1.35, 1.35, 1.15, 1.0, 2.2],
      hl=lambda r: r["Model A"] == NA and r["Dibanding (B)"].startswith("SEBELUM"))
image(f"{FIG}/L1_mae_per_horizon.png", "Gambar 1. MAE per horizon: pembanding, sebelum (v3), dan sesudah (v4)")
t = PITA.rename(columns={"phase": "Fase", "pita": "Horizon", "mean4w": B4, "v3_final": "Sebelum (v3)", "v4_final": "Sesudah (v4)"}); t["Perubahan (%)"] = 100 * (t["Sesudah (v4)"] / t["Sebelum (v3)"] - 1)
table(t, "Tabel 14. MAE per pita horizon", {B4: "{:.2f}", "Sebelum (v3)": "{:.2f}", "Sesudah (v4)": "{:.2f}", "Perubahan (%)": "{:+.1f}"}, widths=[1.5, 1.8, 1.8, 1.5, 1.5, 1.5])
image(f"{FIG}/L2_bias_per_jam.png", "Gambar 2. Bias per jam target, sebelum dan sesudah (0 = tidak bias)")
image(f"{FIG}/L3_mae_per_bulan.png", "Gambar 3. MAE per bulan origin")
t = BULAN.rename(columns={"mean4w": B4, "v3_final": "Sebelum (v3)", "v4_final": "Sesudah (v4)"})
table(t, "Tabel 15. MAE per bulan origin dan uji DM v4 vs v3 per bulan", {B4: "{:.2f}", "Sebelum (v3)": "{:.2f}", "Sesudah (v4)": "{:.2f}", "Selisih v4 − v3": "{:+.2f}",
                                                                          "CI 95% bawah": "{:+.2f}", "CI 95% atas": "{:+.2f}"}, widths=[1.4, 1.4, .9, 1.4, 1.2, 1.2, 1.3, 1.1, 1.1, 2.2])
exec(open(os.path.join(ROOT, "perbaikan", "isi_kesimpulan.py"), encoding="utf-8").read())

# ============================================================ lampiran
story.append(PageBreak())
P("Lampiran A. Gambar dari notebook yang dijalankan", "h2")
image(f"{V3}/figures/02_mae_per_horizon.png", "A1. Model 1 jam: MAE per horizon (notebook v3)")
image(f"{V3}/figures/03_prediksi_vs_aktual_test.png", "A2. Model 1 jam: prediksi vs aktual di test (notebook v3)")
image(f"{V3}/figures/05b_tradeoff_weighted_loss.png", "A3. Model 1 jam: trade-off weighted loss untuk jam beban tinggi", width=13 * cm)
image(f"{V3}/figures/06_kalibrasi_kuantil.png", "A4. Model 1 jam: kalibrasi kuantil", width=13 * cm)
image(f"{R15}/figures/04_mae_tiga_resolusi.png", "A5. MAE per horizon untuk tiga resolusi (notebook 15 menit)")
P("Lampiran B. File yang dihasilkan", "h2")
table(pd.DataFrame([
    ("hasil_eksekusi/", "Tiga notebook yang sudah dieksekusi, run_log.txt, reproduksi_vs_commit.csv, dan skrip jalankan_semua.py."),
    ("outputs_v3_baru/, outputs_30m_baru/, outputs_15m_baru/", "Keluaran notebook (metrik, uji DM, prediksi, gambar). Seleksi beku dan pra-registrasi tidak berubah."),
    ("perbaikan/eksperimen_perbaikan.py", "Eksperimen perbaikan: memuat sel notebook v3 apa adanya, seleksi di validation, pembekuan, lalu test."),
    ("outputs_perbaikan/", "seleksi_validation.csv, seleksi_v4_beku.json dan seleksi_v4_putaran2_beku.json (+ sha256), prediksi_v4.csv.gz, metrik_sebelum_sesudah.csv, uji_dm_v4.csv, tabel per pita/bulan/jam, gambar."),
    ("perbaikan/buat_laporan.py", "Penyusun laporan ini.")], columns=["Lokasi", "Isi"]), widths=[5.5, 11])
P("Untuk mengulang: <i>python hasil_eksekusi/jalankan_semua.py</i>, lalu <i>python perbaikan/eksperimen_perbaikan.py</i>, lalu <i>python perbaikan/buat_laporan.py</i> "
  "(dua yang pertama dengan Python dari kernel skripsi-gpu).", "s")
