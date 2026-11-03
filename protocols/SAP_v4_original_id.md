# Rencana Analisis Statistik (SAP) v4
## Prediksi status TB/U balita pada bulan t+h dari data e-PPGBM rutin, Kabupaten Buton Utara

Versi: 1.0 (DIBEKUKAN) · Tanggal: 16 Oktober 2026
Status: **disetujui dan dibekukan 16 Oktober 2026**. Setelah disetujui, dokumen ini dibekukan (hash SHA-256 dicatat) sebelum kode dijalankan dan sebelum data uji dibuka. Perubahan setelah pembekuan hanya boleh lewat amandemen bertanggal yang dilaporkan di naskah.

Istilah mengikuti daftar istilah di log keputusan v4: bulan prediksi (t), bulan target (t+h), pasangan prediksi, data pelatihan, data penyetelan threshold, data uji.

---

## 1. Tujuan

Menilai apakah model MCE–ILCADT, yang hanya memakai informasi sampai bulan prediksi t, memprediksi status tinggi badan menurut umur (normal, stunted, severely stunted) pada bulan target t+h lebih baik daripada persistence (status bulan t diteruskan), pada balita yang dipantau rutin di 12 puskesmas Kabupaten Buton Utara.

## 2. Data (Tahap 1, disetujui)

- Sumber: 57 ekspor bulanan e-PPGBM (SIGIZI KESGA), Januari 2022–September 2026, 12 puskesmas, 93 desa, 100 posyandu.
- Dataset master v4: 234.073 baris, 10.321 anak, 230.551 observasi eligible (normal 190.959; stunted 31.106; severely stunted 8.486).
- Identitas anak: ID lengkap dari tautan rekaman, diganti pseudonim bergaram (salt v4, rahasia). 16 anak terduga terdaftar ganda dikeluarkan; 1 anak dengan dua tanggal lahir memakai tanggal modus.
- Observasi eligible sebagai bulan prediksi atau bulan target: kategori TB/U e-PPGBM normal, stunted, atau severely stunted; HAZ −6 sampai +6; umur 0–60 bulan; tinggi badan sah.
- Kelas tall (HAZ > +3; 247 observasi, 164 anak) tidak dimodelkan dan dilaporkan secara deskriptif.
- Label: kategori TB/U e-PPGBM (Permenkes 2/2020; standar WHO 2006). Distribusi kelas termasuk *highly imbalanced multiclass classification* (severely stunted 3,7%).

## 3. Tugas prediksi dan pasangan prediksi (Tahap 2, disetujui)

- Pasangan prediksi: anak yang sama, observasi eligible pada bulan t dan tepat pada bulan t+h. Tidak ada imputasi target. Bulan target di 2022 tidak dipakai; bulan prediksi di 2022 boleh.
- Horizon analisis utama: h = 1, 3, 6. Horizon analisis tambahan: h = 2, 4, 5 (kurva horizon). h = 12 tidak dianalisis.
- Pembagian (sama untuk semua h = 1–6):

| Bagian | Aturan |
|---|---|
| Data pelatihan | Bulan target Januari 2023–Desember 2024 |
| Data penyetelan threshold | Bulan prediksi ≥ Januari 2025 dan bulan target ≤ Desember 2025 |
| Data uji | Bulan prediksi ≥ Januari 2026 (bulan target ≤ September 2026) |
| Dibuang | Pasangan yang melintasi batas bagian |

- Jumlah pasangan prediksi (pelatihan / penyetelan / uji): h1 90.956 / 48.866 / 34.969; h2 86.462 / 43.600 / 29.937; h3 82.489 / 38.405 / 25.071; h4 78.798 / 33.505 / 20.380; h5 74.934 / 28.701 / 15.980; h6 70.932 / 24.157 / 11.650.
- Tujuan evaluasi: pemantauan lanjutan anak yang tercatat pada periode kalender berikutnya.

## 4. Prediktor

Semua prediktor hanya memakai data sampai bulan prediksi t.

**4.1 Prediktor v3 (28)**
- Statis: jenis kelamin, berat lahir, panjang lahir.
- Bulan t: umur (bulan), berat, tinggi, cara ukur (berbaring), LiLA dan penanda LiLA diukur, HAZ, WAZ, WHZ, status TB/U bulan t, kode kenaikan berat badan (N, T, O, B).
- Riwayat pertumbuhan 3 bulan: penanda HAZ teramati 1 dan 3 bulan sebelumnya; perubahan HAZ, WAZ, tinggi, dan berat 1 bulan; perubahan HAZ 3 bulan; jumlah observasi sampai t; jarak bulan sejak observasi sebelumnya; jumlah bulan stunted sebelum t; HAZ terendah sampai t.

**4.2 Riwayat pertumbuhan 12 bulan (baru, 5 kolom)**
- Penanda HAZ teramati 6 dan 12 bulan sebelumnya.
- Perubahan HAZ 6 bulan dan 12 bulan.
- Jumlah bulan stunted dalam 12 bulan terakhir sebelum t.

**4.3 Tidak dipakai**
- Variabel program (vitamin A, KPSP, KIA, kelas ibu, MBG).
- Desa, posyandu, dan puskesmas tidak menjadi prediktor.

## 5. Model

**5.1 Base learner (6)**, hyperparameter sama dengan v3, tidak di-tuning:
RF (400 pohon, kedalaman maks. 15), LightGBM (600; lr 0,05), XGBoost (600; lr 0,05; kedalaman 6), CatBoost (800; lr 0,05; kedalaman 6), Gradient Boosting (100; lr 0,1; kedalaman 3), AdaBoost (300; lr 0,5). Seed 42. Imputasi median dan standardisasi dilakukan di dalam setiap fold. Tanpa resampling dan tanpa pembobotan kelas.

**5.2 Probabilitas out-of-fold**
5 fold dikelompokkan per anak, distratifikasi menurut kelas target terberat anak, dengan syarat minimal 2 anak per kelas per fold. Urutan kelas tetap (normal, stunted, severely stunted). 6 × 3 = 18 meta-feature. Untuk data penyetelan threshold dan data uji, meta-feature adalah rata-rata dari 5 model fold.

**5.3 Meta-DNN (MCE)**
Arsitektur v3: input 18; Dense 64 (ReLU), Dropout 0,30; Dense 32 (ReLU), Dropout 0,20; softmax 3. Sparse categorical cross-entropy; Adam (lr 10⁻³, dikurangi separuh setelah 5 epoch tanpa perbaikan, minimum 10⁻⁶); batch 64; maksimum 200 epoch; early stopping 12 epoch. Epoch dipilih dari loss validasi pada 10% anak data pelatihan (validasi meta), lalu model dilatih ulang pada seluruh anak data pelatihan untuk jumlah epoch tersebut. Seed 42, determinisme TensorFlow aktif.

**5.4 ILCADT**
Skor keputusan s_k = σ(logit p_k − logit τ_k), dinormalisasi; kelas = argmax s_k (selalu tepat satu kelas; seri ke indeks kelas terendah). τ_k dicari per kelas pada grid 0,01–0,99 (langkah 0,01) dengan memaksimalkan F1 one-vs-rest pada data penyetelan threshold. Tie-break: recall lebih tinggi, lalu precision lebih tinggi, lalu paling dekat 0,5, lalu threshold lebih rendah. Threshold v3 tidak dipakai. Skor s_k bukan probabilitas dan tidak dipakai untuk kalibrasi.

## 6. Pembanding dan ablasi komponen

Dievaluasi pada pasangan uji yang sama:

| Konfigurasi | Isi |
|---|---|
| Persistence | Status bulan t = prediksi untuk bulan target |
| Best base learner | Base learner dengan macro-F1 (argmax) tertinggi pada data penyetelan threshold. Seri diputus menurut urutan RF, LGBM, XGB, CAT, GBC, ADA. Dipilih sebelum data uji dibuka. |
| Best base + ILCADT | Base learner terpilih dengan threshold per kelas dari data penyetelan threshold |
| MCE | Meta-DNN, argmax |
| MCE + ILCADT | Model lengkap |

Additional file: keenam base learner, masing-masing argmax dan + ILCADT.

## 7. Evaluasi (data uji, per horizon)

**7.1 Metrik utama:** macro-F1.

**7.2 Metrik pendukung**
- Accuracy, balanced accuracy, macro-precision, macro-recall.
- Precision, recall, F1, dan support per kelas; confusion matrix.
- Minority macro-recall (rata-rata recall stunted dan severely stunted) dan minimum minority recall. Kelas minoritas ditetapkan dari data pelatihan.
- Untuk model probabilistik: ROC-AUC one-vs-rest makro, negative log-likelihood, Brier multiclass, ECE (top-label dan per kelas).

**7.3 Kalibrasi per kelas (MCE)**
Calibration-in-the-large dan calibration slope dengan CI yang memperhitungkan klaster anak. Kurva kalibrasi fleksibel hanya bila kelas memiliki ≥200 kejadian dan ≥200 non-kejadian (Van Calster dkk., 2019).

**7.4 Subset evaluasi**
- Seluruh data uji.
- Pasangan yang statusnya berubah antara t dan t+h.
- Anak baru (tidak muncul di data pelatihan).
- Per puskesmas (12), dari prediksi data uji utama tanpa pelatihan ulang.

**7.5 Ketidakpastian**
Bootstrap klaster per anak, 2.000 ulangan, CI persentil 95%. Selisih berpasangan dihitung untuk: MCE + ILCADT − persistence; MCE − persistence; MCE + ILCADT − MCE; best base + ILCADT − best base; MCE − best base. Tidak ada uji hipotesis formal; kesimpulan didasarkan pada besar selisih dan CI-nya.

**7.6 Metrik tak terdefinisi**
Bila sebuah kelas tidak muncul pada subset, metrik kelas tersebut dilaporkan "tidak dapat dihitung" (bukan 0), dan rata-rata makro dihitung dari kelas yang ada dengan catatan.

## 8. Analisis sensitivitas (h = 1, 3, 6)

1. **Label WHO:** label dari HAZ hitung ulang dengan tabel WHO 2006, menggantikan kategori e-PPGBM.
2. **Tanpa status bulan t:** status TB/U dan HAZ bulan t dikeluarkan dari prediktor (persistence tetap memakai status bulan t).
3. **Tanpa observasi bertanda:** observasi dengan penurunan tinggi > 1,5 cm dari bulan sebelumnya tidak dipakai sebagai bulan prediksi maupun bulan target.

Setiap analisis sensitivitas mengulang seluruh pipeline dengan satu perubahan saja.

## 9. Analisis tambahan

- Kurva horizon h = 1–6: macro-F1 persistence, MCE, dan MCE + ILCADT dengan 95% CI per horizon (Additional file).
- Laporan deskriptif kelas tall.
- Validasi antar-puskesmas 12 fold: ditunda, diputuskan setelah run utama.

## 10. Laporan data dan keterbatasan yang sudah direncanakan

- Diagram alur dan tabel jumlah pasangan per bagian data, horizon, dan kelas.
- Audit attrition: perbandingan status anak dengan dan tanpa bulan target.
- Ketersediaan riwayat pertumbuhan 12 bulan pada data pelatihan vs data uji (48,9% vs 73,0% untuk h = 1).
- Anak di data uji yang juga muncul di data pelatihan.
- Kualitas data 2022 (dipakai hanya sebagai bulan prediksi atau riwayat).

## 11. Pembekuan dan reproduktibilitas

1. Kode diaudit per tahap (Tahap 4) sebelum run.
2. Model, threshold, best base learner terpilih, meta-feature uji, dan kode di-hash (SHA-256) dan dicatat di FREEZE_LOG_v4 **sebelum** metrik uji dihitung.
3. Data uji dibuka satu kali. Semua hasil dilaporkan, termasuk yang tidak mendukung model.
4. Versi Python dan pustaka dicatat. Run ulang independen di lingkungan terpisah diulang seperti pada v3.

## 12. Penyimpangan dari v3 (dilaporkan di naskah)

- Dataset: 12 puskesmas (sebelumnya 1), Januari 2022–September 2026 (sebelumnya Januari 2024–Agustus 2026).
- Identitas dari tautan rekaman (kolom NIK disamarkan pada ekspor baru).
- Semua horizon 1–6 memakai data penyetelan threshold temporal (di v3, h = 6 memakai 20% anak data pelatihan).
- Prediktor ditambah riwayat pertumbuhan 12 bulan.
- Ablasi komponen dan kinerja per puskesmas ditambahkan.
- Rolling origin dan validasi antar-desa tidak dilakukan.
