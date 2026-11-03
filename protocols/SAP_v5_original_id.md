# Rencana Analisis Statistik (SAP) v5
## Deteksi perubahan status TB/U balita yang terkonfirmasi (bulan t → t+h), Kabupaten Buton Utara

Versi: 1.0 (DIBEKUKAN) · 17 Oktober 2026 · disetujui pengguna 17 Oktober 2026 (isi identik dengan draf v0.3, SHA-256 d2c8d853494b0d9c45987d6a4fd2fd208538ada34e7d60a679682fb988d50df4)
Revisi dari v0.1 dan v0.2 berdasarkan dua pemeriksaan independen (claude/v5_independent_review.md) dan keputusan pengguna 17 Oktober 2026: (1) status bulan t dikonfirmasi oleh bulan t−1; (2) endpoint utama stunting baru terkonfirmasi pada anak normal, metrik utama AUPRC; (3) kontribusi hanya diklaim bila konsisten di h = 1, 3, dan 6.
Dokumen ini dibekukan (SHA-256 dicatat di FREEZE_LOG_v5) SEBELUM kode v5 ditulis dan dijalankan. Perubahan hanya melalui amandemen bertanggal yang dilaporkan di naskah. Istilah mengikuti log keputusan v4.

---

## 1. Tujuan
Menilai apakah arsitektur MCE–ILCADT (paper 1), yang hanya memakai informasi sampai bulan prediksi t, dapat mendeteksi anak yang akan mengalami perubahan status TB/U terkonfirmasi pada bulan t+h — terutama stunting baru pada anak yang normal — lebih baik daripada pembanding yang lebih sederhana.

Dasar: pada v4, model status masa depan tidak melampaui persistence; 89–95% anak tidak berubah status dan persistence tidak pernah mendeteksi perubahan (claude/v4_results_summary.md).

## 2. Status pembuktian
- Data uji 2026 telah dibuka pada v4. Semua hasil v5 pada data uji 2026 dilaporkan sebagai **eksploratif**; tidak ada klaim kontribusi dari data ini.
- **Konfirmatif:** model v5 yang dibekukan diterapkan tanpa pelatihan ulang pada data dengan bulan prediksi ≥ Oktober 2026. Ukuran sampel dihitung dengan metode Riley dkk. 2024 untuk endpoint utama (biner) dan simulasi bootstrap daya aturan klaim (bagian 9), dan protokolnya ditulis sebagai amandemen bertanggal sebelum data baru dibuka. Untuk h = 6, konfirmasi baru mungkin setelah data sekitar Mei 2027 tersedia.

## 3. Data, populasi, dan target
- Dataset master v4 (234.073 baris; 10.321 anak), aturan eligibilitas v4, label kategori TB/U e-PPGBM (Permenkes 2/2020).
- Horizon: h = 1, 3, 6.
- **Pasangan:** anak yang sama dengan observasi eligible pada bulan t−1, t, t+h, dan t+h+1 (tepat), dan **status bulan t−1 sama dengan status bulan t** (status awal terkonfirmasi). Tanpa imputasi.
- **Populasi sasaran:** balita yang dipantau rutin dengan status stabil pada dua bulan terakhir dan masih hadir pada bulan t+h dan t+h+1. Pasangan tanpa observasi t+h+1 dilaporkan dalam tabel attrition (status bulan t dan arah perubahan tanpa konfirmasi), seperti pada v4.

### 3.1 Target
Urutan keparahan: normal (0) < stunted (1) < severely stunted (2). d1 = tanda(s(t+h) − s(t)); d2 = tanda(s(t+h+1) − s(t)).
- **Memburuk** bila d1 = d2 = +1; **membaik** bila d1 = d2 = −1; selain itu **tetap** (termasuk perubahan sesaat).
- Urutan kelas di kode: tetap (0), memburuk (1), membaik (2).
- **Endpoint utama:** stunting baru terkonfirmasi = memburuk pada anak normal di bulan t (biner, dievaluasi pada anak normal di bulan t).
- Observasi t+h dan t+h+1 hanya dipakai untuk label. Syarat t−1 memakai informasi masa lalu (tersedia pada bulan t), sehingga tidak bocor.

Dasar: perubahan kategori satu kali dapat berasal dari variasi pengukuran dan regresi ke rerata, baik pada pengukuran awal maupun akhir (Barnett dkk. 2005; Roth dkk. 2025; Leroy dkk. 2025; Cameron dkk. 2005; Cliffer dkk. 2022); pemodelan selisih pengukuran berulang memakai asumsi noise panjang badan sekitar 0,5 cm (Wright dkk. 2022); reliabilitas pengukuran panjang/tinggi pada WHO MGRS (WHO MGRS Group 2006); konsep stunting persisten lintas pengukuran (Alam dkk. 2020); pengecualian regresi ke rerata dalam definisi catch-up (Desmond & Casale 2017). Data: pada pelatihan h = 1 (tanpa syarat t−1), 43,4% "membaik" dan 29,0% "memburuk" terkonfirmasi berasal dari status bulan t yang baru berubah dari t−1, dibanding 4,1% pada kelas tetap.

### 3.2 Pembagian (blok waktu; bulan t−1 boleh sebelum blok, bulan konfirmasi di blok yang sama)
| Bagian | Aturan |
|---|---|
| Data pelatihan | Bulan target ≥ Januari 2023 dan bulan konfirmasi (t+h+1) ≤ Desember 2024 |
| Data penyetelan threshold | Bulan prediksi ≥ Januari 2025 dan bulan konfirmasi ≤ Desember 2025 |
| Data uji (eksploratif) | Bulan prediksi ≥ Januari 2026 (bulan konfirmasi ≤ September 2026) |

Jumlah pasangan (pelatihan / penyetelan / uji):
| h | Pasangan | Tetap | Memburuk | Membaik |
|---|---|---|---|---|
| 1 | 69.113 / 38.791 / 26.115 | 67.538 / 37.997 / 25.517 | 775 / 446 / 361 | 800 / 348 / 237 |
| 3 | 61.585 / 29.499 / 17.566 | 58.634 / 28.195 / 16.728 | 1.461 / 761 / 511 | 1.490 / 543 / 327 |
| 6 | 51.172 / 17.336 / 6.494 | 47.534 / 16.039 / 6.035 | 1.736 / 780 / 307 | 1.902 / 517 / 152 |

Endpoint utama — anak normal di bulan t (pasangan; stunting baru; insiden):
| h | Pelatihan | Penyetelan | Uji |
|---|---|---|---|
| 1 | 59.427; 610 (561 anak); 1,03% | 33.076; 353; 1,07% | 22.280; 294 (293 anak); 1,32% |
| 3 | 53.075; 1.156 (618 anak); 2,18% | 25.210; 624; 2,48% | 15.006; 427 (264 anak); 2,85% |
| 6 | 44.271; 1.439 (553 anak); 3,25% | 14.853; 666; 4,48% | 5.555; 270 (186 anak); 4,86% |

Catatan: data uji h = 6 hanya mencakup bulan prediksi Januari–Februari 2026. Distribusi tiga kelas termasuk *highly imbalanced multiclass classification* (kelas minoritas 0,9–4,7% di seluruh bagian data; 1,1–3,7% di data pelatihan).

## 4. Prediktor
33 prediktor v4 (SAP v4 bagian 4), semuanya dari data sampai bulan t, termasuk status dan HAZ bulan t. Variabel program, desa, posyandu, dan puskesmas tidak dipakai sebagai prediktor.

## 5. Model (arsitektur paper 1; dipertahankan dari v4)
- 6 base learner, hyperparameter v4, tanpa tuning, tanpa resampling dan tanpa pembobotan kelas.
- OOF 5 fold per anak. Stratum anak = kelas arah perubahan paling jarang yang dialami anak (urutan kelangkaan dari data pelatihan). Anak per stratum (tetap / memburuk / membaik-saja): h1 5.185 / 682 / 363; h3 4.836 / 742 / 415; h6 4.150 / 650 / 508. Setiap fold wajib memuat ≥ 2 anak untuk kelas memburuk dan membaik.
- 6 × 3 = 18 meta-feature; Meta-DNN dan aturan epoch v4; ILCADT v4 (threshold per kelas dari data penyetelan, grid 0,01–0,99, F1 one-vs-rest, tie-break v4).
- **Masker kelas mustahil** (semua model dan pembanding): untuk anak normal di bulan t, P(membaik) = 0; untuk anak severely stunted, P(memburuk) = 0; probabilitas dinormalisasi ulang sebelum metrik, sebelum threshold ILCADT dipelajari, dan masker diterapkan lagi pada skor keputusan ILCADT akhir sehingga kelas mustahil tidak pernah terpilih.

Dasar tanpa resampling: koreksi ketidakseimbangan kelas memperburuk kalibrasi tanpa menaikkan AUC, dan pergeseran threshold memberi sensitivitas/spesifisitas serupa (van den Goorbergh dkk. 2022); kerusakan kalibrasi juga terjadi pada model machine learning (Carriero dkk. 2025).

## 6. Pembanding (pada pasangan uji yang sama)
| Pembanding | Definisi | Dasar |
|---|---|---|
| Persistence | Selalu "tetap" (untuk endpoint utama: tidak memberi peringkat risiko; AUPRC = insiden) | Pembanding naif |
| Model v4 dipetakan | Probabilitas Meta-DNN v4 beku: P(memburuk) = Σ_{k>s} p_k, P(tetap) = p_s, P(membaik) = Σ_{k<s} p_k; dicocokkan per (anak, bulan t) dengan pemeriksaan sidik jari; argmax dan ILCADT. AUPRC (endpoint utama) memakai pemetaan mentah. Varian dikalibrasi ulang (rekalibrasi logistik multinomial, hanya data penyetelan) dipakai khusus untuk metrik probabilitas (NLL, Brier, ECE, kalibrasi, DCA) | Model status v4 secara matematis memuat probabilitas arah perubahan (tanpa konfirmasi); menguji apakah pelatihan khusus menambah informasi |
| Regresi logistik sederhana | Status bulan t sebagai dua variabel indikator 0/1 (acuan normal), HAZ t dan umur t dengan restricted cubic spline (4 knot), interaksi status × HAZ t, perubahan HAZ 1 bulan + penanda teramati; penalti L2 (C = 1,0) pada prediktor terstandardisasi, solver lbfgs; imputasi median dari data pelatihan; argmax dan ILCADT | Christodoulou dkk. 2019; Clift dkk. 2023; Leroy dkk. 2025 (LAZ terakhir prediktor terkuat); pengodean prediktor kategorik dan spline: Steyerberg 2019 |
| Regresi logistik 33 prediktor | 33 prediktor yang sama dengan MCE; penalti L2 (C = 1,0), standardisasi dan imputasi seperti di atas; argmax dan ILCADT | Perbandingan setara (prediktor sama, algoritma berbeda; Christodoulou dkk. 2019) |
| Ablasi | Best base learner (± ILCADT), MCE (± ILCADT) | Sama dengan v4 |

Penalti L2 dipakai karena kelas mustahil menimbulkan pemisahan sempurna pada regresi logistik tanpa penalti.

## 7. Evaluasi (data uji, per horizon)
**7.1 Metrik utama:** AUPRC stunting baru terkonfirmasi pada anak normal di bulan t, dari probabilitas P(memburuk) (Saito & Rehmsmeier 2015), dihitung sebagai *average precision* bertingkat (scikit-learn `average_precision_score`), bukan interpolasi trapesium yang melebih-lebihkan luas kurva precision–recall (Davis & Goadrich 2006). Nilai acuan tanpa kemampuan = insiden.

**7.2 Metrik sekunder**
- Endpoint utama: AUC, sensitivitas dan PPV bila 5% dan 10% anak normal dengan risiko tertinggi dipantau, net benefit (decision curve analysis, rentang threshold risiko 0,5–10%, sama untuk ketiga horizon; Vickers & Elkin 2006; Van Calster dkk. 2018). Tindakan yang dinilai adalah pemantauan lebih ketat (kunjungan atau konseling tambahan di posyandu), yang murah dan berisiko rendah; karena itu threshold rendah relevan. Threshold 10% berarti petugas bersedia memantau hingga 9 anak tanpa kejadian untuk menemukan 1 anak yang akan mengalami stunting baru.
- Kalibrasi endpoint utama (CITL, slope; kurva bila ≥ 200 kejadian dan ≥ 200 non-kejadian; Van Calster dkk. 2019).
- Tiga kelas (kesinambungan dengan paper 1): macro-F1, recall/precision/F1 per kelas, balanced accuracy, minority macro-recall, minimum minority recall, confusion matrix, AUC one-vs-rest, NLL, Brier, ECE.
- Strata status bulan t: anak stunted (memburuk dan membaik), anak severely stunted (membaik).
- Subgrup deskriptif (TRIPOD+AI): jenis kelamin, kelompok umur (< 24 dan ≥ 24 bulan), puskesmas.
- DCA dan metrik probabilitas memakai probabilitas, tidak pernah skor ILCADT.

**7.3 Ketidakpastian:** bootstrap klaster per anak, 2.000 ulangan, CI persentil 95%. Selisih berpasangan AUPRC (endpoint utama): MCE − model v4 dipetakan; MCE − regresi logistik sederhana; MCE − regresi logistik 33 prediktor. Selisih sekunder (macro-F1 tiga kelas, pasangan uji yang sama): MCE+ILCADT − model v4 dipetakan+ILCADT; MCE+ILCADT − regresi logistik sederhana+ILCADT; MCE+ILCADT − regresi logistik 33 prediktor+ILCADT; MCE+ILCADT − MCE; best base+ILCADT − best base.

**7.4 Aturan klaim (ditetapkan sebelum dijalankan)**
- Kontribusi MCE (probabilitas Meta-DNN; threshold ILCADT tidak memengaruhi AUPRC) untuk deteksi stunting baru diklaim hanya bila, pada **data konfirmatif**, CI 95% selisih AUPRC MCE terhadap **ketiga** pembanding (model v4 dipetakan, regresi logistik sederhana, regresi logistik 33 prediktor) berada seluruhnya di atas nol **pada h = 1, 3, dan 6** (aturan intersection-union; tidak perlu koreksi multiplisitas).
- Pada data uji 2026 (eksploratif), hasil dilaporkan sebagai "konsisten" atau "tidak konsisten" dengan aturan ini, tanpa klaim kontribusi.
- Bila aturan tidak terpenuhi, dilaporkan bahwa model tidak terbukti lebih baik daripada pembanding yang lebih sederhana.

**7.5 Metrik tak terdefinisi:** seperti SAP v4 7.6. Precision kelas yang tidak pernah diprediksi = 0 bila kelas ada pada data (berlaku untuk persistence pada kelas memburuk/membaik), dan dicatat.

## 8. Analisis sensitivitas (h = 1, 3, 6)
1. Tanpa konfirmasi bulan t: syarat observasi t−1 dan status t−1 = status t dihapus; konfirmasi t+h+1 tetap.
2. Tanpa konfirmasi sama sekali: arah perubahan dari t ke t+h saja; syarat t−1 dan t+h+1 dihapus (pasangan v4).

## 9. Ukuran sampel
- Pengembangan: kecukupan ukuran sampel untuk endpoint utama dinilai dengan kriteria Riley dkk. 2020 (outcome biner) dan dilaporkan.
- Konfirmatif, dua bagian, ditulis sebagai amandemen sebelum data baru dibuka: (a) presisi kinerja endpoint utama (kalibrasi, C-statistic, net benefit) dengan Riley dkk. 2024; (b) daya aturan klaim 7.4 melalui simulasi bootstrap: jumlah bulan data konfirmatif yang diperlukan agar kesembilan CI selisih AUPRC (3 pembanding × 3 horizon) dapat mengecualikan nol, dengan asumsi selisih yang ditetapkan dari data penyetelan (bukan data uji). Horizon h = 6 menjadi penentu karena kejadiannya paling sedikit.

## 10. Pembekuan dan reproduktibilitas
Seperti SAP v4 bagian 11: pelaporan mengikuti TRIPOD+AI (Collins dkk. 2024) dan risiko bias dinilai sendiri dengan PROBAST+AI (Moons dkk. 2025); audit kode per tahap oleh agen independen; freeze manifest dan FREEZE_LOG_v5 sebelum metrik uji; lingkungan sama dengan paper 1 (runtime Colab 2026.04, Python 3.12.13, TensorFlow 2.20.0, scikit-learn 1.6.1, XGBoost 3.1.2, LightGBM 4.6.0, CatBoost 1.2.8); run ulang independen.

## 11. Keterbatasan yang sudah direncanakan
- Hasil 2026 eksploratif; konfirmasi menunggu data baru (h = 6 sekitar Mei 2027).
- Syarat observasi t−1 dan t+h+1 mempersempit populasi ke anak yang dipantau rutin; data uji h = 6 kecil.
- Kelas membaik dapat tetap dipengaruhi regresi ke rerata walau terkonfirmasi.
- Tidak ada prediktor penyebab perubahan (penyakit, asupan, sosial-ekonomi) di e-PPGBM.

## Referensi baru (belum di bibliografi naskah)
- Barnett AG, van der Pols JC, Dobson AJ. Regression to the mean: what it is and how to deal with it. Int J Epidemiol. 2005;34(1):215–220.
- van den Goorbergh R, van Smeden M, Timmerman D, Van Calster B. The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression. J Am Med Inform Assoc. 2022;29(9):1525–1534. doi:10.1093/jamia/ocac093
- Carriero A, Luijken K, de Hond A, Moons KGM, Van Calster B, van Smeden M. The harms of class imbalance corrections for machine learning based prediction models: a simulation study. Stat Med. 2025;44(3–4):e10320. doi:10.1002/sim.10320
- Christodoulou E, Ma J, Collins GS, Steyerberg EW, Verbakel JY, Van Calster B. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. J Clin Epidemiol. 2019;110:12–22. doi:10.1016/j.jclinepi.2019.02.004
- Davis J, Goadrich M. The relationship between precision-recall and ROC curves. In: Proceedings of the 23rd International Conference on Machine Learning (ICML). 2006. p. 233–240.
- Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS One. 2015;10(3):e0118432.
- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26(6):565–574. doi:10.1177/0272989X06295361
- Van Calster B, Wynants L, Verbeek JFM, Verbakel JY, Christodoulou E, Vickers AJ, Roobol MJ, Steyerberg EW. Reporting and interpreting decision curve analysis: a guide for investigators. Eur Urol. 2018;74(6):796–804. doi:10.1016/j.eururo.2018.08.038
- Riley RD, Ensor J, Snell KIE, Harrell FE, Martin GP, Reitsma JB, Moons KGM, Collins G, van Smeden M. Calculating the sample size required for developing a clinical prediction model. BMJ. 2020;368:m441. doi:10.1136/bmj.m441
- Riley RD, Snell KIE, Archer L, Ensor J, Debray TPA, Van Calster B, van Smeden M, Collins GS. Evaluation of clinical prediction models (part 3): calculating the sample size required for an external validation study. BMJ. 2024;384:e074821. doi:10.1136/bmj-2023-074821
- Wright CM, Haig C, Harjunmaa U, Sivakanthan H, Cole TJ. Assessing the optimal time interval between growth measurements using a combined data set of weights and heights from 5948 infants. Arch Dis Child. 2022;107(4):341–345. doi:10.1136/archdischild-2021-322479
- WHO Multicentre Growth Reference Study Group. Reliability of anthropometric measurements in the WHO Multicentre Growth Reference Study. Acta Paediatr Suppl. 2006;450:38–46. doi:10.1111/j.1651-2227.2006.tb02374.x
Referensi lain (Roth 2025, Leroy 2025, Cameron 2005, Cliffer 2022, Alam 2020, Desmond 2017, Clift 2023, Van Calster 2019, Steyerberg 2019, TRIPOD+AI 2024, PROBAST+AI 2025) sudah ada di bibliografi naskah.
