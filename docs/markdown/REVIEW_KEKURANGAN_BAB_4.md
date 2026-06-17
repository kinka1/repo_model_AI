# Review Kekurangan BAB 4 dan Daftar Perbaikan

Dokumen ini merangkum kekurangan pada BAB 4 (Eksperimen dan Analisis) berdasarkan analisis
`docs/EVAL1-Nufus-Akmalul-3122600026.docx`, dibandingkan dengan data eksperimen riil
(`experiments/ta_resnet50_20260615_021715`, `ta_resnet101_20260615_044118`) dan desain pada BAB 3.

Tingkat keparahan: 🔴 Kritis · 🟠 Sedang · 🟡 Perlu dilengkapi

---

## 1. Inkonsistensi Data 🔴

Ini masalah paling serius karena membuat angka dalam laporan saling bertentangan.

| No | Masalah | Di dokumen | Seharusnya / Data riil |
|----|---------|-----------|------------------------|
| 1.1 | Distribusi kelas kontradiktif | Teks: "Gram Negatif mayoritas, rasio 3,45:1"; Angka: Gram Positif 9.170 vs Gram Negatif 2.654 | Tentukan satu yang benar. Data eksperimen riil: **6.952 neg / 3.748 pos** (sebelum balancing) |
| 1.2 | Total citra berubah-ubah | 11.824 (Tabel Parameter) vs 9.170 (bagian Pembagian Data) | Pilih satu angka konsisten di seluruh bab |
| 1.3 | Rasio split berbeda antar bab | BAB 3: 70:20:10 — BAB 4: 70:15:15 | Samakan; data riil: train 13.904 / val 3.478 |
| 1.4 | Class weight tidak cocok | 0,65 (neg) / 2,23 (pos) | Data riil: **0,7696 (neg) / 1,4274 (pos)** |

> **Catatan:** kelas minoritas pada data riil adalah **Gram Positif**, dan class weight dihitung
> dari distribusi asli sebelum balancing (6.952 neg / 3.748 pos).

---

## 2. Tabel Ringkasan Skenario 5 Bertentangan dengan Detail Per-Model 🔴

Tabel ringkasan akurasi Skenario 5 tidak cocok dengan hasil confusion matrix per-arsitektur di
section yang sama:

| Model | Tabel ringkasan | Detail confusion matrix | Status |
|-------|----------------:|------------------------:|--------|
| ResNet50 | 0,9517 | 0,9517 | ✅ cocok |
| ResNet101 | 0,9688 | **0,9445** | ❌ beda |
| EfficientNet-B3 | 0,9846 | **0,9313** | ❌ beda |
| VGG-16 | 0,8087 | **0,9313** | ❌ beda |
| VGG-19 | 0,8756 | **0,8841** | ❌ beda |
| DenseNet121 | 0,8092 | **0,9310** | ❌ beda |

**Akibat fatal:** kesimpulan di BAB 5 menyatakan *"EfficientNet-B3 model terbaik (98,46%)"*,
padahal berdasarkan detail yang konsisten **ResNet50 (95,17%) adalah yang terbaik**. Tabel ringkasan
harus dihitung ulang dari confusion matrix masing-masing model.

---

## 3. Kesalahan Copy-Paste 🟠

- Paragraf kesimpulan **EfficientNet-B0** menyalin angka ResNet101 ("akurasi 94,45%, recall 95,51%,
  78 citra…") padahal tabel B0 menunjukkan 90,20%.
- **Confusion matrix EfficientNet-B3 identik dengan VGG-16** (TN 1.611 / FP 128 / FN 111 / TP 1.628).
- Paragraf kesimpulan **VGG-19** menulis "akurasi 93,13%" padahal tabelnya 88,41%.
- Tabel spesifikasi Skenario 3: sub-skenario 3b tertulis "ResNet50" (seharusnya **ResNet101**).
- Placeholder teks **"Sad"** dan banyak referensi **"Gambar x.x" / "Tabel 4.2"** yang belum ternomori.

---

## 4. Metrik / Analisis yang Dijanjikan tapi Hilang 🟡

| Dijanjikan di | Belum ada di BAB 4 | Tersedia di data riil? |
|---------------|--------------------|------------------------|
| Metode evaluasi (AUC-ROC) | **ROC-AUC tidak dilaporkan** | Ya — 0,9883 (ResNet50), 0,9882 (ResNet101) |
| BAB 3: "model dipilih berdasarkan akurasi **dan waktu inferensi**" | **Waktu inferensi tidak diukur** | Ya — 2,94 ms (ResNet50), 4,88 ms (ResNet101) |
| — | **Ukuran model / jumlah parameter** tidak dibandingkan konsisten | Ya — 93,99 MB / 166,74 MB |
| — | **Tabel ringkasan akhir yang benar & konsisten** untuk semua arsitektur | Perlu dibuat ulang |

---

## 5. Komponen Sistem yang Belum Diuji 🟡

| Komponen | Dijanjikan di | Status di BAB 4 |
|----------|---------------|-----------------|
| **Deteksi YOLO11** (sistem two-stage: deteksi → crop → klasifikasi) | BAB 3 (inti sistem) | ❌ Tidak ada evaluasi deteksi (mAP, precision/recall deteksi) |
| **Grad-CAM / XAI** | BAB 2 (dibahas panjang) | ❌ Tidak ada visualisasi maupun analisis |
| **Uji data di luar dataset** (generalisasi) | Tujuan sistem (lab nyata) | ❌ Hanya untuk Simple CNN, tidak untuk model terbaik |

---

## 6. Masalah Metodologi 🟠

- **Protokol uji tidak seragam:** Skenario 1–2 diuji dengan "10 citra", Skenario 3–5 dengan
  validation set penuh → tidak bisa dibandingkan apple-to-apple.
- **"Akurasi per gambar" keliru konsep:** nilai 0,633 / 0,712 dst untuk satu citra sebenarnya
  **confidence score**, bukan akurasi (satu citra hasilnya benar atau salah).
- **Tidak ada test set hold-out:** seluruh hasil Skenario 3–5 dievaluasi pada **validation set**
  (`eval_split = val`), bukan test set terpisah seperti dijanjikan di BAB 3.

---

## 7. Inkonsistensi Teknologi (lintas bab) 🟠

Penguji hampir pasti menyoroti ketidakcocokan berikut:

| Aspek | Tertulis di laporan | Implementasi sebenarnya |
|-------|---------------------|-------------------------|
| Framework model | TensorFlow **dan** PyTorch (campur) | PyTorch |
| Backend | Django **dan** FastAPI (campur) | FastAPI |
| Frontend | ReactJS | HTML/CSS/JS statis |
| Versi YOLO | YOLOv8 | YOLO11 |

---

## 8. Yang Perlu Ditambahkan ke BAB 4 (Checklist)

### Prioritas 1 — Wajib (perbaikan konsistensi)
- [ ] Samakan distribusi kelas, total citra, dan rasio split dengan data eksperimen riil.
- [ ] Hitung ulang tabel ringkasan Skenario 5 dari confusion matrix tiap model.
- [ ] Perbaiki kesimpulan BAB 5: model terbaik adalah **ResNet50 (95,17%)**, bukan EfficientNet-B3.
- [ ] Perbaiki paragraf copy-paste (EfficientNet-B0, VGG-16, VGG-19) agar sesuai tabel masing-masing.
- [ ] Samakan istilah teknologi (PyTorch, FastAPI, HTML statis, YOLO11) di seluruh bab.

### Prioritas 2 — Sangat disarankan (memperkuat klaim)
- [ ] Tambahkan kolom **ROC-AUC** pada tabel hasil tiap model.
- [ ] Tambahkan kolom **waktu inferensi** dan **ukuran model / jumlah parameter**.
- [ ] Buat **satu tabel ringkasan akhir** membandingkan semua arsitektur pada metrik yang sama.
- [ ] Tambahkan subbab **Uji Generalisasi pada Data di Luar Dataset** — terapkan ke ResNet50,
      ResNet101, EfficientNet-B0/B3; laporkan per citra (label, prediksi, confidence, benar/salah)
      lalu agregasi (mis. 9/10) + confusion matrix kecil. Minimal 10 citra, idealnya 20–50.
- [ ] Seragamkan protokol evaluasi antar skenario, dan jelaskan bahwa hasil utama pada validation set
      (atau siapkan test set hold-out terpisah).

### Prioritas 3 — Pelengkap (nilai tambah)
- [ ] Tambahkan subbab **Evaluasi Deteksi YOLO11** (mAP, precision/recall deteksi).
- [ ] Tambahkan subbab **Visualisasi Grad-CAM / XAI** untuk model terbaik.
- [ ] Lengkapi penomoran gambar/tabel dan hapus placeholder ("Sad", "Gambar x.x").

---

## 9. Catatan Penting

Versi **markdown** BAB 4 (`docs/markdown/BAB_4_EKSPERIMEN_DAN_ANALISIS.md`) yang sedang dikerjakan
sudah JAUH lebih benar daripada docx EVAL1 ini — sudah memuat ROC-AUC, class weight riil, data
ResNet riil, dan confusion matrix yang konsisten. Jadi kekurangan Prioritas 1 dan sebagian Prioritas 2
sebenarnya **sudah teratasi di markdown**. Yang benar-benar masih kosong di kedua versi:
**uji generalisasi data eksternal, evaluasi YOLO11, dan Grad-CAM.**
