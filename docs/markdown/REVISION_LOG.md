# Revision Log

Sumber: `docs/EVAL1-Nufus-Akmalul-3122600026.docx`

## Global
- Memisahkan dokumen DOCX menjadi Markdown per BAB di `docs/markdown/`.
- Merapikan heading utama dan subheading menjadi format Markdown.
- Merapikan sebagian ejaan/spasi yang terbaca menempel dari hasil ekstraksi DOCX.
- Melakukan pembersihan lanjutan pada kata yang tersambung dan artefak spasi hasil ekstraksi di BAB 1 sampai BAB 5.
- Mempertahankan angka hasil eksperimen, nilai akurasi, tabel metrik, dan daftar pustaka dari dokumen asli.

## Revisi Kesesuaian Project Aktual
- Mengganti penyebutan backend `Django` menjadi `FastAPI` sesuai implementasi di `app/main.py` dan router `/api/...`.
- Mengganti framework model `TensorFlow` menjadi `PyTorch` sesuai implementasi `torch`, `torchvision`, dan file model `.pth`.
- Mengganti penyebutan `YOLOv8` menjadi `YOLO11` / Ultralytics YOLO sesuai model dan script training YOLO di project.
- Mengganti frontend `ReactJS` menjadi `HTML/CSS/JS` statis sesuai folder `frontend/` saat ini.
- Menormalkan penyebutan database menjadi `PostgreSQL`.

## BAB 3 - Desain Sistem
- Menyesuaikan narasi backend menjadi FastAPI sebagai penghubung frontend, database, dan pipeline AI.
- Menyesuaikan narasi model AI menjadi PyTorch classifier dan Ultralytics YOLO detection.
- Menyesuaikan alur sistem dengan pipeline aktual: upload spesimen, YOLO detection, auto crop, CNN classification, simpan hasil, validasi dokter.

## BAB 4 - Eksperimen dan Analisis
- Angka hasil eksperimen dipertahankan sesuai dokumen asli.
- Istilah arsitektur diperbaiki secara ringan, misalnya `RestNet` menjadi `ResNet` dan `efficientnet` menjadi `EfficientNet`.
- Menambahkan penjelasan tambahan terkait baseline Simple CNN, perbandingan karakter arsitektur CNN, serta pengaruh transfer learning dan fine-tuning tanpa mengubah angka hasil eksperimen.
- Menghapus kolom dan narasi `rata-rata` pada hasil skenario, lalu menggantinya dengan analisis model terbaik, model terendah, rentang performa, dan karakter arsitektur.
- Menambahkan template confusion matrix, template analisis confusion matrix per model, metrik turunan dari confusion matrix, serta bagian loss function sebagai landasan pengisian hasil evaluasi berikutnya.
- Catatan: apabila ada perbedaan angka dengan hasil training terbaru di repository, angka pada Markdown ini tetap mengikuti dokumen asli sesuai instruksi pengguna.

## Penambahan Detail Arsitektur CNN
- BAB 2 ditambahkan teori arsitektur CNN, fungsi layer utama, serta ringkasan VGG, ResNet, EfficientNet, dan DenseNet.
- BAB 3 ditambahkan detail implementasi CNN pada sistem, termasuk input 224x224, output dua kelas, hubungan YOLO11 dengan CNN, dan alur klasifikasi.
- BAB 4 ditambahkan analisis pemilihan arsitektur CNN serta dampak transfer learning dan fine-tuning terhadap performa model.

## Tidak Diubah
- Nilai akurasi, precision, recall, F1-score, AUC, dan metrik eksperimen.
- Substansi kesimpulan numerik.
- Daftar pustaka.

## Perbaikan Konsistensi Lanjutan
- BAB 1: Memperbaiki tanda baca rumusan masalah ke-2 (`:` menjadi `?`), serta menambahkan deskripsi BAB 4 dan BAB 5 pada bagian Sistematika Penulisan.
- BAB 2: Menghapus heading Markdown (`##`) yang nyangkut di tengah paragraf (artefak ekstraksi DOCX).
- BAB 3: Mengganti teks placeholder "YANG SAYA KERJAKAN" sebelum Gambar 3.1 dengan kalimat pengantar yang sesuai.
- BAB 3: Memperbaiki deskripsi tabel `ai_model` yang sebelumnya merupakan duplikat dari deskripsi tabel `classifications`, disesuaikan dengan field pada `AIModelSummaryResponse` di `app/schemas.py`.
- BAB 3: Menyamakan Strategi Validasi (80:10:10 -> 70:15:15 stratified split) dan daftar arsitektur pada Tabel Skenario Uji Coba agar konsisten dengan Tabel Skenario dan hasil eksperimen di BAB 4.
- BAB 4: Memperbaiki total dataset pada narasi Pembagian Data (9.170 -> 11.824) agar konsisten dengan Tabel 4.2/4.3.
- BAB 4: Memperbaiki pertukaran label kelas mayoritas/minoritas (Gram Positif/Gram Negatif) pada narasi class imbalance, class weights, dan sharpness-based filtering agar konsisten dengan Tabel 4.2.
- BAB 4: Menambahkan kalimat isi pada bagian TEMPAT UJICOBA (ujicoba dilakukan mandiri menggunakan laptop pribadi).
- Daftar Pustaka: Memisahkan dua referensi (Limkriangkrai & Preechayasomboon 2023; Smith, Kang, & Kirn 2018) yang sebelumnya tergabung dalam satu baris.
- BAB 4: Menambahkan heading `## Skenario 1` (sebelumnya tidak memakai format heading seperti Skenario 2-5) dan menghapus heading Markdown (`##`) yang nyangkut di tengah kalimat pengantar Tabel 4.8 dan Tabel 4.9 (artefak ekstraksi DOCX, sama seperti temuan di BAB 2).

## Pendalaman Metode (Skenario, Arsitektur, Confusion Matrix, Loss Function)
- BAB 4: Menghapus catatan "(rata-rata 8,12 KB)" pada Karakteristik Data sesuai permintaan pengguna.
- BAB 4: Memperbaiki Tabel 4.7 baris sub-skenario 3b yang sebelumnya tertulis backbone "ResNet50" (duplikat dari 3a) menjadi "ResNet101", sesuai heading "Skenario 3b: ResNet101" dan jumlah total parameter (43.550.274) yang memang lebih besar dari ResNet50.
- BAB 4 - Skenario 2: Menambahkan subbab "Spesifikasi Arsitektur dan Konfigurasi Training" (arsitektur identik Skenario 1 + augmentasi on-the-fly, optimizer Adam lr 0,001, epoch maksimum 40).
- BAB 4 - Skenario 4: Menambahkan subbab "Spesifikasi Konfigurasi Training dan Strategi Fine-Tuning Skenario 4" yang menjelaskan proses training dua fase (head training dengan backbone frozen, lalu fine-tuning sebagian layer terakhir), optimizer AdamW (lr 0,001 pada fase head, 0,0001 pada fase fine-tuning), weighted CrossEntropyLoss, scheduler ReduceLROnPlateau, dan early stopping, beserta layer yang dibuka per arsitektur (ResNet: layer4; EfficientNet: beberapa block terakhir + conv_head).
- BAB 4 - Skenario 5: Menambahkan subbab "Spesifikasi Arsitektur Tambahan Skenario 5" berisi total parameter dan strategi fine-tuning untuk VGG-16, VGG-19, dan DenseNet121, serta menegaskan bahwa ketujuh arsitektur menggunakan konfigurasi training dua fase yang sama seperti Skenario 4.
- BAB 4: Menambahkan section baru "Perbandingan Karakteristik Arsitektur CNN" berisi tabel perbandingan kedelapan arsitektur (kedalaman/mekanisme, total parameter, karakteristik untuk citra Gram-stain) beserta analisis pengelompokan karakter arsitektur (residual connection, compound scaling, dense connection/VGG).
- BAB 4: Mengubah "Template Confusion Matrix" dan "Template Loss Function" dari template tunggal generik menjadi template per-arsitektur (8 tabel confusion matrix kosong dan 1 tabel ringkasan loss per arsitektur untuk Simple CNN, ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3, VGG-16, VGG-19, DenseNet121), agar nilai TP/TN/FP/FN dan train/val loss dapat diisi manual oleh pengguna setelah evaluasi tersedia. Tidak ada angka hasil eksperimen pada Tabel 4.8-4.12 yang diubah.
- BAB 2: Memperdalam teori arsitektur CNN - menambahkan subbab "Simple CNN" (arsitektur baseline) dan "Perbandingan Arsitektur CNN", serta memperluas penjelasan VGG (jumlah layer, ~138/143 juta parameter), ResNet (formula residual block, bottleneck, jumlah parameter ResNet50/101), EfficientNet (MBConv, compound scaling, jumlah parameter B0/B3), dan DenseNet (struktur dense block 6-12-24-16, transition layer, ~8 juta parameter).

## Penambahan Desain Arsitektur CNN di BAB 3
- BAB 3: Menambahkan section baru "Desain Arsitektur CNN" pada bagian Lapisan Model AI, berisi penjelasan dua kelompok arsitektur yang digunakan (Simple CNN baseline dan transfer learning berbasis backbone pretrained).
- BAB 3: Menambahkan Tabel 3.2 "Desain Arsitektur Simple CNN" yang merinci layer-by-layer (5 convolutional layer + max pooling, channel progression 3->32->64->128->256->512, flatten 25.088, 3 fully connected layer 25.088->1.024->512->2, ReLU dan Softmax) beserta output shape tiap layer, konsisten dengan spesifikasi Skenario 1 di BAB 4.
- BAB 3: Menambahkan Tabel 3.3 "Desain Umum Arsitektur Transfer Learning" yang menjelaskan struktur backbone (pretrained ImageNet, frozen/fine-tuned) + classifier head baru untuk 2 kelas (Gram-positif/Gram-negatif), serta referensi ke detail strategi fine-tuning per arsitektur di BAB 4.
- Menambahkan diagram blok arsitektur Simple CNN (`docs/images/simple_cnn_architecture.png`, dibuat via `docs/generate_simple_cnn_diagram.py`) yang menggambarkan alur Input -> 5x (Conv+ReLU+MaxPool) -> Flatten -> 3x Fully Connected -> Output, lengkap dengan ukuran channel dan output shape tiap layer.
- BAB 3: Menyisipkan diagram tersebut pada subbab "Desain Arsitektur Simple CNN".
- BAB 4 - Skenario 1: Mengganti placeholder teks "Gambar 4.2 diagram arsitektur simple cnn" dengan diagram arsitektur yang sama.

## Pengisian Data Evaluasi ResNet50 (experiments/ta_resnet50_20260615_021715)
- BAB 4 - Tabel 4.X.2 (Confusion Matrix Model ResNet50): Diisi dengan hasil evaluasi model ResNet50 fine-tuning pada data validasi (3.478 citra: 1.739 Gram-negatif, 1.739 Gram-positif) dari `experiments/ta_resnet50_20260615_021715/metrics_summary.json` - TN=1.649, FP=90, FN=78, TP=1.661. Ditambahkan catatan bahwa data ini merupakan evaluasi tambahan/pelengkap, terpisah dari pengujian 10 citra pada Tabel 4.10, sesuai kesepakatan sebelumnya.
- BAB 4: Menambahkan gambar `images/resnet50_confusion_matrix.png` dan `images/resnet50_training_curves.png` (disalin dari folder experiments) beserta subbab analisis "Analisis Confusion Matrix Model ResNet50 (Hasil Evaluasi)" dan "Analisis Loss Model ResNet50 (Hasil Evaluasi)" sebagai instansiasi template dengan data nyata.
- BAB 4 - Tabel 4.Y (Ringkasan Loss per Arsitektur): Mengisi baris ResNet50 - Best Epoch 31, Train Loss 0,1054, Validation Loss 0,1466, Status "Konvergen (indikasi mild overfitting setelah epoch ke-31)".
- Tidak ada angka hasil eksperimen pada Tabel 4.8-4.12 (pengujian 10 citra) yang diubah; data baru ditambahkan sebagai pelengkap analisis pada bagian template Confusion Matrix dan Loss Function.

## Update Tabel 4.12 - ResNet50 (Skenario 5)
- BAB 4 - Tabel 4.12: Nilai akurasi ResNet50 diperbarui dari 0,951663 menjadi 0,951696 sesuai hasil eksperimen `experiments/ta_resnet50_20260615_021715/metrics_summary.json` (final_metrics.accuracy), atas konfirmasi pengguna bahwa eksperimen ini merupakan hasil Skenario 5 untuk ResNet50. Persentase pada narasi (95,17%) tidak berubah karena pembulatan sama.
- 6 baris arsitektur lain pada Tabel 4.12 (ResNet101, EfficientNet-B0/B3, VGG-16/19, DenseNet121) dibiarkan apa adanya karena belum ada hasil eksperimen baru yang lengkap untuk arsitektur tersebut (folder `ta_resnet101_20260615_041238` masih kosong).

## Restrukturisasi Skenario 5 dengan Data Aktual per Arsitektur
- BAB 4 - Skenario 5: Mengubah struktur menjadi konfigurasi umum pelatihan, perbedaan hyperparameter per arsitektur, hasil pelatihan per arsitektur, dan analisis perbandingan hasil. Tabel ringkasan hasil dipindahkan ke section analisis sesuai permintaan pengguna.
- BAB 4 - Skenario 5: Menambahkan analisis hasil untuk ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3, VGG-16, VGG-19, dan DenseNet121 berdasarkan artefak eksperimen yang tersedia.
- BAB 4 - DenseNet121: Menggunakan data `experiments/retrain_densenet121_20260414_171014/summary.json` sesuai konfirmasi pengguna, dengan akurasi 0,868678, macro F1 0,868514, TN=1.573, FP=166, FN=291, TP=1.450, train loss 0,4002, dan validation loss 0,3044.
- BAB 4: Mengisi confusion matrix dan ringkasan loss untuk model yang memiliki artefak evaluasi, serta mengganti placeholder ROC-AUC yang tidak tersedia menjadi keterangan `Tidak tersedia`.
