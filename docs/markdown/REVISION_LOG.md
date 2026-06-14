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
