# BAB 5 PENUTUP

## Kesimpulan

Berdasarkan hasil penelitian yang telah dilakukan dalam pengembangan sistem deteksi otomatis bakteri Gram dari citra mikroskopis menggunakan Convolutional Neural Network (CNN), dapat ditarik beberapa kesimpulan sebagai berikut :

Sistem deteksi otomatis klasifikasi bakteri Gram-positif dan Gram-negatif berbasis CNN berhasil dibangun dan diuji melalui lima skenario eksperimen yang mencakup: Simple CNN (from scratch), Simple CNN dengan augmentasi data, tujuh arsitektur CNN standar (from scratch), transfer learning dengan fine-tuning tahap pertama, serta transfer learning dengan fine-tuning penuh. Sistem ini terbukti mampu mengklasifikasikan citra bakteri hasil pewarnaan Gram secara otomatis tanpa memerlukan interpretasi manual dari analis laboratorium.

Perbandingan lima skenario pelatihan menunjukkan bahwa pendekatan transfer learning dengan fine-tuning penuh (Skenario 5) menghasilkan performa terbaik secara rata-rata, dengan nilai akurasi rata-rata sebesar 89,69% (0,8969). Model EfficientNet-B3 pada skenario ini mencatatkan akurasi tertinggi sebesar 98,46% (0,9846), melampaui seluruh model dan skenario lain yang diuji, menjadikannya kandidat model terbaik untuk sistem klasifikasi bakteri Gram berbasis deep learning.

Penerapan augmentasi data pada Simple CNN terbukti memberikan peningkatan performa yang signifikan dan konsisten. Nilai rata-rata akurasi meningkat dari 70,21% (Simple CNN tanpa augmentasi) menjadi 75,96% (Simple CNN dengan augmentasi), dengan rentang fluktuasi antar data uji yang menyempit dari 15,3 menjadi 10,0 poin persentase. Hal ini mengonfirmasi bahwa augmentasi data efektif dalam meningkatkan kemampuan generalisasi model, khususnya pada kondisi data set yang terbatas.

Arsitektur berbasis residual connection (ResNet50 dan ResNet101) secara konsisten menunjukkan performa unggul pada seluruh skenario pelatihan from scratch, dengan ResNet101 mencatatkan akurasi tertinggi sebesar 92,62% pada Skenario 3. Mekanisme skip connection yang menjadi ciri khas ResNet terbukti efektif dalam mengatasi masalah vanishing gradient sehingga memungkinkan optimasi yang lebih stabil meskipun tanpa pretrained weights. Sebaliknya, arsitektur VGG-16 dan DenseNet121 secara konsisten menunjukkan performa lebih rendah di seluruh skenario yang diuji.

Keterbatasan utama penelitian ini adalah jumlah data set citra mikroskopis bakteri Gram yang tersedia masih terbatas. Kondisi ini menjadi faktor pembatas utama yang memengaruhi kemampuan generalisasi model, terutama pada saat dihadapkan dengan data yang belum pernah dilihat model sebelumnya (unseen data). Penggunaan augmentasi data konvensional telah dilakukan sebagai solusi parsial, namun belum sepenuhnya dapat menggantikan kebutuhan akan data asli yang lebih beragam.

## Saran

Berdasarkan hasil penelitian dan keterbatasan yang ditemukan, berikut adalah saran-saran yang dapat dipertimbangkan untuk pengembangan penelitian lebih lanjut.

Perluasan dan diversifikasi data set perlu diprioritaskan dalam penelitian selanjutnya. Dataset yang lebih besar dan beragam, mencakup variasi kondisi pewarnaan, jenis bakteri, kualitas preparat, dan sampel dari berbagai institusi laboratorium yang berbeda, akan sangat meningkatkan kemampuan generalisasi model terhadap kondisi klinis yang sesungguhnya.

Eksplorasi arsitektur yang lebih ringan namun tetap akurat disarankan sebagai arah penelitian berikutnya, mengingat sistem ini ditargetkan untuk diintegrasikan ke dalam platform berbasis web dan berpotensi digunakan di fasilitas kesehatan dengan sumber daya komputasi terbatas. Arsitektur seperti MobileNet V3, ShuffleNet V2, atau EfficientNet-Lite dapat dievaluasi sebagai alternatif yang menyeimbangkan akurasi dan efisiensi inferensi.

Penambahan kemampuan deteksi objek (object detection) menggunakan YOLO atau arsitektur serupa perlu diintegrasikan ke dalam sistem. Kemampuan ini akan memungkinkan sistem tidak hanya mengklasifikasikan seluruh citra, tetapi juga melakukan lokalisasi dan penghitungan bakteri secara individual dalam satu bidang pandang mikroskop, yang jauh lebih mendekati kebutuhan klinis nyata di laboratorium.

Penyelesaian integrasi sistem secara end-to-end perlu dipercepat, meliputi penyempurnaan seluruh endpoint API yang diperlukan oleh front-end, pengujian skenario 4 dan 5 secara lengkap, serta pengujian integrasi sistem secara menyeluruh. Selain itu, uji klinis terbatas (pilot study) di laboratorium rumah sakit atau puskesmas perlu dilakukan untuk mengukur kinerja sistem dalam kondisi nyata dan mendapatkan umpan balik dari pengguna akhir, yaitu analis laboratorium dan tenaga medis.

Penelitian lanjutan disarankan untuk mengeksplorasi strategi fine-tuning yang lebih optimal, seperti penggunaan learning rate scheduler yang adaptif, teknik layer freezing secara bertahap, atau pendekatan progressive resizing, guna mengatasi fenomena stagnansi akurasi yang ditemukan pada arsitektur VGG-16 dan performa yang tidak merata antar arsitektur pada skenario transfer learning.

