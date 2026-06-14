# BAB 1 PENDAHULUAN

## PENDAHULUAN

1.1 LATAR BELAKANG

Pewarnaan Gram adalah teknik baku untuk mengkategorikan bakteri menjadi Gram-positif dan Gram-negatif berdasarkan karakteristik dinding selnya. Prosedur ini memberikan informasi awal yang cepat tentang keberadaan infeksi bakteri dan morfologi penyebabnya, sehingga dapat menjadi dasar pemilihan terapi antibiotik awal [22]. Sebagai contoh, pewarnaan Gram membantu membedakan infeksi bakteri paru (misalnya pneumonia) atau saluran kemih secara cepat, sehingga dokter dapat menyesuaikan antibiotik yang digunakan [21]. Karena metode ini sederhana dan murah, serta telah digunakan lebih dari satu abad, pewarnaan Gram secara luas dianggap penting dalam pengambilan keputusan klinis dan konservasi antibiotik (antibiotic stewardship) [23]. Dengan demikian, klasifikasi Gram masih menjadi penentu awal dalam diagnosis mikrobiologi klinis dan membantu menentukan langkah perawatan medis selanjutnya [21].

Meskipun bermanfaat, teknik pewarnaan Gram secara manual memiliki berbagai kendala. Proses laboratorium meliputi beberapa tahap (fiksasi, pewarnaan, dekolorisasi, kontra pewarnaan) yang memerlukan ketelitian tinggi. Setiap langkah ini rentan menghasilkan variasi hasil tergantung pada keahlian teknisi dan kondisi pelaksanaannya [22]. Misalnya, waktu dekolorisasi yang tidak tepat atau preparat yang terlalu tebal dapat membuat bakteri Gram-positif tampak Gram-negatif, dan sebaliknya. Kesalahan manusia dan subjektivitas dalam menafsirkan warna mikroskopis dapat menyebabkan pelaporan yang keliru atau berbeda-beda antara petugas [22]. Studi kompetensi di Ethiopia menunjukkan bahwa banyak petugas laboratorium tidak mendapatkan pelatihan berulang atau supervisi yang memadai, sehingga tingkat pengetahuan dan keterampilan pewarnaan Gram mereka masih rendah [24]. Akibatnya, kesalahan interpretasi cukup sering terjadi, misalnya beberapa kasus bakteri Gram-positif dilaporkan sebagai Gram-negatif karena kurangnya pengalaman [24].

Keterbatasan sumber daya manusia dan fasilitas juga menambah tantangan. Di banyak daerah dengan sumber daya terbatas, kultur mikrobiologi tradisional sulit dilakukan karena kurangnya laboratorium lengkap dan bahan reagen. Oleh karena itu, pewarnaan Gram sering menjadi satu-satunya metode pemeriksaan mikrobiologi yang dapat diakses di fasilitas kesehatan tersebut [24]. Kondisi ini memperbesar kebutuhan akan hasil pewarnaan Gram yang cepat dan akurat. Namun tanpa tersedianya peralatan otomatis dan ahli mikroskopi yang memadai, laboratorium di fasilitas terbatas harus bergantung pada metode manual, yang meningkatkan beban kerja dan risiko kesalahan [22]. Variabilitas hasil pewarnaan juga menjadi masalah serius dalam kondisi seperti itu, karena interpretasi yang tidak konsisten dapat mempengaruhi ketepatan diagnosis dan pengobatan pasien.

1.2 PERMASALAHAN

Berdasarkan uraian dari latar belakang di atas, penulis merumuskan beberapa permasalahan yang berfungsi sebagai pedoman pemecahan masalah serta batasan dalam melakukan penelitian ini. Permasalahan tersebut antara lain:

Variabilitas dalam interpretasi mikroskopis terjadi di antara dokter.

Tidak ada sistem otomatis untuk mengklasifikasikan bakteri Gram-positif dan Gram-negatif dari gambar mikroskopis.

1.3 PERUMUSAN MASALAH

Berdasarkan permasalahan yang telah diidentifikasi, maka rumusan masalah dalam penelitian ini adalah sebagai berikut:

Bagaimana membangun sistem otomatis yang mampu mengklasifikasikan bakteri Gram-positif dan Gram-negatif berdasarkan citra mikroskopis?

Bagaimana penerapan metode Convolutional Neural Network (CNN) dapat mengurangi variabilitas interpretasi mikroskopis antar dokter dalam proses klasifikasi bakteri Gram?:

1.4 TUJUAN

Dari permasalahan yang ingin diselesaikan, proyek akhir ini bertujuan untuk mengembangkan sistem deteksi otomatis bakteri Gram dari citra mikroskopis menggunakan metode Convolutional Neural Network (CNN). Adapun tujuan dari pengembangan sistem ini adalah sebagai berikut:

Mengembangkan sistem berbasis CNN yang mampu melakukan deteksi dan klasifikasi otomatis terhadap bakteri Gram-positif dan Gram-negatif dari citra mikroskopis.

Merancang arsitektur sistem deteksi yang mencakup tahapan pra-proses data, pelatihan model, validasi, penyimpanan model, dan evaluasi performa.

1.5 MANFAAT

Dengan adanya sistem deteksi otomatis bakteri Gram dari citra mikroskopis menggunakan metode Convolutional Neural Network (CNN), diharapkan dapat meningkatkan efisiensi dan akurasi dalam proses identifikasi bakteri, mempercepat penegakan diagnosis, serta mengurangi risiko kesalahan manusia dalam interpretasi citra mikroskopis. Hal ini akan memberikan dampak positif dalam meningkatkan kualitas layanan laboratorium dan pengambilan keputusan klinis. Selain itu, penulis berharap penelitian ini dapat memberikan manfaat bagi Tenaga medis dan Analis Laboratorium ialah :

Membantu tenaga medis dan analis mikrobiologi dalam mengidentifikasi jenis bakteri Gram-positif atau Gram-negatif secara cepat dan akurat berdasarkan citra mikroskopik.

Memungkinkan deteksi lebih dini terhadap infeksi bakteri, sehingga pengobatan yang tepat dapat diberikan lebih cepat kepada pasien.

1.6 SISTEMATIKA PENULISAN

Sistematika penulisan yang menjadi langkah-langkah dalam proses penyusunan Proyek Akhir ini, yaitu:

Pada bab ini dijelaskan mengenai latar belakang, perumusan masalah, tujuan, serta manfaat dari Proyek Akhir ini.

Latar belakang berisi penjelasan mengenai pentingnya deteksi cepat dan akurat terhadap bakteri Gram dalam dunia medis, serta peran teknologi deep learning dalam mengatasi keterbatasan metode konvensional yang bersifat subjektif.

Perumusan masalah memaparkan tantangan utama seperti variabilitas interpretasi manual dalam analisis mikroskopis dan belum tersedianya sistem otomatis yang akurat.

Tujuan berupa pengembangan sistem berbasis CNN untuk klasifikasi otomatis bakteri Gram-positif dan Gram-negatif.

Manfaat yang diharapkan meliputi peningkatan efisiensi laboratorium, akurasi diagnosis, dan kontribusi terhadap teknologi diagnostik modern.

Bab ini membahas teori-teori yang digunakan sebagai dasar dan landasan dalam penelitian. Teori-teori yang dikaji meliputi konsep dasar bakteri Gram-positif dan Gram-negatif, prinsip pewarnaan Gram, dasar-dasar citra digital, serta teori tentang Convolutional Neural Network (CNN), klasifikasi citra, dan penelitian terdahulu yang relevan.

Pada bab ini dijelaskan perancangan sistem deteksi otomatis bakteri Gram menggunakan CNN. Bab ini mencakup deskripsi permasalahan yang akan diselesaikan, arsitektur sistem, tahapan pengolahan data seperti pra-proses citra, pelatihan model CNN, validasi dan evaluasi hasil klasifikasi, serta perancangan alur kerja sistem secara menyeluruh.

