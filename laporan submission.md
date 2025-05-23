# Laporan Proyek Machine Learning - Cika Rahmannia Febrianti

## Domain Proyek

Depresi di kalangan mahasiswa merupakan isu kesehatan mental yang signifikan, terutama di Indonesia. Berbagai faktor seperti tekanan akademik, ekspektasi keluarga, masalah keuangan, serta kurangnya sistem dukungan sosial sering kali dianggap sebagai hal yang lumrah, padahal secara nyata dapat meningkatkan kerentanan mahasiswa terhadap gangguan kesehatan mental, khususnya depresi. Memahami faktor-faktor risiko ini sangat penting untuk mengembangkan strategi pencegahan dan intervensi yang lebih efektif dan tepat sasaran.

Secara global, diperkirakan 280 juta orang hidup dengan depresi, termasuk 5% populasi dewasa (6% perempuan dan 4% laki-laki). Wanita dilaporkan lebih rentan terhadap depresi dibandingkan pria. Meskipun terdapat pengobatan yang efektif untuk depresi ringan, sedang, dan berat, lebih dari 75% penderita di negara berpenghasilan rendah dan menengah tidak menerima perawatan yang memadai karena kurangnya investasi, keterbatasan tenaga kesehatan terlatih, dan stigma sosial terhadap gangguan mental (WHO, 2023). Menurut data Badan Kebijakan Pembangunan Kesehatan Kementerian Kesehatan RI (2023), prevalensi depresi secara nasional mencapai 1,4%, dengan kelompok usia 15–24 tahun mencatat angka tertinggi, yaitu sebesar 2%. Depresi merupakan penyebab utama disabilitas pada remaja. Depresi dapat menjadi penyebab bunuh diri, dan bunuh diri merupakan 

Dampak depresi pada mahasiswa sangat luas dan mendalam. Tidak hanya menurunkan performa akademik, depresi juga berdampak pada kesejahteraan emosional, kualitas tidur, pola makan, serta menurunkan produktivitas dan kemampuan sosial. Apabila tidak dideteksi dan ditangani sejak dini, kondisi ini dapat berkembang menjadi gangguan mental yang lebih serius bahkan berujung pada konsekuensi fatal. Oleh karena itu, upaya deteksi dini menjadi krusial. Salah satu pendekatan yang menjanjikan adalah penggunaan teknologi machine learning untuk memprediksi risiko depresi berdasarkan data yang relevan, seperti durasi tidur, tekanan akademik, stres keuangan, riwayat keluarga dengan gangguan mental, hingga adanya pikiran untuk bunuh diri. 

Machine learning menawarkan pendekatan berbasis data untuk mendeteksi gejala depresi sejak dini. Dengan memanfaatkan data demografi, gaya hidup, dan tekanan akademik, model prediktif dapat membantu institusi pendidikan atau tenaga konselor mengidentifikasi mahasiswa berisiko tinggi sehingga intervensi dapat dilakukan lebih cepat dan tepat sasaran. Dengan meningkatnya prevalensi depresi dan terbatasnya tenaga psikolog di kampus-kampus, pendekatan berbasis teknologi seperti machine learning menjadi solusi yang skalabel dan efisien. Deteksi dini ini diharapkan dapat membantu menekan angka dropout, meningkatkan kesejahteraan mahasiswa, dan mendukung terciptanya ekosistem pendidikan yang lebih sehat secara psikologis.

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Pernyataan Masalah 1
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

