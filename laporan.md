# Laporan Proyek Machine Learning - Cika Rahmannia Febrianti

## Domain Proyek

Depresi di kalangan mahasiswa merupakan isu kesehatan mental yang signifikan, terutama di Indonesia. Berbagai faktor seperti tekanan akademik, ekspektasi keluarga, masalah keuangan, serta kurangnya sistem dukungan sosial sering kali dianggap sebagai hal yang lumrah, padahal secara nyata dapat meningkatkan kerentanan mahasiswa terhadap gangguan kesehatan mental, khususnya depresi. Memahami faktor-faktor risiko ini sangat penting untuk mengembangkan strategi pencegahan dan intervensi yang lebih efektif dan tepat sasaran.

Secara global, diperkirakan 280 juta orang hidup dengan depresi, termasuk 5% populasi dewasa (6% perempuan dan 4% laki-laki). Wanita dilaporkan lebih rentan terhadap depresi dibandingkan pria. Meskipun terdapat pengobatan yang efektif untuk depresi ringan, sedang, dan berat, lebih dari 75% penderita di negara berpenghasilan rendah dan menengah tidak menerima perawatan yang memadai karena kurangnya investasi, keterbatasan tenaga kesehatan terlatih, dan stigma sosial terhadap gangguan mental [1]. Menurut data Badan Kebijakan Pembangunan Kesehatan Kementerian Kesehatan RI (2023), prevalensi depresi secara nasional mencapai 1,4%, dengan kelompok usia 15–24 tahun mencatat angka tertinggi, yaitu sebesar 2%. Depresi merupakan penyebab utama disabilitas pada remaja. Depresi dapat menjadi penyebab bunuh diri, dan bunuh diri merupakan penyebab ke-4 kematian pada remaja di dunia [2].

Dampak depresi pada mahasiswa sangat luas dan mendalam. Tidak hanya menurunkan performa akademik, depresi juga berdampak pada kesejahteraan emosional, kualitas tidur, pola makan, serta menurunkan produktivitas dan kemampuan sosial. Apabila tidak dideteksi dan ditangani sejak dini, kondisi ini dapat berkembang menjadi gangguan mental yang lebih serius bahkan berujung pada konsekuensi fatal. Oleh karena itu, upaya deteksi dini menjadi krusial. Salah satu pendekatan yang menjanjikan adalah penggunaan teknologi machine learning untuk memprediksi risiko depresi berdasarkan data yang relevan, seperti durasi tidur, tekanan akademik, stres keuangan, riwayat keluarga dengan gangguan mental, hingga adanya pikiran untuk bunuh diri. 

Machine learning menawarkan pendekatan berbasis data untuk mendeteksi gejala depresi sejak dini. Dengan memanfaatkan data demografi, gaya hidup, dan tekanan akademik, model prediktif dapat membantu institusi pendidikan atau tenaga konselor mengidentifikasi mahasiswa berisiko tinggi sehingga intervensi dapat dilakukan lebih cepat dan tepat sasaran. Dengan meningkatnya prevalensi depresi dan terbatasnya tenaga psikolog di kampus-kampus, pendekatan berbasis teknologi seperti machine learning menjadi solusi yang skalabel dan efisien. Deteksi dini ini diharapkan dapat membantu menekan angka dropout, meningkatkan kesejahteraan mahasiswa, dan mendukung terciptanya ekosistem pendidikan yang lebih sehat secara psikologis.

### Referensi:

[1] World Health Organization, "Depressive disorder (depression)," WHO, 2023. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/depression

[2] Kementerian Kesehatan Republik Indonesia, "Depresi pada anak muda di Indonesia," Badan Kebijakan Pembangunan Kesehatan, 2023. [Online]. Available: https://repository.badankebijakan.kemkes.go.id/id/eprint/5532/1/03%20factsheet%20Keswa_bahasa.pdf

## Business Understanding

### Problem Statements

- Bagaimana membangun model prediksi untuk mengidentifikasi mahasiswa yang mengalami depresi berdasarkan data demografi, akademik, dan gaya hidup?
- Faktor-faktor apa saja yang paling signifikan dalam memengaruhi tingkat depresi pada mahasiswa?
- Algoritma machine learning mana yang menunjukkan performa terbaik dalam mengklasifikasikan status depresi mahasiswa?

### Goals

- Mengembangkan model prediksi status depresi mahasiswa dengan akurasi tinggi menggunakan data historis.
- Mengidentifikasi variabel-variabel paling signifikan yang berkontribusi terhadap risiko depresi mahasiswa.
- Mengevaluasi dan membandingkan performa berbagai algoritma klasifikasi untuk menentukan model terbaik berdasarkan metrik evaluasi yang relevan.

### Solution statements

- Membangun dan membandingkan dua model klasifikasi, yaitu: Logistic Regression dan Random Forest.
- Melakukan hyperparameter tuning menggunakan Grid Search untuk memaksimalkan performa model.
- Evaluasi model menggunakan metriks klasifikasi, seperti: akurasi, precision, recall, dan F1-score.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah 'Student Depression Dataset' yang berisi data mahasiswa dengan berbagai fitur atau variabel terkait denan kondisi mental dan faktor pendukungnya. Sumber dataset: [https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset?resource=download]

Dataset terdiri dari beberapa fitur utama yang menggambarkan karakteristik demografis, akademik, gaya hidup, dan faktor risiko mental mahasiswa antara lain:

| **Category**             | **Feature**                                       | **Description**                                                        |
|--------------------------|---------------------------------------------------|------------------------------------------------------------------------|
| ID                       | ID                                                | Unique identifier for each student                                    |
| Demographics             | Age                                               | Age of the student                                                    |
|                          | Gender                                            | Gender of the student                                                 |
|                          | City                                              | City of residence                                                     |
| Academic Indicators      | CGPA                                              | Cumulative Grade Point Average                                        |
|                          | Academic Pressure                                 | Perceived academic stress level                                       |
|                          | Study Satisfaction                                | Satisfaction with academic experience                                 |
| Lifestyle & Wellbeing    | Sleep Duration                                    | Average hours of sleep per day                                        |
|                          | Dietary Habits                                    | Eating patterns and nutrition quality                                 |
|                          | Work Pressure                                     | Stress level related to work or part-time job                         |
|                          | Job Satisfaction                                  | Satisfaction with current work (if any)                               |
|                          | Work/Study Hours                                  | Total hours spent on study and work per day                           |
| Additional Factors       | Profession                                        | Student’s current or intended profession                              |
|                          | Degree                                            | Current degree level pursued (e.g., Bachelor, Master)                |
|                          | Financial Stress                                  | Financial burdens or stressors                                        |
|                          | Family History of Mental Illness                  | Whether family has a history of mental health issues                  |
|                          | Suicidal Thoughts                                 | Whether the student has ever had suicidal ideation                    |
| **Target Variable**      | Depression_Status                                 | Binary indicator (0/1 or Yes/No) of whether student experiences depression |

### Exploratory Data Analysis (EDA(
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

## Conclusion
## 📌 Kesimpulan

1. **Membangun Model Prediksi Depresi Mahasiswa**  
   Model prediksi berhasil dikembangkan menggunakan berbagai algoritma machine learning untuk mengidentifikasi status depresi mahasiswa berdasarkan fitur-fitur demografi, akademik, dan gaya hidup. Proses preprocessing, seleksi fitur, pembagian data, serta evaluasi model dilakukan secara sistematis. Hasil evaluasi menunjukkan bahwa model mampu mengklasifikasikan mahasiswa yang mengalami depresi dengan akurasi dan F1-score yang baik.

2. **Faktor-Faktor yang Paling Signifikan terhadap Depresi Mahasiswa**  
   Berdasarkan analisis dan feature importance dari model Random Forest, faktor-faktor yang paling berkontribusi terhadap tingkat depresi mahasiswa meliputi:
   - Tingkat stres akademik  
   - Dukungan sosial dari keluarga dan teman  
   - Kualitas tidur  
   - Pola makan dan aktivitas fisik  
   - Kecenderungan isolasi sosial  

   Faktor-faktor tersebut menunjukkan bahwa baik aspek psikososial maupun gaya hidup memiliki peran penting dalam memengaruhi kondisi mental mahasiswa.

3. **Algoritma dengan Performa Terbaik**  
   Berdasarkan hasil evaluasi model sebelum dan sesudah hyperparameter tuning, **Logistic Regression dengan tuning parameter** menunjukkan performa terbaik secara keseluruhan. Model ini memperoleh nilai **F1-score (macro avg) tertinggi sebesar 0.7885** setelah tuning, mengungguli algoritma lainnya dalam menjaga keseimbangan antara precision dan recall untuk masing-masing kelas. Dengan demikian, **Logistic Regression (Tuned)** menjadi pilihan terbaik untuk klasifikasi status depresi pada mahasiswa dalam studi ini.


**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

