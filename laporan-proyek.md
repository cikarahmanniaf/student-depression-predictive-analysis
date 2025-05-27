# Laporan Proyek Machine Learning - Cika Rahmannia Febrianti

## Project Overview

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
- Melakukan hyperparameter tuning menggunakan Random Search untuk memaksimalkan performa model.
- Evaluasi model menggunakan metriks klasifikasi, seperti: akurasi, precision, recall, dan F1-score.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah 'Student Depression Dataset' yang berisi 27901 data mahasiswa dengan berbagai fitur atau variabel terkait denan kondisi mental dan faktor pendukungnya. Sumber dataset: [https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset?resource=download]

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

### Import Library
Beberapa library yang akan dipakai dalam analisis ini sebagai berikut.

```python
# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown

sns.set(style='whitegrid')
plt.style.use('seaborn-v0_8-whitegrid')

# Preprocessing & splitting
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Warning control
import warnings
warnings.filterwarnings("ignore")
```

### Exploratory Data Analysis (EDA)
Dilakukan tahapan eksplorasi data sebelum modelling dengan tujuan untuk memahami karakteristik data, termasuk:
- Loading dataset
- Struktur dan tipe data 
- Statistika deskriptif
- Missing value
- Data Duplikat
- Sebaran variabel target ('Depression') untuk mengetahui keseimbangan data
- Distribusi variabel numerik menggunakan histogram
- Korelasi variabel numerik
- Boxplot untuk pemeriksaan outlier
- Distribusi variabel kategorik terhadap variabel target menggunakan count plot

Setelah dilakukan eksplorasi, didapatkan beberapa informasi seperti: 
- Ada beberapa fitur string yang harus dikonversi ke numerik ('Sleep Duration'), Encoding fitur kategorikal, dan drop kolom yang tidak relevan.
- Tidak ada missing value, duplicate data, serta invalid value (dilihat dari statistika deskriptif)
- Sebaran status depresi ada 'Ya': 16336 dan 'Tidak': 11565
- Fitur 'Work Pressure' dan 'Job Satisfaction' nampak pada histogram seperti hanya ada nilai '0', tetapi sebenarnya tidak. Hal itu karena kecilnya jumlah mahasiswa pekerja.
- Ada beberapa fitur yang memiliki korelasi rendah-cukup dengan fitur 'Depression', termasuk: 'Academic Pressure' dan 'Work/Study Hours' dengan korelais postif: 'Age' dan 'Study Satisfaction' dengan korelasi negatif.
- Fitur 'Age' terdeteksi memiliki outlier dalma boxplot, tetapi diputuskan tidak dilakukan imputasi maupun dropping karena rentang masih dalam tahap wajar dan masuk akal terjadi (ada mahasiswa baru mulai pendidikan tinggi di usia di atas 24 tahun).
- Lelaki lebih banyak terkena depresi dibanding dengan perempuan. Jumlah tertinggi depresinya memiliki rata-rata durasi tidur kurang dari 5 jam, pola diet yang tidak sehat, dan ada masalah tekanan finansial.

## Data Preparation
Tahapan ini bertujuan untuk mempersiapkan data agar dapat digunakan dalam proses pemodelan machine learning. Seluruh langkah dilakukan sesuai urutan dan isi notebook sebagai berikut:

### 1. Menghapus fitur tidak relevan
Fitur-fitur berikut dihapus karena dianggap: tidak berkontribusi signifikan terhadap prediksi depresi, memiliki variansi rendah, dan berpotensi mirip dengan label dan menyebabkan data leakage.
```python
df.drop([
    'id',
    'City',
    'Profession',
    'Job Satisfaction',
    'Have you ever had suicidal thoughts ?',
    'Degree'
], axis=1, inplace=True)
```
### 2. Menangani nilai tidak valid
Pada kolom Financial Stress, terdapat nilai '?' yang tidak valid. Nilai tersebut diganti dengan modus dari kolom yang sama, kemudian diubah ke tipe data numerik.
```python
print(df['Financial Stress'].value_counts())

# Ganti '?' dengan modus
mode_val = df.loc[df['Financial Stress'] != '?', 'Financial Stress'].mode()[0]
df['Financial Stress'] = df['Financial Stress'].replace('?', mode_val)
df['Financial Stress'] = df['Financial Stress'].astype(float)
```

### 3. Pembersihan string
Beberapa fitur kategorikal memiliki spasi atau tanda kutip tambahan yang perlu dibersihkan untuk menghindari inkonsistensi nilai.
```python
df['Gender'] = df['Gender'].str.strip().str.replace("'", "")
df['Family History of Mental Illness'] = df['Family History of Mental Illness'].str.strip().str.replace("'", "")
df['Sleep Duration'] = df['Sleep Duration'].str.strip().str.replace("'", "")
df['Dietary Habits'] = df['Dietary Habits'].str.strip().str.replace("'", "")
```

### 4. Encoding fitur kategorikal
Untuk fitur biner digunakan Label Encoding:
```python
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})
```
Untuk fitur ordinal digunakan Ordinal Encoding:
```python
sleep_map = {
    'Less than 5 hours': 0,
    '5-6 hours': 1,
    '7-8 hours': 2,
    'More than 8 hours': 3,
    'Others': 4
}
df['Sleep Duration'] = df['Sleep Duration'].replace(sleep_map)

diet_map = {
    'Unhealthy': 0,
    'Moderate': 1,
    'Healthy': 2,
    'Others': 3
}
df['Dietary Habits'] = df['Dietary Habits'].replace(diet_map)
```

### 5. Standarisasi fitur numerik
Fitur numerik (num_cols) seperti Age, CGPA, dan Work/Study Hours distandarisasi menggunakan StandardScaler.
```python
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
```

### 6. Pemisahan fitur (X) dan target (y)
```python
X = df.drop('Depression', axis=1)
y = df['Depression']
```

Dengan tahapan di atas, data telah siap digunakan untuk pelatihan model machine learning

## Modeling
Tahapan ini bertujuan untuk membangun model machine learning yang dapat mengklasifikasikan status depresi mahasiswa. Dua algoritma yang digunakan adalah Logistic Regression dan Random Forest Classifier. Berikut ini penjelasan prinsip kerja dan alasan pemilihannya.

### 1. Penjelasan model
#### a. Logistic Regression
Logistic Regression adalah algoritma klasifikasi linier yang digunakan untuk memprediksi probabilitas kejadian suatu kelas. Fungsi utamanya adalah fungsi sigmoid/logistik yang membatasi output antara 0 dan 1. Dengan asumsi dasar bahwa tidak terdapat hubungan linear antara fitur dan logit dari probabilitas serta tidak ada multikolinearitas antar fitur. Dengan parameter utama:

- C: Mengontrol kekuatan regularisasi (semakin kecil, regularisasi semakin kuat).
- penalty: Jenis regularisasi (L2 digunakan untuk mencegah overfitting).
- solver: Metode optimasi seperti lbfgs atau saga.

#### b. Random Forest Classifier
Random Forest adalah algoritma ensemble berbasis pohon keputusan yang bekerja dengan membangun banyak decision tree dan menggabungkan hasil voting untuk klasifikasi. Keunggulannya adalah mampu menangkap hubungan non linier dan interaksi antar fitur serta tahan terhadap outlier dan multikolinearitas. Dengan parameter utama:

- n_estimators: Jumlah pohon dalam hutan.
- max_depth: Kedalaman maksimum pohon.
- min_samples_split: Minimum jumlah sampel untuk split node.
- min_samples_leaf: Minimum sampel pada daun.
- bootstrap: Sampling dengan pengembalian atau tidak.

### 2. Splitting Data
Sebelum pemodelan, data dibagi menjadi data latih dan data uji dengan rasio 80:20 dan stratifikasi target.
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 3. Pemodelan
Pemodelan awal dilakukan menggunakan parameter default dari masing-masing algoritma untuk mendapatkan baseline performance.
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
```
Parameter default digunakan terlebih dahulu untuk membandingkan performa awal antar model secara objektif sebelum dilakukan tuning

## Evaluation
Kedua model dievaluasi menggunakan metrik evaluasi klasifikasi (accuracy, precision, recall, dan F1-score). 

Hasil Evaluasi Model

| Model               | Accuracy | Precision (macro avg) | Recall (macro avg) | F1-score (macro avg) |
|---------------------|----------|------------------------|---------------------|----------------------|
| Logistic Regression | 0.7961   | 0.7916                 | 0.7851              | 0.7877               |
| Random Forest       | 0.7900   | 0.7843                 | 0.7809              | 0.7824               |

Berdasarkan tabel di atas, model Logistic Regression menunjukkan performa yang sedikit lebih baik dibandingkan Random Forest pada semua metrik evaluasi utama. Meski perbedaannya tidak signifikan, Logistic Regression dipilih sebagai model akhir karena memberikan akurasi dan generalisasi yang lebih stabil dengan kompleksitas yang lebih rendah.

Untuk meningkatkan performa, dilakukan pencarian kombinasi parameter terbaik menggunakan RandomizedSearchCV. Pendekatan ini lebih efisien daripada Grid Search karena mengevaluasi kombinasi parameter secara acak.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

param_grids = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'C': uniform(0.01, 10),
            'solver': ['lbfgs', 'saga'],
            'penalty': ['l2']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
    }
}
```

Diperoleh model terbaik setelah tuning adalah 
- Logistic Regression
  - `C`: 0.017787658410143285
  - `penalty`: `'l2'`
  - `solver`: `'saga'`

- Random Forest
  - `n_estimators`: 50  
  - `min_samples_split`: 5  
  - `min_samples_leaf`: 1  
  - `max_depth`: 10  
  - `bootstrap`: `True`
 
Dengan hasil evaluasi setelah dilakukan tuning sebagai berikut.

| Model                       | Accuracy | Precision (macro avg) | Recall (macro avg) | F1-score (macro avg) |
|-----------------------------|----------|------------------------|---------------------|----------------------|
| Logistic Regression (Tuned) | 0.7968   | 0.7924                 | 0.7859              | 0.7885               |
| Random Forest (Tuned)       | 0.7923   | 0.7883                 | 0.7802              | 0.7833               |

Berdasarkan hasil tuning, Logistic Regression tetap mempertahankan performa yang lebih baik dibanding Random Forest. Oleh karena itu, Logistic Regression (Tuned) dipilih sebagai model akhir untuk digunakan dalam sistem prediksi depresi mahasiswa.

## Conclusion

1. Membangun Model Prediksi Depresi Mahasiswa
Model prediksi dikembangkan menggunakan berbagai algoritma machine learning untuk mengidentifikasi status depresi mahasiswa berdasarkan fitur-fitur demografi, akademik, dan gaya hidup. Proses preprocessing, seleksi fitur, pembagian data, serta evaluasi model dilakukan. Hasil evaluasi terbaik ditunjukkan melalui metrik evauasi (accuracy, precision, recall, dan F1-score) paling baik.

2. Faktor-Faktor yang Paling Signifikan terhadap Depresi Mahasiswa
Dilakukan analisis korelasi Pearson terhadap fitur-fitur numerik untuk mengetahui faktor atau fitur mana yang signifikan mempengaruhi status depresi mahasiswa. Berikut adalah fitur-fitur dengan korelasi tertinggi terhadap variabel target (Depression):

- Academic Pressure (`+0.47`): Korelasi positif yang cukup kuat menunjukkan bahwa semakin tinggi tekanan akademik, semakin tinggi risiko depresi.
- Age (`-0.23`): Korelasi negatif sedang menunjukkan bahwa mahasiswa yang lebih muda cenderung memiliki risiko depresi yang lebih tinggi.
- Work/Study Hours (`+0.21`): Semakin tinggi waktu kerja/belajar, cenderung berkorelasi dengan peningkatan risiko depresi.
- Study Satisfaction (`-0.17`): Semakin rendah kepuasan belajar, semakin tinggi kecenderungan mengalami depresi.

3. Algoritma dengan Performa Terbaik
Berdasarkan hasil evaluasi model sebelum dan sesudah hyperparameter tuning, Logistic Regression dengan tuning parameter menunjukkan performa terbaik secara keseluruhan. Model ini memperoleh nilai F1-score (macro avg) tertinggi sebesar 0.7885 setelah tuning, melebihi algoritma lainnya dalam menjaga keseimbangan antara precision dan recall untuk masing-masing kelas. Dengan demikian, Logistic Regression (Tuned) menjadi pilihan terbaik untuk klasifikasi status depresi pada mahasiswa dalam studi ini.


