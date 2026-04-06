# 🏦 Analisis Transaksi Perbankan: Clustering & Klasifikasi Nasabah

> **Submission Belajar Machine Learning untuk Pemula (BMLP) — Dicoding**  
> Menggabungkan pendekatan **Unsupervised Learning** (Clustering) dan **Supervised Learning** (Klasifikasi) untuk menganalisis pola transaksi nasabah bank.

---

## 📌 Deskripsi Proyek

Proyek ini bertujuan untuk menganalisis perilaku transaksi nasabah perbankan menggunakan dua pendekatan machine learning:

1. **Unsupervised Learning** — Mengelompokkan nasabah berdasarkan karakteristik transaksi tanpa label menggunakan algoritma clustering (K-Means).
2. **Supervised Learning** — Melatih model klasifikasi berdasarkan hasil cluster untuk memprediksi segmen nasabah baru.

Dataset yang digunakan mencakup data transaksi perbankan dengan fitur seperti jumlah transaksi, usia nasabah, durasi transaksi, saldo akun, lokasi, dan lain-lain.

---

## 📂 Struktur Proyek

```
Submission-BMLP-Unsupervised-Supervised/
│
├── notebook/
│   ├── 01_EDA_Preprocessing.ipynb        # Eksplorasi dan preprocessing data
│   ├── 02_Unsupervised_Clustering.ipynb  # K-Means clustering & analisis cluster
│   └── 03_Supervised_Classification.ipynb# Model klasifikasi berbasis label cluster
│
├── data/
│   └── bank_transactions.csv             # Dataset transaksi nasabah
│
├── requirements.txt                      # Library yang dibutuhkan
└── README.md
```

---

## 🧩 Fitur Dataset

| Fitur | Deskripsi |
|---|---|
| `TransactionAmount` | Jumlah nominal transaksi |
| `CustomerAge` | Usia nasabah |
| `TransactionDuration` | Durasi transaksi (detik) |
| `LoginAttempts` | Jumlah percobaan login |
| `AccountBalance` | Saldo rekening nasabah |
| `TransactionType` | Jenis transaksi (Debit/Kredit) |
| `Location` | Lokasi transaksi |
| `Channel` | Saluran transaksi (Branch/Online/ATM) |
| `CustomerOccupation` | Pekerjaan nasabah |
| `AgeGroup` | Kategori kelompok usia nasabah |

---

## 🔍 Tahapan Analisis

### 1. 📊 Exploratory Data Analysis (EDA)
- Statistik deskriptif seluruh fitur
- Visualisasi distribusi dan korelasi antar fitur
- Identifikasi missing values dan outlier

### 2. ⚙️ Preprocessing
- Penanganan missing values
- Encoding fitur kategorikal
- Normalisasi / scaling fitur numerik
- Feature engineering (`AgeGroup`)

### 3. 🔵 Unsupervised Learning — K-Means Clustering
- Penentuan jumlah cluster optimal menggunakan **Elbow Method** & **Silhouette Score**
- Pelatihan model K-Means
- Inverse transform hasil scaling untuk interpretasi
- Analisis karakteristik tiap cluster:

| Cluster | Label | Karakteristik Utama |
|---|---|---|
| Cluster 0 | Nasabah Dewasa - Transaksi Stabil | Usia ~45 th, Dokter, Saldo tinggi, Branch |
| Cluster 1 | Nasabah Muda - Transaksi Aktif | Usia ~44 th, Mahasiswa, Saldo lebih rendah |

### 4. 🟢 Supervised Learning — Klasifikasi
- Label cluster dijadikan target klasifikasi
- Algoritma yang digunakan: *(contoh: Random Forest / Decision Tree / KNN)*
- Evaluasi model: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## 📈 Hasil & Evaluasi

### Clustering
- **Jumlah Cluster Optimal:** 2
- **Silhouette Score:** *(isi nilai aktual)*

### Klasifikasi
| Metrik | Nilai |
|---|---|
| Accuracy | *xx%* |
| Precision | *xx%* |
| Recall | *xx%* |
| F1-Score | *xx%* |

---

## 🛠️ Teknologi yang Digunakan

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Array-lightblue?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-yellow)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-teal)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## 🚀 Cara Menjalankan

### 1. Clone repositori
```bash
git clone https://github.com/aldinidziaulhaq/Submission-BMLP-Unsupervised-Supervised-.git
cd Submission-BMLP-Unsupervised-Supervised-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan notebook secara berurutan
```
01_EDA_Preprocessing.ipynb
02_Unsupervised_Clustering.ipynb
03_Supervised_Classification.ipynb
```

---

## 📋 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

---

## 👤 Author

**Aldini Dziaulhaq**  
📧 *(email opsional)*  
🔗 [GitHub](https://github.com/aldinidziaulhaq)

---

## 📄 Lisensi

Proyek ini dibuat sebagai bagian dari submission kelas **Belajar Machine Learning untuk Pemula** di [Dicoding](https://www.dicoding.com/).
