# 🏦 Deteksi Pola Transaksi Nasabah Bank — Clustering & Klasifikasi

> **Submission Akhir | Belajar Machine Learning untuk Pemula (BMLP) — Dicoding**
>
> Proyek ini menganalisis perilaku transaksi 2.512 nasabah bank menggunakan dua pendekatan machine learning secara berurutan — mulai dari mengelompokkan nasabah tanpa label (unsupervised), lalu membangun model prediksi berbasis hasil pengelompokan tersebut (supervised).

---

## 🧠 Latar Belakang

Dataset ini menyajikan gambaran mendalam mengenai perilaku transaksi dan pola aktivitas keuangan nasabah, sehingga sangat ideal untuk eksplorasi **deteksi penipuan (fraud detection)** dan **identifikasi anomali**. Dengan memahami pola transaksi secara menyeluruh, lembaga keuangan dapat lebih proaktif dalam menjaga keamanan finansial nasabahnya.

---

## 📂 Struktur Proyek

```
Submission-BMLP-Unsupervised-Supervised/
│
├── 📓 -Clustering-_Submission_Akhir_BMLP_ALDINI_DZIAUL_HAQ.ipynb
│      Notebook utama untuk tahap Unsupervised Learning (K-Means Clustering)
│
├── 📓 -Klasifikasi-_Submission_Akhir_BMLP_ALDINI_DZIAUL_HAQ.ipynb
│      Notebook utama untuk tahap Supervised Learning (Decision Tree & Random Forest)
│
├── 📦 model_clustering.h5                        # Model K-Means hasil training
├── 📦 PCA_model_clustering.h5                    # Model K-Means + PCA (Advanced)
├── 📦 decision_tree_model.h5                     # Model Decision Tree
├── 📦 explore_RandomForest_classification.h5     # Model Random Forest
├── 📦 tuning_classification.h5                   # Model setelah Hyperparameter Tuning
│
├── 📄 data_clustering.csv                        # Output hasil clustering (scaled)
├── 📄 data_clustering_inverse.csv                # Output hasil clustering (nilai asli)
│
└── 📄 README.md
```

---

## 🗂️ Fitur Dataset

Dataset mencakup **2.512 sampel transaksi** dengan atribut berikut:

| Fitur | Tipe | Keterangan |
|---|---|---|
| `TransactionID` | ID | Pengidentifikasi unik setiap transaksi |
| `AccountID` | ID | ID unik akun nasabah |
| `TransactionAmount` | Numerik | Nilai nominal transaksi |
| `TransactionDate` | Date | Tanggal & waktu transaksi |
| `TransactionType` | Kategorikal | Jenis transaksi: `Credit` / `Debit` |
| `Location` | Kategorikal | Lokasi geografis transaksi |
| `Channel` | Kategorikal | Saluran transaksi: Branch / Online / ATM |
| `CustomerAge` | Numerik | Usia nasabah |
| `CustomerOccupation` | Kategorikal | Pekerjaan nasabah |
| `TransactionDuration` | Numerik | Durasi transaksi (detik) |
| `LoginAttempts` | Numerik | Jumlah percobaan login |
| `AccountBalance` | Numerik | Saldo rekening nasabah |
| `AgeGroup` | Kategorikal | Kelompok usia hasil binning: Muda / Dewasa / Senior |

> Kolom `TransactionID`, `AccountID`, `TransactionDate`, dan IP Address di-drop sebelum pemodelan.

---

## 🔄 Alur Proyek

```
📥 Load Dataset
      ↓
🔍 EDA & Visualisasi (Heatmap, Histogram, Boxplot, Violinplot)
      ↓
🧹 Preprocessing
   ├─ Drop missing values & duplikat (21 baris)
   ├─ Drop kolom ID & Date
   ├─ Label Encoding (fitur kategorikal)
   ├─ Handling Outlier (IQR Method)
   ├─ Feature Binning: CustomerAge → AgeGroup
   └─ Standard Scaling (fitur numerik)
      ↓
🔵 UNSUPERVISED — K-Means Clustering
   ├─ Elbow Method (Silhouette Score, k=2–10)
   ├─ Fit K-Means → 2 Cluster optimal
   ├─ Visualisasi PCA 2D
   └─ Inverse Transform → Interpretasi Cluster
      ↓
🟢 SUPERVISED — Klasifikasi
   ├─ Load data_clustering_inverse.csv
   ├─ One-Hot Encoding
   ├─ Train/Test Split (80:20, stratified)
   ├─ Decision Tree Classifier
   ├─ Random Forest Classifier
   └─ Hyperparameter Tuning (GridSearchCV)
      ↓
📊 Evaluasi & Simpan Model
```

---

## 🔵 Bagian 1 — Unsupervised Learning: Clustering

### Preprocessing
- Menghapus **missing values** dan **21 data duplikat**
- Drop kolom tidak relevan (ID, Date, IP)
- **Label Encoding** untuk semua fitur kategorikal
- **Handling Outlier** menggunakan metode IQR
- **Feature Binning**: `CustomerAge` → `AgeGroup` (Muda / Dewasa / Senior) via `pd.qcut`
- **Standard Scaling** untuk fitur numerik

### Pemodelan
- Algoritma: **K-Means Clustering** (`sklearn.cluster.KMeans`)
- Penentuan jumlah cluster optimal: **KElbowVisualizer** dengan metrik `silhouette`, range k=2–10
- **Jumlah cluster optimal: 2**
- Evaluasi: **Silhouette Score**
- Visualisasi: **PCA 2 komponen** (scatter plot berwarna per cluster)
- Advanced: K-Means berbasis data PCA (`PCA_model_clustering.h5`)

### Hasil Interpretasi Cluster (setelah Inverse Transform)

**CLUSTER 0 — Nasabah Dewasa, Transaksi Stabil**

| Fitur | Mean | Min | Max |
|---|---|---|---|
| TransactionAmount | 255.55 | 0.32 | 903.19 |
| CustomerAge | 45.06 | 18.00 | 80.00 |
| TransactionDuration | 121.12 | 10.00 | 300.00 |
| LoginAttempts | 1.00 | 1.00 | 1.00 |
| AccountBalance | 5,142.17 | 117.98 | 14,942.78 |
| TransactionType (mode) | Debit | | |
| Location (mode) | Charlotte | | |
| Channel (mode) | Branch | | |
| CustomerOccupation (mode) | Doctor | | |
| AgeGroup (mode) | Dewasa | | |

> Cluster 0 merepresentasikan nasabah dewasa (±45 tahun) yang berprofesi dominan sebagai Dokter. Saldo akun lebih tinggi (5.142) dengan transaksi stabil melalui Branch secara Debit. LoginAttempts yang seragam (1) menunjukkan perilaku transaksi yang normal dan terpercaya.

---

**CLUSTER 1 — Nasabah Muda, Transaksi Lebih Aktif**

| Fitur | Mean | Min | Max |
|---|---|---|---|
| TransactionAmount | 258.15 | 0.26 | 889.01 |
| CustomerAge | 44.33 | 18.00 | 80.00 |
| TransactionDuration | 117.30 | 10.00 | 299.00 |
| LoginAttempts | 1.00 | 1.00 | 1.00 |
| AccountBalance | 5,058.81 | 102.20 | 14,977.99 |
| TransactionType (mode) | Debit | | |
| Location (mode) | Tucson | | |
| Channel (mode) | Branch | | |
| CustomerOccupation (mode) | Student | | |
| AgeGroup (mode) | Muda | | |

> Cluster 1 merepresentasikan nasabah muda yang berprofesi dominan sebagai Mahasiswa. Nilai transaksi sedikit lebih tinggi (258.15) namun saldo lebih rendah (5.058) dibanding Cluster 0 — mengindikasikan pengeluaran yang lebih agresif. Berlokasi dominan di Tucson dengan durasi transaksi lebih singkat (117 detik).

---

## 🟢 Bagian 2 — Supervised Learning: Klasifikasi

### Preprocessing
- Input: `data_clustering_inverse.csv` (hasil inverse transform dari tahap clustering)
- **One-Hot Encoding** untuk fitur kategorikal
- **Train/Test Split**: 80% training, 20% testing, `stratify=y`, `random_state=42`

### Model yang Dibangun

| Model | File |
|---|---|
| Decision Tree (baseline) | `decision_tree_model.h5` |
| Random Forest | `explore_RandomForest_classification.h5` |
| Decision Tree + GridSearchCV | `tuning_classification.h5` |

### Hyperparameter Tuning
```python
params = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
# GridSearchCV, cv=5, scoring='accuracy'
```

### Evaluasi Model

Semua model dievaluasi menggunakan `classification_report` dari scikit-learn yang mencakup:

| Metrik | Keterangan |
|---|---|
| **Accuracy** | Proporsi prediksi yang benar |
| **Precision** | Ketepatan prediksi per kelas |
| **Recall** | Kelengkapan prediksi per kelas |
| **F1-Score** | Harmonic mean precision & recall |

---

## 🛠️ Teknologi yang Digunakan

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerik-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisasi-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualisasi-4c72b0)
![Yellowbrick](https://img.shields.io/badge/Yellowbrick-Elbow%20Method-9b59b6)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

---

## 🚀 Cara Menjalankan

### 1. Clone repositori
```bash
git clone https://github.com/aldinidziaulhaq/Submission-BMLP-Unsupervised-Supervised-.git
cd Submission-BMLP-Unsupervised-Supervised-
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick joblib
```

### 3. Jalankan notebook secara berurutan

```
1️⃣  -Clustering-_Submission_Akhir_BMLP_ALDINI_DZIAUL_HAQ.ipynb
2️⃣  -Klasifikasi-_Submission_Akhir_BMLP_ALDINI_DZIAUL_HAQ.ipynb
```

> ⚠️ Notebook clustering **harus dijalankan lebih dulu** karena notebook klasifikasi membutuhkan `data_clustering_inverse.csv` sebagai input.

---

## 👤 Author

**Aldini Dziaulhaq**  
🔗 [GitHub](https://github.com/aldinidziaulhaq)

---

## 📄 Lisensi

Proyek ini dibuat sebagai bagian dari submission kelas **Belajar Machine Learning untuk Pemula** di [Dicoding Indonesia](https://www.dicoding.com/).
