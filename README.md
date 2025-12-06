# Machine Learning & Deep Learning UTS – Owen

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Tools-Google_Colab-orange?style=for-the-badge&logo=googlecolab&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-yellow?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/Deep_Learning-TensorFlow-ff6f00?style=for-the-badge&logo=tensorflow&logoColor=white)

---

## 👤 Student Identification

> This repository is submitted as part of the **Machine Learning & Deep Learning** coursework / UTS.

- **Name:** Josua Owen Fernandi Silaban  
- **NIM:** 1103223117  
- **Class:** TK-46-04  

---

## 🎯 Repository Purpose

This repository documents three **end-to-end ML & DL pipelines** yang dikerjakan sebagai contoh UTS:

1. **Customer Clustering** – Unsupervised learning (K-Means + Autoencoder).  
2. **Fraud Detection** – Binary classification (ML baselines + ANN classifier).  
3. **Regression** – Predicting a continuous target (ML baselines + ANN regressor).

Setiap notebook menunjukkan alur lengkap:

> **Load data → Preprocessing → Modeling (ML & DL) → Evaluation → Interpretation.**

---

## 📂 Project Structure (Main Notebooks)

- `Customer_Clustering_ML.ipynb`  
  Unsupervised learning + Deep Learning (Autoencoder) untuk segmentasi nasabah kartu kredit.

- `E2E_Fraud_Detection_ML.ipynb`  
  End-to-end fraud detection (klasifikasi) dengan model ML dan ANN.

- `E2E_Regression_Pipeline.ipynb`  
  End-to-end regression pipeline untuk memprediksi nilai kontinu (tahun rilis lagu).

---

## 1️⃣ Customer Clustering – K-Means & Autoencoder

**File:** `Customer_Clustering_ML.ipynb`  
**Task:** Mengelompokkan nasabah berdasarkan perilaku penggunaan kartu kredit.

### Main Steps

- **Data Preparation**
  - Load dataset nasabah kartu kredit dari Google Drive.
  - Drop kolom ID (`CUST_ID`) dan handle missing values.
  - Standarisasi semua fitur numerik menggunakan `StandardScaler` → `df_scaled`.

- **Baseline Clustering (Machine Learning)**
  - Menentukan jumlah cluster dengan **Elbow Method** dan **Silhouette Score**.
  - Menerapkan **K-Means** pada `df_scaled`.
  - Menghitung metrik:
    - Silhouette Score  
    - Davies–Bouldin Index  
    - Calinski–Harabasz Score  

- **Deep Learning – Autoencoder + K-Means**
  - Membangun **autoencoder** simetris dengan latent dimension (mis. 8):
    - Encoder: beberapa Dense layer → `latent_layer`.
    - Decoder: merekonstruksi kembali fitur asli.
  - Training:
    - Loss: MSE
    - Optimizer: Adam
    - EarlyStopping pada `val_loss`.
  - Ekstraksi representasi laten → `latent_df`.
  - Jalankan **K-Means** di `latent_df` → AE + KMeans.
  - Bandingkan Silhouette K-Means asli vs AE + KMeans.

- **Cluster Profiling & Business Insight**
  - Membuat ringkasan rata-rata fitur per cluster (`cluster_summary`, `cluster_summary_ae`).
  - Interpretasi cluster, contoh:
    - **Active Purchasers** (sering belanja, pembayaran cukup sehat)  
    - **Cash-Advance Risky Users** (tarik tunai tinggi, jarang full payment)  
    - **Heavy Cash-Advance but Good Payers**  
    - **High Balance Mixed Users**  
  - Menyimpulkan model mana yang lebih baik: pada dataset ini, **K-Means + scaling** memberikan separasi cluster terbaik.

---

## 2️⃣ Fraud Detection – ML Baselines & ANN Classifier

**File:** `E2E_Fraud_Detection_ML.ipynb`  
**Task:** Memprediksi probabilitas sebuah transaksi online merupakan fraud (`isFraud`).

### Main Steps

- **Data Loading**
  - Load `train_transaction.csv` (dan `test_transaction.csv`) dari Google Drive.
  - Pisahkan:
    - `X_train`, `y_train` (label `isFraud`).
    - `X_test` untuk prediksi akhir / submission.

- **Preprocessing (Scikit-Learn)**
  - Pisah **numerical** dan **categorical** features.
  - Gunakan `ColumnTransformer`:
    - `StandardScaler` untuk numerik.
    - `OneHotEncoder(handle_unknown="ignore")` untuk kategorikal.
  - Split train–validation menggunakan `train_test_split` dengan `stratify=y`.

- **Machine Learning Baselines**
  - **Logistic Regression** (`class_weight="balanced"`).  
  - **Random Forest Classifier** (meng-handle non-linearitas).  
  - Evaluasi:
    - ROC–AUC (utama)
    - Precision, Recall, F1
    - Confusion Matrix (fokus di kelas fraud).

- **Deep Learning – ANN Classifier**
  - Gunakan preprocessing yang sama → `X_train_dl`, `X_val_dl`, `X_test_dl`.
  - Hitung **class weight** untuk mengatasi imbalance.
  - Arsitektur ANN:
    - Dense(128, ReLU) → Dropout(0.3)  
    - Dense(64, ReLU) → Dropout(0.3)  
    - Dense(1, Sigmoid)
  - Compile:
    - Loss: Binary Crossentropy
    - Optimizer: Adam
    - Metrics: AUC, Precision, Recall
  - Training dengan **EarlyStopping** pada `val_auc`.

- **Evaluation & Comparison**
  - Bandingkan:
    - AUC Logistic Regression
    - AUC Random Forest
    - AUC Deep Learning (ANN)
  - Lihat trade-off:
    - Recall fraud vs False Positive.
  - Model terbaik dipilih berdasarkan kombinasi AUC + kebutuhan bisnis (lebih penting mengurangi **fraud miss** daripada false alarm).

- **Submission (Optional)**
  - Gunakan model terbaik untuk memprediksi `isFraud` di `test_transaction.csv`.
  - Simpan sebagai `submission_fraud.csv` dengan format:
    - `TransactionID`, `isFraud` (probabilitas).

---

## 3️⃣ Regression – Song Release Year Prediction (ML & DL)

**File:** `E2E_Regression_Pipeline.ipynb`  
**Task:** Memprediksi **tahun rilis** lagu berdasarkan fitur-fitur numerik audio.

### Main Steps

- **Data Loading**
  - Download dataset dari Google Drive menggunakan `gdown`:
    - `midterm-regresi-dataset.csv`.
  - Dataset tidak memiliki header:
    - Kolom pertama → target (year).  
    - Kolom berikutnya → fitur (`feature_1`, `feature_2`, ...).

- **Preprocessing**
  - `y = df.iloc[:, 0]` (tahun rilis).  
  - `X = df.iloc[:, 1:]` (fitur).  
  - Train–validation split (mis. 80/20).
  - `StandardScaler` untuk semua fitur numerik → `X_train_scaled`, `X_val_scaled`.

- **Machine Learning Models**
  - **Baseline:** model sederhana (mis. mean predictor) sebagai referensi awal.  
  - **Linear Regression:** model regresi klasik.  
  - (Opsional) **Random Forest Regressor / Tree-based** untuk pola non-linear.  
  - Evaluasi dengan:
    - MSE, RMSE
    - MAE
    - R² (coefficient of determination)

- **Deep Learning – ANN Regressor**
  - Arsitektur:
    - Dense(128, ReLU) → Dropout(0.3)  
    - Dense(64, ReLU) → Dropout(0.3)  
    - Dense(1, Linear)
  - Compile:
    - Loss: MSE
    - Optimizer: Adam
    - Metrics: MAE, MSE
  - Training:
    - Input: `X_train_scaled`, `X_val_scaled`
    - EarlyStopping pada `val_loss`.

- **Evaluation & Comparison**
  - Hitung MSE, RMSE, MAE, R² untuk:
    - Linear Regression
    - (opsional) Random Forest
    - Deep Learning (ANN)
  - Bandingkan performa:
    - RMSE lebih rendah dan R² lebih tinggi → model lebih baik.
  - Catat apakah deep learning memberikan peningkatan signifikan dibanding model ML klasik.

---

## 🚀 How to Run

1. Buka **Google Colab**.  
2. Upload notebook (`.ipynb`) dan jalankan cell dari atas ke bawah.  
3. Pastikan:
   - Dataset tersedia (via Google Drive atau `gdown`).  
   - Path / `file_id` sudah disesuaikan dengan lokasi file kamu.  
4. Perhatikan penggunaan RAM (sampling / float32) jika dataset besar.

---

<p align="center">
  <i>Created with ❤️ by Josua Owen Fernandi Silaban for Machine Learning & Deep Learning coursework / UTS.</i>
</p>
