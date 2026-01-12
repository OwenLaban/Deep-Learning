# Machine Learning Coursework & UTS – Owen

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Tools-Google_Colab-orange?style=for-the-badge&logo=googlecolab&logoColor=white)

## 👤 Student Identification
> This repository is submitted as part of the Machine Learning & Deep Learning coursework.

* **Name:** Josua Owen Fernandi Silaban  
* **Class:** TK-46-04  
* **NIM:** 1103223117  

---

## 📌 Repository Purpose

This repository documents my learning journey in **Machine Learning** and **Deep Learning** during the course.  
It is organized to show my progress from basic data processing to building complete, end-to-end ML/DL pipelines for:

1. **Supervised Learning**
   * Regression – predicting continuous targets (e.g. song release year).
   * Classification – fraud detection on transaction data.
2. **Unsupervised Learning**
   * Customer clustering and representation learning using autoencoders.
3. **Model Evaluation & Interpretation**
   * Using appropriate metrics (MSE, RMSE, R², AUC, F1-score, Silhouette, etc.).
   * Visualizing model performance and feature importance.

---

## 📚 Weekly Assignments

The `Weekly_Assignments` folder (or equivalent) summarizes my step-by-step learning:

- **Chapter 1–3**
  - Python refresher, NumPy arrays, Pandas DataFrame operations.
  - Basic visualizations: line plots, histograms, boxplots.

- **Chapter 4–6**
  - Data cleaning (missing values, outliers, scaling).
  - Intro to supervised learning: Linear Regression, Logistic Regression, KNN, basic model metrics.

- **Chapter 7–8**
  - Train/validation/test split, cross-validation.
  - Model tuning (GridSearch / RandomizedSearch).
  - Handling imbalanced datasets and basic feature engineering.

> Exact notebook names may differ depending on the course template used.

---

## 📂 UTS – End-to-End Projects

The `UTS` folder contains **three main end-to-end projects** based on the official mid-term instructions.

### 1. 🕵️‍♂️ End-To-End Fraud Detection (Classification)

* **File:** `UTS_Fraud_Detection.ipynb`  
* **Dataset:** `train_transaction.csv`, `test_transaction.csv` (with `isFraud` label in train).  
* **Objective:**  
  Build a classifier that predicts the probability that a transaction is fraudulent (`isFraud = 1`), while handling **class imbalance**.

**Main Steps**

- **Data Loading & Cleaning**
  - Load train & test transaction data from Kaggle-like dataset.
  - Select a subset of informative features (e.g. `TransactionAmt`, `ProductCD`, `card4`, `card6`, `addr1`, `dist1`, etc.).
  - Handle missing values using `SimpleImputer` (median/most_frequent).

- **Preprocessing Pipeline**
  - Separate **numeric** and **categorical** features.
  - Numeric: median imputation + `StandardScaler`.
  - Categorical: most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`.
  - Combine using `ColumnTransformer`.

- **Models Explored**
  - **Logistic Regression** with `class_weight='balanced'`.
  - **Random Forest Classifier** with `class_weight='balanced'`.

- **Evaluation**
  - Train/validation split with `stratify=y`.
  - Metrics:
    - **ROC-AUC** (main metric).
    - **Classification report** (Precision, Recall, F1).
    - **Confusion Matrix** to inspect False Positive / False Negative.
    - ROC Curve plots and class distribution plot to show imbalance.

- **Prediction & Submission**
  - Apply the same preprocessing pipeline to `test_transaction.csv`.
  - Use the best model to generate `isFraud` probabilities.
  - Save as `submission_fraud.csv` with columns: `TransactionID`, `isFraud`.

---

### 2. 📈 End-To-End Regression Pipeline (Song Year Prediction)

* **File:** `UTS_Regression_Pipeline.ipynb`  
* **Dataset:** `midterm-regresi-dataset.csv` (first column = target year, remaining columns = audio features).  
* **Objective:**  
  Predict the **release year** of a song based on its numerical audio features.

**Main Steps**

- **Data Understanding**
  - Load dataset without header; column 0 used as `year` target.
  - Rename features as `feature_1`, `feature_2`, … for clarity.
  - (Optional) **Row sampling** and `float32` casting to reduce RAM usage.

- **Preprocessing**
  - Clip extreme values (1%–99%) to reduce the effect of outliers.
  - Split into train/test sets.
  - Apply `StandardScaler` for all numeric features.

- **Models Explored**
  - **Linear Regression** as baseline.
  - **Random Forest Regressor** as a non-linear model.
  - (Optional) **Deep Learning Regressor** using a small Dense Neural Network in TensorFlow/Keras.

- **Evaluation Metrics**
  - **MSE** (Mean Squared Error).
  - **RMSE** (Root Mean Squared Error).
  - **MAE** (Mean Absolute Error).
  - **R² Score** (coefficient of determination).

- **Visualization**
  - Plot **True vs Predicted year** for the best model.
  - Distribution of target `year` to understand the prediction difficulty (older vs newer songs).

---

### 3. 👥 Customer Clustering with Autoencoder (Unsupervised)

* **File:** `UTS_Customer_Clustering.ipynb`  
* **Dataset:** `clusteringmidterm.csv` (credit card usage & payment behavior).  
* **Objective:**  
  Segment customers into groups based on their transaction behavior, using a combination of **Autoencoder** and **K-Means Clustering**.

**Main Steps**

- **Preprocessing**
  - Drop `CUST_ID`.
  - Handle missing values (median filling).
  - Standardize all numeric features (`StandardScaler`, `float32` for memory efficiency).
  - (Optional) Row sampling for Colab RAM limitation.

- **Representation Learning (Autoencoder)**
  - Build a symmetric autoencoder with a low-dimensional latent space (e.g. 4–8 neurons).
  - Train to reconstruct input features using MSE loss.
  - Extract the **latent representation** (bottleneck layer) for each customer.

- **Clustering**
  - Run **K-Means** on latent features for various `k`.
  - Use **Elbow method** and **Silhouette Score** to select the best number of clusters.
  - Assign cluster labels back to customers.

- **Analysis**
  - Compute mean statistics per cluster (average balance, purchase frequency, cash advance, etc.).
  - Interpret business meaning of each cluster (e.g., “high spender”, “cash advance heavy user”, “low usage customer”).

---

## 🚀 How to Run the Notebooks

1. **Clone or Download the Repository**
   ```bash
   git clone https://github.com/owenlaban/Machine-Learning.git
   cd Machine-Learning

---

## 🎯 UAS Purpose

UAS ini dirancang sebagai rangkuman dari materi **Deep Learning** selama satu semester.  
Di dalam folder `UAS/` terdapat **tiga notebook utama**:

- `Task 1.ipynb`  
- `Task 2.ipynb`  
- `Task 3.ipynb`  

Masing-masing notebook membangun **end-to-end deep learning pipeline**:

> **Load data → Preprocess → Build DL model (Keras/TensorFlow) → Train + Evaluate → Interpretasi.**

Secara garis besar:

1. **Task 1 – Deep Learning for Classification**  
   - Fully-connected / ANN classifier untuk data tabular (misalnya fraud detection atau klasifikasi biner lainnya).  
2. **Task 2 – Deep Learning for Regression**  
   - ANN regressor untuk memprediksi nilai kontinu (misalnya tahun rilis / skor numerik).  
3. **Task 3 – Representation Learning & Clustering**  
   - Autoencoder untuk belajar representasi laten + analisis cluster (misalnya customer segmentation).

Struktur dan jenis task menyesuaikan instruksi resmi UAS; isi masing-masing notebook bersifat mandiri namun konsisten dengan alur yang sama.

---

## 📂 Project Structure (UAS Folder)

```text
UAS/
├── Task 1.ipynb   # Deep learning untuk klasifikasi
├── Task 2.ipynb   # Deep learning untuk regresi
└── Task 3.ipynb   # Autoencoder & clustering / representation learning
