# UTS – End-to-End Machine Learning & Deep Learning Pipelines

This repository contains three end-to-end projects for the Machine Learning & Deep Learning midterm:

1. **Customer Clustering** – Unsupervised learning (ML + Autoencoder).  
2. **Fraud Detection** – Binary classification (ML + Deep Learning).  
3. **Regression** – Predicting a continuous target (ML + Deep Learning).

All notebooks are designed to be runnable in **Google Colab** with minimal setup.

---

## 1. Customer Clustering (Unsupervised Learning)

**Notebook:** `Customer_Clustering_ML.ipynb`  
**Task:** Segment customers based on their credit card usage patterns.

### Dataset

- **Source:** Google Drive (credit card customer dataset).  
- **File:** `clusteringmidterm.csv`  
- Each row represents one customer; columns describe behaviors such as:
  `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `PAYMENTS`,
  `PURCHASES_FREQUENCY`, `CASH_ADVANCE_FREQUENCY`,
  `PRC_FULL_PAYMENT`, `TENURE`, etc.

### Pipeline Overview

1. **Data Loading & Cleaning**
   - Load dataset from Google Drive.
   - Handle missing values (imputation / dropping if needed).
   - Remove non-informative columns and check data types.

2. **Preprocessing**
   - Separate numeric and categorical features (if any).
   - Apply **StandardScaler** to numeric features.
   - Prepare `df_scaled` for traditional ML models.

3. **Optimal Number of Clusters**
   - **Elbow Method** using Within-Cluster Sum of Squares (inertia).
   - **Silhouette Score** analysis for different `k`.
   - Select a reasonable `k` (e.g., `k = 4`) for comparison.

4. **Clustering Algorithms (ML)**
   - **K-Means** on `df_scaled` (baseline clustering).
   - Evaluation:
     - Silhouette Score
     - Davies–Bouldin Index
     - Calinski–Harabasz Score
   - Cluster size distribution.

5. **Deep Learning: Autoencoder + K-Means**
   - Build a **symmetric autoencoder** with:
     - Encoder → latent dimension `z` (e.g. 8).
     - Decoder → reconstruct original features.
   - Train with **MSE loss** and **Adam optimizer**.
   - Use **EarlyStopping** on `val_loss`.
   - Extract latent features (`latent_df`) from the bottleneck layer.
   - Run **K-Means** on `latent_df` and evaluate using Silhouette Score.

6. **Cluster Profiling & Business Interpretation**
   - For both:
     - **K-Means (df_scaled)**
     - **AE + K-Means (latent_df)**
   - Create summary tables (mean of key features per cluster).
   - Interpret segments, for example:
     - Active Purchasers (healthy usage)
     - Cash-Advance–heavy, higher risk
     - Heavy cash-advance but disciplined payer
     - Mixed users with high balances
   - Compare cluster quality:
     - Example result: K-Means Silhouette ≈ 0.92 vs AE+KMeans ≈ 0.79.

7. **Key Takeaways**
   - For this dataset, **K-Means + StandardScaler** gives the best cluster separation.
   - **Autoencoder** is useful for representation learning, but does not always outperform simple ML for clustering on moderate-sized tabular data.
   - Clusters can be used for targeted marketing and risk management strategies.

---

## 2. Fraud Detection (Binary Classification)

**Notebook:** `E2E_Fraud_Detection_ML.ipynb`  
**Task:** Predict whether an online transaction is fraudulent (`isFraud`).

### Dataset

- **Source:** Google Drive (fraud transaction dataset).  
- **Files:**
  - `train_transaction.csv` – labeled training set with `isFraud`.
  - `test_transaction.csv` – unlabeled test set for submission.
- Each row is a single transaction with many features:
  amount, time, product code, card information, address, device info, etc.

### Pipeline Overview

1. **Data Loading & Merging**
   - Load `train_transaction.csv` (and optional identity table if provided).
   - Split into:
     - Features `X`
     - Target `y = isFraud`.

2. **Preprocessing**
   - Identify **numeric** and **categorical** features.
   - Use `ColumnTransformer`:
     - `StandardScaler` for numeric columns.
     - `OneHotEncoder` for categorical columns (`handle_unknown="ignore"`).
   - Split into **train / validation** sets.

3. **Machine Learning Baselines**
   - Models typically used:
     - **Logistic Regression**
     - **Random Forest** (or other tree-based model).
   - Evaluation on validation set:
     - **ROC–AUC** (main metric)
     - Precision, Recall, F1-score
     - Confusion matrix (focus on fraud recall).

4. **Class Imbalance Handling**
   - Fraud cases are rare → class imbalance.
   - Strategies:
     - `class_weight="balanced"` for some models.
     - Evaluation emphasizes **Recall of fraud class** and **AUC**.

5. **Deep Learning Model (ANN)**
   - Reuse the same `preprocessor` to create:
     - `X_train_dl`, `X_val_dl`, `X_test_dl` (float32).
   - Compute **class weights** from `y_train`.
   - Build an **ANN classifier**:
     - Dense(128, ReLU) → Dropout(0.3)
     - Dense(64, ReLU) → Dropout(0.3)
     - Dense(1, Sigmoid) – output probability of fraud.
   - Compile:
     - Loss: `binary_crossentropy`
     - Optimizer: `Adam`
     - Metrics: `AUC`, `Precision`, `Recall`.
   - Use **EarlyStopping** on `val_auc` with `restore_best_weights=True`.

6. **Evaluation & Comparison**
   - For the ANN:
     - Compute validation **AUC**, precision, recall.
     - Show classification report & confusion matrix.
   - Compare:
     - Logistic Regression AUC vs Random Forest AUC vs ANN AUC.
   - Analyze trade-offs between:
     - Catching more fraud (higher recall) vs false alarms.

7. **Submission**
   - Generate fraud probabilities for `test_transaction.csv`.
   - Create submission file:
     - `TransactionID`, `isFraud` (probabilities from best model).

8. **Key Takeaways**
   - With a consistent preprocessing pipeline, both **Scikit-Learn models** and **Keras ANN** can be compared fairly using ROC–AUC.
   - On this dataset, the chosen final model balances:
     - High AUC
     - Sufficient recall for fraud class
     - Acceptable false positive rate.

---

## 3. Regression – Song Release Year Prediction

**Notebook:** `E2E_Regression_Pipeline.ipynb`  
**Task:** Predict a continuous target, e.g., **song release year**, from audio features.

### Dataset

- **Source:** Google Drive.  
- **File:** `midterm-regresi-dataset.csv`  
- Characteristics:
  - No header row.
  - **First column**: target label (e.g., release year).  
  - Remaining columns: numeric features derived from the audio signal.

### Pipeline Overview

1. **Data Loading**
   - Download dataset from Google Drive using `gdown`.
   - Load with `pandas.read_csv(header=None)`.
   - Split into:
     - `y` = first column (target).
     - `X` = remaining columns (features).

2. **Train–Validation Split**
   - Use `train_test_split` (e.g., 80% train, 20% validation).
   - Random seed fixed (e.g., 42) for reproducibility.

3. **Preprocessing**
   - Apply **StandardScaler** to all numeric features.
   - Obtain:
     - `X_train_scaled`
     - `X_val_scaled`.

4. **Machine Learning Models**
   - **Baseline model** (e.g., predict mean year) as a naive reference.
   - **Linear Regression** as a simple parametric model.
   - Optionally:
     - Random Forest Regressor / Gradient Boosting for non-linear patterns.
   - Evaluation metrics:
     - **MSE** (Mean Squared Error)
     - **RMSE** (Root Mean Squared Error)
     - **MAE** (Mean Absolute Error)
     - **R²** (Coefficient of Determination)

5. **Deep Learning Regression (ANN)**
   - Use the scaled features as input to a neural network:
     - Dense(128, ReLU) → Dropout(0.3)
     - Dense(64, ReLU) → Dropout(0.3)
     - Dense(1, Linear) – predicts a single continuous value (year).
   - Compile:
     - Loss: `mse`
     - Optimizer: `Adam`
     - Metrics: `mae`, `mse`.
   - Train with **EarlyStopping** on `val_loss` and appropriate `batch_size`.
   - Plot training vs validation loss curves.

6. **Evaluation & Comparison**
   - Compute MSE, RMSE, MAE, R² on the validation set for:
     - Linear Regression
     - Tree-based model (if used)
     - ANN regression
   - Compare performance:
     - Lower MSE/RMSE/MAE → better.
     - Higher R² → better fit.
   - Decide whether deep learning provides meaningful improvement over classical ML for this dataset.

7. **Key Takeaways**
   - Demonstrates an **end-to-end regression pipeline** with:
     - Data loading
     - Preprocessing
     - ML baselines
     - Deep learning model
     - Unified evaluation.
   - Shows that, depending on dataset size & complexity, classical models may compete closely with or sometimes outperform deep learning, especially on tabular data.

---

## How to Run the Notebooks

1. Open **Google Colab**.
2. Upload the `.ipynb` notebook and run cells from top to bottom.  
3. Make sure:
   - Google Drive is mounted if datasets are stored there.  
   - `gdown` is installed when using Drive file IDs.  
4. Adjust paths or `file_id` variables if your dataset locations are different.

---

## Overall Learning Outcomes

Across the three notebooks, this project covers:

- **Supervised Learning**
  - Classification (fraud detection) with ML and Deep Learning.
  - Regression (year prediction) with ML and Deep Learning.
- **Unsupervised Learning**
  - Customer segmentation with K-Means and Autoencoder-based clustering.
- **End-to-End ML Engineering**
  - Data loading from Google Drive.
  - Preprocessing with `ColumnTransformer` and scalers/encoders.
  - Model training, evaluation, and comparison.
  - Result interpretation and business insights.

These pipelines can serve as templates for future projects involving tabular datasets, fraud detection, customer analytics, and numeric regression tasks.
