# Feature Selection on the Heart Disease UCI Dataset

## 📘 Overview
This notebook explores **feature selection techniques** to identify the most predictive variables for heart disease diagnosis.  
Feature selection improves model performance, interpretability, and computational efficiency by removing redundant or irrelevant features.  

Using the **Heart Disease UCI dataset**, we compare multiple techniques — from simple correlation filters to advanced embedded methods — and evaluate their effect on a Random Forest classifier’s performance.  

---

## Dataset Information

- **Dataset:** Heart Disease UCI  
- **Source:** [Kaggle - Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)  
- **Instances:** 920 records  
- **Features:** 16 (mix of categorical, numerical, and boolean)  
- **Target variable:** `num` → converted to binary `target` (1 = Heart Disease, 0 = No Disease)

### Key Preprocessing Steps
1. Removed unnecessary columns (`id`, etc.).
2. Encoded categorical values such as sex, chest pain type, and thal.
3. Handled missing values and normalized numerical features using `StandardScaler`.
4. Converted the multi-class column `num` into binary:  
   ```python
   df["target"] = (df["num"] > 0).astype(int)
   ```

---

## Methodology

### Step 1 – Data Loading & Cleaning
- Loaded `heart_disease_uci.csv` from `./Data/`.
- Inspected missing values and datatypes.
- Removed irrelevant or empty columns.
- Converted the target column into binary.

### Step 2 – Data Visualization
- Used Seaborn’s `countplot` and correlation heatmaps to explore relationships.
- Identified strongly correlated numeric features with `target`.

### Step 3 – Feature and Target Split
- Split dataset into:
  ```python
  X = df.drop("target", axis=1)
  y = df["target"]
  ```
- Then standardized features using:
  ```python
  StandardScaler().fit(X_train)
  ```

### Step 4 – Baseline Model
- Trained a `RandomForestClassifier` using all features.
- Recorded Accuracy, Precision, Recall, F1, and ROC-AUC.
- Served as the baseline for comparison.

### Step 5 – Correlation-Based Filter
- Computed feature-target correlations using Pearson correlation.
- Selected features where `|corr| > 0.15`.
- Retrained the model on these reduced features.
- Observed similar performance with fewer variables — meaning some redundancy was removed.

### Step 6 – ANOVA F-test (SelectKBest)
- Used `SelectKBest(f_classif, k=8)` to select top 8 statistically significant features.
- Works best for numeric features and categorical target.
- Performance was comparable to correlation-based selection.

### Step 7 – Recursive Feature Elimination (RFE)
- Used `RFE(RandomForestClassifier, n_features_to_select=8)` to iteratively remove weak features.
- Slightly lower performance but useful for understanding feature importance hierarchies.

### Step 8 – Embedded Method (Tree-Based Importance)
- Used Random Forest’s intrinsic `feature_importances_` property.
- Selected features with importance > 0.04.
- Achieved strong F1 performance with fewer features (~6–8).

### Step 9 – Embedded Method (L1 Regularization via LinearSVC)
- Used `LinearSVC` with L1 penalty to enforce sparsity.
- Automatically eliminated features with near-zero coefficients.
- Slightly reduced performance, but excellent dimensionality reduction.

### Step 10 – Comparison Automation
- Each model’s metrics were logged using:
  ```python
  record_results(method_name, y_test, y_pred, feature_count)
  ```
- Stored in a global list `results` and displayed at the end.

### Step 11 – Summary and Visualization
- Compiled all results into a single DataFrame.
- Displayed:
  - Gradient-colored performance table  
  - Bar plots comparing F1 Score and Feature Count  
- Automatically identified the **best-performing method**.

---

## Results Summary

| Method | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Feature Count |
|---------|-----------|------------|----------|-----------|------------|----------------|
| Tree-Based Importance | 0.88 | 0.87 | 0.88 | **0.87** | 0.89 | 6 |
| ANOVA F-Test (Top 8) | 0.87 | 0.86 | 0.86 | 0.86 | 0.88 | 8 |
| Correlation > 0.15 | 0.86 | 0.85 | 0.85 | 0.85 | 0.87 | 9 |
| RFE (Top 8) | 0.86 | 0.85 | 0.84 | 0.84 | 0.87 | 8 |
| L1 Regularization | 0.85 | 0.84 | 0.83 | 0.83 | 0.86 | 7 |
| All Features (Baseline) | 0.88 | 0.87 | 0.88 | 0.87 | 0.90 | 14 |

### Observations
- The **Tree-Based Importance** method achieved the best trade-off between accuracy and feature count.  
- Removing redundant or correlated features did **not degrade performance**, proving that simpler models can be just as effective.
- The ANOVA F-Test also provided competitive results with fewer features.

---

## 🔬 How This Differs from the Original Breast Cancer Feature Selection Lab

| Aspect | Breast Cancer Dataset | Heart Disease Dataset |
|---------|------------------------|-----------------------|
| **Dataset Type** | Numeric-only biomedical measurements (e.g., radius, area, texture) | Mixed data types (categorical, boolean, numeric) |
| **Target Variable** | Binary (Malignant vs Benign) | Multiclass originally, converted to binary (Disease vs No Disease) |
| **Preprocessing** | Minimal — mostly numeric normalization | Extensive — categorical encoding, missing value handling, logical mapping |
| **Feature Relationships** | High correlation between similar measures | Weaker, more heterogeneous correlations |
| **Goal of Exercise** | Compare selection techniques conceptually | Apply them in a more realistic, messy dataset with mixed datatypes |
| **Outcome** | F-test and Correlation performed best | Tree-Based model importance performed best |

**In short:**  
> The Heart Disease version applies the same theory to a more complex, mixed-type dataset — demonstrating how feature selection scales to real-world medical data with categorical and numerical features combined.

---

## How to Run This Notebook

1. **Clone or open your project directory:**
   ```bash
   cd Lab_10/Model_Developemet_Labs/Feature_Selection/
   ```
2. Ensure your structure looks like:
   ```
   Feature_Selection/
   ├── Data/
   │   └── heart_disease_uci.csv
   ├── feature_selection_assignment.ipynb
   └── README.md
   ```
3. **Install requirements:**
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```
4. **Run the notebook in Jupyter or VS Code:**
   ```bash
   jupyter notebook feature_selection_assignment.ipynb
   ```
5. **Execute cells 1 → 11 sequentially** to reproduce results.

---

## Final Thoughts
Feature selection not only reduces model complexity but also helps identify the most informative predictors of heart disease.  
This Heart Disease project demonstrates how statistical, wrapper, and embedded methods work in tandem to improve real-world models.

---
