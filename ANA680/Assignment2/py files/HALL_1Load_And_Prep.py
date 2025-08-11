#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer (UCI Original) — Shared Loader & Helpers
# 
# This section sets up **shared utilities** to reuse across all model scripts:
# - Load the **UCI Breast Cancer Wisconsin (Original)** dataset **directly from the source**.
# - Clean and prepare the data.
# - Provide a **stratified train/test split** (75/25).
# 

# ## Dataset Source (UCI – Original)
# 
# We will use the original dataset hosted by UCI:
# 
# - URL: `https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)`  
# - Direct data file: `breast-cancer-wisconsin.data`
# 
# **Columns (per UCI):**
# 1. `sample_code_number` *(ID — will be dropped)*  
# 2. `clump_thickness`  
# 3. `uniformity_cell_size`  
# 4. `uniformity_cell_shape`  
# 5. `marginal_adhesion`  
# 6. `single_epithelial_cell_size`  
# 7. `bare_nuclei` *(contains missing values encoded as '?')*  
# 8. `bland_chromatin`  
# 9. `normal_nucleoli`  
# 10. `mitoses`  
# 11. `class` *(2 = benign, 4 = malignant)*
# 
# **Target mapping for classifiers:**  
# - `2 → 0` (benign)  
# - `4 → 1` (malignant)
# 

# ## Preprocessing Plan
# 
# 1. **Load** the raw CSV directly from the UCI URL (no local file).  
# 2. **Handle missing values**: Replace `'?'` with `NaN` and drop rows with missing values (only affects `bare_nuclei`).  
# 3. **Drop ID**: Remove `sample_code_number`.  
# 4. **Map labels**: `class` values `2→0` (benign) and `4→1` (malignant).  
# 5. **Return features/labels** as `X, y`.  
# 6. **Split** with `train_test_split` using `test_size=0.25`, `random_state=42`, and `stratify=y`.
# 

# In[9]:


# Imports & constants
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

COLS = [
    "sample_code_number",            # ID (drop)
    "clump_thickness",
    "uniformity_cell_size",
    "uniformity_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
    "class"                          # 2=benign, 4=malignant
]


# ## `load_clean_data()`
# 
# - Reads from the UCI URL.  
# - Replaces `'?'` with `NaN` (in `bare_nuclei`).  
# - Drops rows with missing values.  
# - Drops the ID column.  
# - Maps `class` from `{2:0, 4:1}`.  
# - Returns `X, y`.
# 

# In[10]:


def load_clean_data():
    # Read directly from UCI (no local CSV)
    df = pd.read_csv(UCI_URL, header=None, names=COLS)
    # Handle missing values marked as "?"
    df = df.replace("?", pd.NA)
    df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"], errors="coerce")
    df = df.dropna().copy()

    # Drop ID column
    df = df.drop(columns=["sample_code_number"])

    # Map class: 2 -> 0 (benign), 4 -> 1 (malignant)
    df["class"] = df["class"].map({2: 0, 4: 1})

    X = df.drop(columns=["class"])
    y = df["class"].astype(int)
    return X, y


# ## `get_data_splits(test_size=0.25, random_state=42)`
# 
# - Returns a stratified 75/25 train/test split.  
# - Keeps results reproducible via `random_state`.
# 

# In[11]:


def get_data_splits(test_size=0.25, random_state=42):
    X, y = load_clean_data()
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# ## `evaluate_and_log(model_name, y_true, y_pred, results_dir="results")`
# 
# Logs metrics for compiling the **Word table later**:
# 
# - Appends one row per model to `results/metrics.csv` with: `model`, `accuracy`.  
# - Appends one row per model to `results/confusion_matrices.csv` with: `model`, `tn`, `fp`, `fn`, `tp`.  
# - Prints summary to the console.
# 

# In[12]:


def evaluate_and_log(model_name, y_true, y_pred, results_dir="results"):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)  # [[TN, FP], [FN, TP]]

    os.makedirs(results_dir, exist_ok=True)

    # Append metrics row
    metrics_path = os.path.join(results_dir, "metrics.csv")
    mrow = pd.DataFrame([{
        "model": model_name,
        "accuracy": acc
    }])
    if os.path.exists(metrics_path):
        mrow.to_csv(metrics_path, mode="a", header=False, index=False)
    else:
        mrow.to_csv(metrics_path, index=False)

    # Append confusion matrix row as a flat record
    cm_path = os.path.join(results_dir, "confusion_matrices.csv")
    cmrow = pd.DataFrame([{
        "model": model_name,
        "tn": int(cm[0,0]),
        "fp": int(cm[0,1]),
        "fn": int(cm[1,0]),
        "tp": int(cm[1,1]),
    }])
    if os.path.exists(cm_path):
        cmrow.to_csv(cm_path, mode="a", header=False, index=False)
    else:
        cmrow.to_csv(cm_path, index=False)

    print(f"{model_name} | Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)


# ## Preview the Cleaned Dataset
# 
# Let’s load the data and inspect the first few rows to confirm:
# - Missing values are gone  
# - `sample_code_number` is dropped  
# - `class` is mapped to `0` (benign) and `1` (malignant)
# 

# In[13]:


# Preview cleaned data
X, y = load_clean_data()
print("Features shape:", X.shape)
print("Labels distribution:\n", y.value_counts())

# Show first 5 rows
pd.concat([X, y], axis=1).head()


# ## Sanity check: class balance after split
# Confirm stratification worked and the test size is 25%.
# 

# In[15]:


X_train, X_test, y_train, y_test = get_data_splits()
print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
print("Train label distribution:\n", y_train.value_counts(normalize=True).round(3))
print("Test  label distribution:\n", y_test.value_counts(normalize=True).round(3))


# In[ ]:




