#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer (UCI Original) — Shared Loader & Helpers
# 
# This section sets up **shared utilities** to reuse across all model scripts:
# - Load the **UCI Breast Cancer Wisconsin (Original)** dataset **directly from the source**.
# - Clean and prepare the data.
# - Provide a **stratified train/test split** (75/25).

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

# In[2]:


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

# In[3]:


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

# In[4]:


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

# In[5]:


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

# In[6]:


# Preview cleaned data
X, y = load_clean_data()
print("Features shape:", X.shape)
print("Labels distribution:\n", y.value_counts())

# Show first 5 rows
pd.concat([X, y], axis=1).head()


# ## Sanity check: class balance after split
# Confirm stratification worked and the test size is 25%.
# 

# In[7]:


X_train, X_test, y_train, y_test = get_data_splits()
print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
print("Train label distribution:\n", y_train.value_counts(normalize=True).round(3))
print("Test  label distribution:\n", y_test.value_counts(normalize=True).round(3))


# ## Model — Logistic Regression
# 
# We’ll use a simple pipeline:
# - `StandardScaler` (helps LR converge & keeps coefficients on comparable scales)
# - `LogisticRegression(max_iter=1000, random_state=42)`
# 
# Output: accuracy and confusion matrix (via `evaluate_and_log`).
# 

# In[8]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1) get data
X_train, X_test, y_train, y_test = get_data_splits()

# 2) pipeline: scale -> logistic regression
logreg_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

# 3) fit & predict
logreg_pipe.fit(X_train, y_train)
y_pred_lr = logreg_pipe.predict(X_test)

# 4) log results (prints accuracy & confusion matrix + appends to results/)
evaluate_and_log("Logistic Regression", y_test, y_pred_lr)


# ## Summary of Results
# 
# So far, we’ve:
# - Loaded the **Breast Cancer Wisconsin (Original)** dataset directly from UCI.
# - Cleaned it by removing missing values, dropping the ID column, and mapping class labels to binary.
# - Verified that stratified splitting preserved class balance.
# - Implemented Logistic Regression with scaling, achieving:
#   - **Accuracy:** 95.91%
#   - **Confusion Matrix:** TN=106, FP=5, FN=2, TP=58
# 

# In[11]:


import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve

# make sure we have probabilities
if hasattr(logreg_pipe, "predict_proba"):
    y_proba = logreg_pipe.predict_proba(X_test)[:, 1]
else:
    # logistic regression should have predict_proba; fallback if not
    from sklearn.preprocessing import MinMaxScaler
    y_proba = MinMaxScaler().fit_transform(logreg_pipe.decision_function(X_test).reshape(-1,1)).ravel()

y_pred = (y_proba >= 0.5).astype(int)  # current threshold

# confusion matrix & derived metrics
cm = confusion_matrix(y_test, y_pred)  # [[tn, fp], [fn, tp]]
tn, fp, fn, tp = cm.ravel()
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)            # sensitivity
spec = tn / (tn + fp) if (tn + fp) else np.nan # specificity
f1   = f1_score(y_test, y_pred)

# curves
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc     = roc_auc_score(y_test, y_proba)

prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
ap          = average_precision_score(y_test, y_proba)

# calibration (reliability)
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="uniform")

metrics_summary = {
    "Accuracy": acc, "Precision": prec, "Recall (Sensitivity)": rec,
    "Specificity": spec, "F1": f1, "ROC AUC": roc_auc, "PR AUC (AP)": ap,
    "TN": tn, "FP": fp, "FN": fn, "TP": tp
}
metrics_summary


# In[12]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
(ax_cm, ax_roc), (ax_pr, ax_cal) = axes

# 1) Confusion matrix heatmap with annotations
im = ax_cm.imshow(cm, cmap="Blues")
ax_cm.set_title("Confusion Matrix")
ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
ax_cm.set_xticks([0,1]); ax_cm.set_xticklabels(["Benign (0)","Malignant (1)"])
ax_cm.set_yticks([0,1]); ax_cm.set_yticklabels(["Benign (0)","Malignant (1)"])
for (i,j), val in np.ndenumerate(cm):
    total = cm[i].sum()
    pct = (val/total*100) if total else 0
    ax_cm.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", fontsize=10, color="black")
fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

# 2) ROC curve
ax_roc.plot([0,1], [0,1], linestyle="--")
ax_roc.plot(fpr, tpr, linewidth=2)
ax_roc.set_title(f"ROC Curve (AUC = {roc_auc:.3f})")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.grid(alpha=0.3)

# 3) Precision–Recall curve
ax_pr.plot(rec_curve, prec_curve, linewidth=2)
ax_pr.set_title(f"Precision–Recall (AP = {ap:.3f})")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_xlim(0,1); ax_pr.set_ylim(0,1)
ax_pr.grid(alpha=0.3)

# 4) Calibration (reliability) curve
ax_cal.plot([0,1],[0,1], linestyle="--", label="Perfectly calibrated")
ax_cal.plot(prob_pred, prob_true, marker="o", linewidth=1.5, label="LogReg")
ax_cal.set_title("Calibration Curve")
ax_cal.set_xlabel("Predicted probability")
ax_cal.set_ylabel("Observed frequency")
ax_cal.legend()
ax_cal.grid(alpha=0.3)

# overall title w/ key metrics
fig.suptitle(
    f"Logistic Regression Diagnostic Panel\n"
    f"Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | Spec={spec:.3f} | F1={f1:.3f}",
    y=1.02, fontsize=13
)
plt.tight_layout()
plt.show()


# ## Logistic Regression — Model Performance Summary
# 
# The diagnostic panel above breaks down the performance of our Logistic Regression classifier on the Breast Cancer Wisconsin (Original) dataset:
# 
# - **Overall Accuracy:** 95.9% — The model correctly classified almost 96% of all cases.
# - **Precision (92.1%)** — When the model predicts malignant, it’s correct about 92% of the time.
# - **Recall / Sensitivity (96.7%)** — It identifies nearly all malignant cases, missing only 2 out of 60.
# - **Specificity (95.5%)** — It correctly identifies benign cases 95% of the time.
# - **F1 Score (94.3%)** — Strong balance between precision and recall.
# - **ROC AUC (0.991)** — Excellent ability to separate malignant from benign cases.
# - **PR AUC (0.983)** — Maintains high precision even at high recall levels.
# - **Confusion Matrix:**  
#   - **True Negatives:** 106  
#   - **False Positives:** 5  
#   - **False Negatives:** 2 (critical in cancer detection)  
#   - **True Positives:** 58
# - **Calibration Curve:** Predictions are mostly well-calibrated, meaning predicted probabilities align closely with observed frequencies.
# 
# **Interpretation:**  
# This model delivers strong diagnostic performance with both high sensitivity (important for detecting cancer) and high specificity (avoiding false alarms). The ROC and Precision–Recall curves confirm its robust discrimination, and calibration suggests trustworthy probability estimates. Logistic Regression provides a reliable baseline for this dataset.
# 

# In[ ]:




