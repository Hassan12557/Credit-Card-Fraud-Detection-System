# Credit Card Fraud Detection - Model Performance Report

This report documents the performance evaluation of the classifiers trained on the processed credit card transaction data.

## 📊 Summary Performance Comparison

| Model | Evaluation Strategy | Recall (Catch Rate) | Precision | ROC-AUC | True Positives (TP) | False Negatives (FN) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Balanced SMOTE Train | 0.0000 | 0.0000 | 0.4444 | 0 | 1 |
| **AdaBoost Classifier** | Balanced SMOTE Train | 0.0000 | 0.0000 | 0.2222 | 0 | 1 |

---

## 🔍 Detailed Classification Logs

### 1. Logistic Regression Model
```text
              precision    recall  f1-score   support

           0       0.90      1.00      0.95         9
           1       0.00      0.00      0.00         1

    accuracy                           0.90        10
   macro avg       0.45      0.50      0.47        10
weighted avg       0.81      0.90      0.85        10

```
**Confusion Matrix Grid:**
* True Negatives (Legit caught): 9
* False Positives (False Alarms): 0
* False Negatives (Missed Fraud): 1
* True Positives (Fraud Caught): 0

### 2. AdaBoost Classifier Model
```text
              precision    recall  f1-score   support

           0       0.90      1.00      0.95         9
           1       0.00      0.00      0.00         1

    accuracy                           0.90        10
   macro avg       0.45      0.50      0.47        10
weighted avg       0.81      0.90      0.85        10

```
**Confusion Matrix Grid:**
* True Negatives (Legit caught): 9
* False Positives (False Alarms): 0
* False Negatives (Missed Fraud): 1
* True Positives (Fraud Caught): 0

---
*Report automatically generated on execution of `model.py` pipeline.*
