import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Import your custom data pipeline class from your data_pipeline file
from data_pipeline import CreditCardDataPipeline


def evaluate_classifier(model, X_test, y_test):
    """Calculates specific evaluation metrics and matrices for credit card fraud."""
    y_pred = model.predict(X_test)

    # Use predict_proba for ROC-AUC to evaluate probability thresholds
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    # Extract performance metrics
    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    # Map out the confusion matrix parameters explicitly
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = cm[0][0], 0, 0, 0

    metrics = {
        "recall": recall,
        "precision": precision,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "report": classification_report(y_test, y_pred, zero_division=0),
    }
    return metrics


def generate_markdown_report(lr_metrics, ada_metrics, report_path):
    """Generates a structured performance comparison markdown artifact without multi-line triple quotes."""
    # Assembling the file contents using a list to prevent IDE string parsing bugs
    lines = [
        "# Credit Card Fraud Detection - Model Performance Report\n\n",
        "This report documents the performance evaluation of the classifiers trained on the processed credit card transaction data.\n\n",
        "## 📊 Summary Performance Comparison\n\n",
        "| Model | Evaluation Strategy | Recall (Catch Rate) | Precision | ROC-AUC | True Positives (TP) | False Negatives (FN) |\n",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n",
        f"| **Logistic Regression** | Balanced SMOTE Train | {lr_metrics['recall']:.4f} | {lr_metrics['precision']:.4f} | {lr_metrics['roc_auc']:.4f} | {lr_metrics['tp']} | {lr_metrics['fn']} |\n",
        f"| **AdaBoost Classifier** | Balanced SMOTE Train | {ada_metrics['recall']:.4f} | {ada_metrics['precision']:.4f} | {ada_metrics['roc_auc']:.4f} | {ada_metrics['tp']} | {ada_metrics['fn']} |\n",
        "\n---\n\n",
        "## 🔍 Detailed Classification Logs\n\n",
        "### 1. Logistic Regression Model\n",
        "```text\n",
        str(lr_metrics["report"]),
        "\n```\n",
        "**Confusion Matrix Grid:**\n",
        f"* True Negatives (Legit caught): {lr_metrics['tn']}\n",
        f"* False Positives (False Alarms): {lr_metrics['fp']}\n",
        f"* False Negatives (Missed Fraud): {lr_metrics['fn']}\n",
        f"* True Positives (Fraud Caught): {lr_metrics['tp']}\n\n",
        "### 2. AdaBoost Classifier Model\n",
        "```text\n",
        str(ada_metrics["report"]),
        "\n```\n",
        "**Confusion Matrix Grid:**\n",
        f"* True Negatives (Legit caught): {ada_metrics['tn']}\n",
        f"* False Positives (False Alarms): {ada_metrics['fp']}\n",
        f"* False Negatives (Missed Fraud): {ada_metrics['fn']}\n",
        f"* True Positives (Fraud Caught): {ada_metrics['tp']}\n\n",
        "---\n",
        "*Report automatically generated on execution of `model.py` pipeline.*\n",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"-> Saved performance report: {report_path}")


# ==========================================
# MAIN EXECUTION ENGINE
# ==========================================
if __name__ == "__main__":
    # Define directory path links
    RAW_DATA_PATH = r"D:\Data Science Projects\Credit-Card-Fraud-Detection-System\data\raw\Credit.xlsx"
    MODEL_OUTPUT_DIR = (
        r"D:\Data Science Projects\Credit-Card-Fraud-Detection-System\models"
    )
    REPORT_OUTPUT_DIR = (
        r"D:\Data Science Projects\Credit-Card-Fraud-Detection-System\reports"
    )

    # Create directories if they don't exist yet
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

    # 1. Initialize and run your custom data pipeline
    print("Orchestrating Preprocessing Pipeline Asset...")
    pipeline = CreditCardDataPipeline(file_path=RAW_DATA_PATH)
    X_train, X_test, y_train, y_test = pipeline.run_pipeline()

    # 2. Train Logistic Regression
    print("\n⚡ Training Logistic Regression Classifier...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_results = evaluate_classifier(lr_model, X_test, y_test)

    # 3. Train AdaBoost
    print("⚡ Training AdaBoost Ensemble Classifier...")
    # FIXED: Removed algorithm="SAMME" to support scikit-learn 1.6+
    ada_model = AdaBoostClassifier(random_state=42)
    ada_model.fit(X_train, y_train)
    ada_results = evaluate_classifier(ada_model, X_test, y_test)

    # 4. Save the Performance Comparison Report
    report_file_path = os.path.join(REPORT_OUTPUT_DIR, "model_results.md")
    generate_markdown_report(lr_results, ada_results, report_file_path)

    # 5. Model Comparison Logic: Save the best model based on ROC-AUC performance
    print("\n💾 Selecting and Exporting Best Model Configuration...")
    if ada_results["roc_auc"] >= lr_results["roc_auc"]:
        best_model = ada_model
        best_name = "AdaBoost Classifier"
    else:
        best_model = lr_model
        best_name = "Logistic Regression"

    model_pickle_path = os.path.join(MODEL_OUTPUT_DIR, "model.pkl")
    joblib.dump(best_model, model_pickle_path)

    print(
        f"🎉 Process Complete! The best-performing model ({best_name}) has been saved to:"
    )
    print(f"➡️ {model_pickle_path}")