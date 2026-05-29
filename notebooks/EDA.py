import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set a clean, professional aesthetic for our charts
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"  # Safe cross-platform clean font


def load_clean_eda_data(file_path):
    """Loads raw data and runs basic structural parsing for visualization metrics."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")

    df = pd.read_excel(file_path, sheet_name="Raw Transaction Data")

    # Quick structural cleaning for accurate plots (fixing amount formatting)
    if df["amount"].dtype == "O":
        df["amount"] = (
            df["amount"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Fill invalid ages temporarily for age distribution analysis
    df.loc[df["card_holder_age"] <= 0, "card_holder_age"] = np.nan

    return df


def plot_class_imbalance(df, output_dir):
    """Visualizes the severe target class imbalance."""
    plt.figure(figsize=(6, 5))

    # Calculate percentages for chart presentation labels
    counts = df["is_fraud"].value_counts()
    pcts = df["is_fraud"].value_counts(normalize=True) * 100

    ax = sns.countplot(
        x="is_fraud",
        data=df,
        palette=["#2E4053", "#C0392B"],
        hue="is_fraud",
        legend=False,
    )

    # Annotate bars with counts and percentages
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2.0,
            height + 0.5,
            f"{counts[i]} ({pcts[i]:.1f}%)",
            ha="center",
            weight="bold",
        )

    plt.title("Target Label Distribution (Extreme Class Imbalance)", fontsize=12, pad=15)
    plt.xticks([0, 1], ["Legitimate (0)", "Fraudulent (1)"])
    plt.xlabel("Transaction Status")
    plt.ylabel("Record Count")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "01_class_imbalance.png"), dpi=300)
    print("-> Saved: 01_class_imbalance.png")
    plt.close()


def plot_amount_distribution(df, output_dir):
    """Visualizes transaction financial amounts across target values using log-scaling."""
    plt.figure(figsize=(10, 5))

    # Drop missing values to prevent plotting sequence warnings
    clean_amt_df = df.dropna(subset=["amount"])

    # KDE overlay density distribution plot
    sns.histplot(
        data=clean_amt_df,
        x="amount",
        hue="is_fraud",
        element="step",
        stat="density",
        common_norm=False,
        kde=True,
        palette=["#2E4053", "#C0392B"],
        log_scale=True,  # Crucial for fraud data due to extreme outlier spend gaps
    )

    plt.title(
        "Transaction Value Density Profiling (Log Scale)", fontsize=12, pad=15
    )
    plt.xlabel("Transaction Volume Value ($ Log Scale)")
    plt.ylabel("Density Distribution")
    plt.legend(["Fraudulent", "Legitimate"], frameon=True)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "02_amount_distribution.png"), dpi=300)
    print("-> Saved: 02_amount_distribution.png")
    plt.close()


def plot_merchant_fraud_vectors(df, output_dir):
    """Tracks which category segments reveal elevated fraud vulnerabilities."""
    plt.figure(figsize=(10, 6))

    # Standardize names for unified aggregation indicators
    df["merchant_category"] = (
        df["merchant_category"].astype(str).str.strip().str.capitalize()
    )

    # Group records by category and evaluate fraud rates
    fraud_rates = (
        df.groupby("merchant_category")["is_fraud"]
        .mean()
        .reset_index()
        .sort_values(by="is_fraud", ascending=False)
    )

    sns.barplot(
        x="is_fraud",
        y="merchant_category",
        data=fraud_rates,
        palette="Reds_r",
        hue="merchant_category",
        legend=False,
    )

    plt.title("Fraud Occurrence Probability across Merchant Sectors", fontsize=12, pad=15)
    plt.xlabel("Fraud Ratio (Percentage Mean)")
    plt.ylabel("Merchant Category Classification")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "03_merchant_fraud_vectors.png"), dpi=300)
    print("-> Saved: 03_merchant_fraud_vectors.png")
    plt.close()


def plot_age_vs_amount_scatter(df, output_dir):
    """Bivariate feature mapping analyzing the intersection of Age vs. Transaction Amount."""
    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="card_holder_age",
        y="amount",
        hue="is_fraud",
        style="is_fraud",
        palette=["#2C3E50", "#E74C3C"],
        size="amount",
        sizes=(40, 400),
        alpha=0.85,
    )

    plt.title("Bivariate Behavioral Space: Transaction Value vs. Cardholder Age", fontsize=12, pad=15)
    plt.xlabel("Cardholder Age Profile")
    plt.ylabel("Transaction Charge Value ($)")
    plt.legend(["Legitimate", "Fraudulent"], loc="upper right")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "04_age_vs_amount_scatter.png"), dpi=300)
    print("-> Saved: 04_age_vs_amount_scatter.png")
    plt.close()


# ==========================================
# MAIN EXECUTION ORCHESTRATION ENGINE
# ==========================================
if __name__ == "__main__":
    # 1. Define asset structural references
    RAW_DATA_PATH = r"D:\Data Science Projects\Credit-Card-Fraud-Detection-System\data\raw\Credit.xlsx"
    REPORT_DIR = r"D:\Data Science Projects\Credit-Card-Fraud-Detection-System\reports\figures"

    # Ensure output subdirectory paths are compiled safely
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("📊 Loading Visual Analysis Profiler Subsystem...")
    raw_dataframe = load_clean_eda_data(RAW_DATA_PATH)

    print("\n📈 Executing Visualization Pipelines...")
    plot_class_imbalance(raw_dataframe, REPORT_DIR)
    plot_amount_distribution(raw_dataframe, REPORT_DIR)
    plot_merchant_fraud_vectors(raw_dataframe, REPORT_DIR)
    plot_age_vs_amount_scatter(raw_dataframe, REPORT_DIR)

    print(f"\n🎉 EDA Visualization Completed! Charts stored in:\n➡️ {REPORT_DIR}")