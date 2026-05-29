import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CreditCardDataPipeline:

    def __init__(self, file_path, test_size=0.2, random_state=42):
        """Initializes the data pipeline with parameters."""
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state

        # Pipeline components to be saved after fitting
        self.scaler = StandardScaler()
        self.smote = None  # Will be dynamically set based on sample sizes
        self.training_columns = None  # To preserve feature alignment

    def load_data(self):
        """Step 1 & 2: Loads data from Excel and prints raw shapes."""
        print("=== Step 1: Ingesting Raw Data ===")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Target file not found at: {self.file_path}"
            )

        df = pd.read_excel(self.file_path, sheet_name="Raw Transaction Data")
        print(f"-> Ingested {df.shape[0]} rows and {df.shape[1]} features.")
        return df

    def clean_data(self, df):
        """Step 3: Resolves duplicates, structural string formats, anomalies, and null values."""
        print("\n=== Step 2: Running Data Cleaning Engine ===")

        # A. Deduplication
        initial_rows = df.shape[0]
        df = df.drop_duplicates(subset=["transaction_id"], keep="first")
        print(f"-> Removed {initial_rows - df.shape[0]} duplicate rows.")

        # B. Clean 'amount' field (Remove string symbols and fill Nulls)
        if df["amount"].dtype == "O":
            df["amount"] = (
                df["amount"]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
            )
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["amount"] = df["amount"].fillna(df["amount"].median())

        # C. Clean 'card_holder_age' field (Fix logical negative outliers and fill Nulls)
        df.loc[df["card_holder_age"] <= 0, "card_holder_age"] = np.nan
        df["card_holder_age"] = df["card_holder_age"].fillna(
            df["card_holder_age"].median()
        )

        # D. Standardize Categorical formatting casing
        df["merchant_category"] = (
            df["merchant_category"].astype(str).str.strip().str.capitalize()
        )

        # E. Categorical Imputation (Mode substitution)
        mode_country = df["device_country"].mode()[0]
        df["device_country"] = df["device_country"].fillna(mode_country)

        print("-> Cleaned structural and logical defects successfully.")
        return df

    def encode_features(self, df):
        """Step 4: Isolates target and encodes categorical values via dummy flags."""
        print("\n=== Step 3: Transforming Categorical Matrices ===")

        # Isolate target variable
        target = df["is_fraud"]

        # Drop operational keys / identification strings
        df_features = df.drop(
            columns=["transaction_id", "timestamp", "card_number", "is_fraud"]
        )

        # One-Hot Encoding
        df_encoded = pd.get_dummies(
            df_features,
            columns=["merchant_category", "device_country"],
            drop_first=True,
        )

        # Store explicit structural layout for future parsing/inference
        self.training_columns = df_encoded.columns.tolist()
        print(f"-> Generated {df_encoded.shape[1]} columns after encoding.")
        return df_encoded, target

    def split_and_scale(self, X, y):
        """Step 5 & 6: Data splitting and isolated training normalization to avert feature leakage."""
        print("\n=== Step 4: Splitting and Normalizing Matrices ===")

        # Stratified Splitting
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Standard Scaling numeric indexes explicitly
        numerical_cols = ["card_holder_age", "amount"]

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[numerical_cols] = self.scaler.fit_transform(
            X_train[numerical_cols]
        )
        X_test_scaled[numerical_cols] = self.scaler.transform(
            X_test[numerical_cols]
        )

        print("-> Normalized numeric variants safely without feature leakage.")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def balance_training_set(self, X_train, y_train):
        """Step 7: Dynamically configures and runs SMOTE minority balancing."""
        print("\n=== Step 5: Synthesizing Imbalanced Classes via SMOTE ===")

        minority_count = np.bincount(y_train)[1]
        print(f"-> Detected minority (Fraud) class size in training: {minority_count}")

        # Set safe neighbor value adaptively to avoid ValueError limitations
        safe_k = min(3, minority_count - 1) if minority_count <= 5 else 5

        print(f"-> Initializing SMOTE Engine with k_neighbors={safe_k}...")
        self.smote = SMOTE(k_neighbors=safe_k, random_state=self.random_state)

        X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
        print(f"-> Balanced Class Array distribution: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled

    def run_pipeline(self):
        """Executes the sequential end-to-end data processing workflow."""
        print("⚡ Orchestrating Credit Card Processing Pipeline ⚡")

        raw_df = self.load_data()
        cleaned_df = self.clean_data(raw_df)
        X, y = self.encode_features(cleaned_df)

        X_train, X_test, y_train, y_test = self.split_and_scale(X, y)
        X_train_bal, y_train_bal = self.balance_training_set(X_train, y_train)

        print("\n🎉 Pipeline Execution Completed Successfully!")
        return X_train_bal, X_test, y_train_bal, y_test


# ==========================================
# LOCAL PIPELINE EXECUTION ENGINE
# ==========================================
if __name__ == "__main__":
    # Define absolute directory pointing to your local assets
    DATA_PATH = r"D:\Data Science Projects\Credit-Card-Fraud-Detection-System\data\raw\Credit.xlsx"

    # Inception instance of the framework object
    pipeline = CreditCardDataPipeline(file_path=DATA_PATH)

    # Trigger complete transform loop
    X_train_ready, X_test_ready, y_train_ready, y_test_ready = (
        pipeline.run_pipeline()
    )

    # View final dimension arrays to verify accuracy before forwarding to models
    print("\n--- Final Matrix Verification ---")
    print(f"Final Processed Training Set Features Shape: {X_train_ready.shape}")
    print(f"Final Processed Training Set Labels Shape:   {y_train_ready.shape}")
    print(f"Final Unaltered Testing Set Features Shape:  {X_test_ready.shape}")