import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD THE DATASET
# ==========================================
print("=== Step 1: Loading Raw Data ===")
# Using 'r' before the string handles Windows backslashes perfectly
file_path = r"D:\Data Science Projects\Credit-Card-Fraud-Detection-System\data\raw\Credit.xlsx"
# Read only the raw transaction sheet
df = pd.read_excel(file_path, sheet_name="Raw Transaction Data")
print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.\n")
# ==========================================
# 2. IDENTIFY MISSING / NULL VALUES
# ==========================================
print("=== Step 2: Checking for Missing/Null Values ===")
null_counts = df.isnull().sum()
print("Null values per column before cleaning:")
print(null_counts[null_counts > 0])
print("\n")

# ==========================================
# 3. DATA CLEANING & PREPROCESSING
# ==========================================
print("=== Step 3: Resolving Data Quality Issues ===")

# A. Remove Exact Structural Duplicates (Fixes TXN10006 duplication)
initial_rows = df.shape[0]
df = df.drop_duplicates(subset=["transaction_id"], keep="first")
print(f"-> Removed {initial_rows - df.shape[0]} duplicate transaction row(s).")

# B. Clean 'amount' Column (Fixes string/currency character issue like "$22.50")
if df["amount"].dtype == "O":  # If detected as an Object/String type
    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

# Impute missing amount values with the median amount
median_amount = df["amount"].median()
df["amount"] = df["amount"].fillna(median_amount)

# C. Handle Logical Outliers & Missing Values in 'card_holder_age'
# Treat negative ages (e.g., -5) as invalid (NaN)
df.loc[df["card_holder_age"] <= 0, "card_holder_age"] = np.nan

# Impute missing/invalid ages using the median age
median_age = df["card_holder_age"].median()
df["card_holder_age"] = df["card_holder_age"].fillna(median_age)

# D. Standardize Categorical Text Formatting (Fixes "retail", "Retail", "RETAIL")
df["merchant_category"] = (
    df["merchant_category"].astype(str).str.strip().str.capitalize()
)

# E. Impute Missing Values in Categorical Data ('device_country')
# Fill missing countries with the most frequent value (Mode)
mode_country = df["device_country"].mode()[0]
df["device_country"] = df["device_country"].fillna(mode_country)

print("Data cleaning complete. Verified null counts:")
print(df[["card_holder_age", "amount", "device_country"]].isnull().sum())
print("\n")

# ==========================================
# 4. ENCODE CATEGORICAL DATA
# ==========================================
print("=== Step 4: Encoding Categorical Features ===")
# Drop unique/high-cardinality identifiers that won't help model generalization
df_features = df.drop(
    columns=["transaction_id", "timestamp", "card_number", "is_fraud"]
)
target = df["is_fraud"]

# Apply One-Hot Encoding to categorical variables (merchant_category, device_country)
# drop_first=True prevents multicollinearity (the dummy variable trap)
df_encoded = pd.get_dummies(
    df_features, columns=["merchant_category", "device_country"], drop_first=True
)
print(f"Features after One-Hot Encoding: {df_encoded.shape[1]} columns.")
print(df_encoded.columns.tolist(), "\n")

# ==========================================
# 5. SPLIT THE DATASET (TRAIN / TEST)
# ==========================================
print("=== Step 5: Splitting Dataset (Train & Test) ===")
# FIXED: Changed test_type to test_size
X_train, X_test, y_train, y_test = train_test_split(
    df_encoded, target, test_size=0.2, random_state=42, stratify=target
)
print(f"Training Features Shape: {X_train.shape} | Labels: {y_train.shape}")
print(f"Testing Features Shape:  {X_test.shape}  | Labels: {y_test.shape}")
print(f"Fraud distribution in Training Set: {np.bincount(y_train)}")
print(f"Fraud distribution in Testing Set:  {np.bincount(y_test)}\n")

# ==========================================
# 6. FEATURE SCALING
# ==========================================
print("=== Step 6: Feature Scaling ===")
scaler = StandardScaler()

# Fit only on the training set and transform both to avoid data leakage
numerical_cols = ["card_holder_age", "amount"]

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
print("Numerical features successfully normalized using StandardScaler.\n")

# ==========================================
# 7. BALANCE THE DATASET (SMOTE)
# ==========================================
# ==========================================
# 7. BALANCE THE DATASET (SMOTE)
# ==========================================
print("=== Step 7: Balancing Training Dataset via SMOTE ===")
# FIXED: Adjusted k_neighbors to accommodate the 5 minority class samples
smote = SMOTE(k_neighbors=3, random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled, y_train
)

print("Dataset successfully balanced!")
print(f"Resampled Training Features Shape: {X_train_resampled.shape} | Labels: {y_train_resampled.shape}")
print(f"Balanced Class Counts: {np.bincount(y_train_resampled)}")
print("\nPreprocessing Pipeline Complete! Data is ready for Model Training.")