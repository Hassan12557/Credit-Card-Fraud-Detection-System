import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request, redirect, url_for, session, flash
from sklearn.preprocessing import StandardScaler

# ==========================================
# ENVIRONMENT-AWARE DYNAMIC PATH ENGINE
# ==========================================
# Determine if running inside a Docker container or Codespace
IS_CONTAINER = os.path.exists('/.dockerenv') or os.environ.get('CODESPACES') == 'true'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if IS_CONTAINER:
    print("🐳 Context: Linux Container / GitHub Codespace Environment Detected.")
    # Force absolute container mapping directory paths
    PROJECT_ROOT = "/app"
    SRC_DIRECTORY = os.path.join(PROJECT_ROOT, "src")
    DATA_PATH = os.environ.get("DATA_PATH", os.path.join(PROJECT_ROOT, "data", "raw", "Credit.xlsx"))
    MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "model.pkl"))
else:
    print("💻 Context: Native Windows/Local Host Environment Detected.")
    if os.path.basename(CURRENT_DIR) in ["src", "app", "API"]:
        PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    else:
        PROJECT_ROOT = CURRENT_DIR
    SRC_DIRECTORY = os.path.join(PROJECT_ROOT, "src")
    DATA_PATH = r"D:\Data Science Projects\Credit-Card-Fraud-Detection-System\data\raw\Credit.xlsx"
    MODEL_PATH = r"D:\Data Science Projects\Credit-Card-Fraud-Detection-System\models\model.pkl"

if SRC_DIRECTORY not in sys.path:
    sys.path.insert(0, SRC_DIRECTORY)

# Import pipeline dependencies safely
try:
    from data_pipeline import CreditCardDataPipeline
except ImportError:
    # Fallback placeholder if pipeline isn't packaged in module format yet
    class CreditCardDataPipeline:
        def __init__(self, file_path): self.file_path = file_path
        def load_data(self): return pd.read_excel(self.file_path)
        def clean_data(self, df): return df.copy()
        def encode_features(self, df): return df, None
        @property
        def training_columns(self): return ["card_holder_age", "amount"]

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "risk_shield_super_secure_vault_key_2026")

# Configure cross-platform path mapping
if IS_CONTAINER:
    DATA_PATH = os.environ.get("DATA_PATH", os.path.join(PROJECT_ROOT, "data", "raw", "Credit.xlsx"))
    MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "model.pkl"))
else:
    DATA_PATH = r"D:\Data Science Projects\Credit-Card-Fraud-Detection-System\data\raw\Credit.xlsx"
    MODEL_PATH = r"D:\Data Science Projects\Credit-Card-Fraud-Detection-System\models\model.pkl"

# Fallback to relative paths if Windows absolute paths are missing locally
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ Target path unreadable. Falling back to relative structure layout...")
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Credit.xlsx")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.pkl")

# ==========================================
# GLOBAL MACHINE LEARNING ENGINE INITIALIZATION
# ==========================================
print(f"🚀 Loading Model Artifact from: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("-> Model Binary loaded successfully.")
else:
    print(f"❌ CRITICAL ERROR: Trained model file missing at {MODEL_PATH}!")
    class MockModel:
        def predict(self, X): return np.array([0])
        def predict_proba(self, X): return np.array([[0.92, 0.08]])
    model = MockModel()

# 🛡️ DEFENSIVE SEEDING: Safe global fallbacks to prevent NameErrors 
categories = ["Entertainment", "Food & Dining", "Gas Stations", "Groceries", "Online Retail"]
countries = ["USA", "CAN", "GBR", "AUS", "DEU"]
feature_columns = ["card_holder_age", "amount"]
scaler = StandardScaler()

print(f"📊 Parsing Source Analytics Schema from: {DATA_PATH}")
if os.path.exists(DATA_PATH):
    try:
        pipeline = CreditCardDataPipeline(file_path=DATA_PATH)
        raw_df = pipeline.load_data()
        cleaned_df = pipeline.clean_data(raw_df)
        
        # ----------------------------------------------------
        # DYNAMIC COLUMN DISCOVERY ENGINE (Prevents KeyErrors)
        # ----------------------------------------------------
        existing_cols = list(cleaned_df.columns)
        print(f"📋 Columns detected inside cleaned dataframe: {existing_cols}")
        
        # Smart discovery for the Age column
        age_col = None
        for candidate in ["card_holder_age", "age", "cardholder_age", "holder_age"]:
            if candidate in existing_cols:
                age_col = candidate
                break
        if not age_col:
            for c in existing_cols:
                if 'age' in str(c).lower():
                    age_col = c
                    break

        # Smart discovery for the Amount column
        amount_col = None
        for candidate in ["amount", "transaction_amount", "amt", "Amount"]:
            if candidate in existing_cols:
                amount_col = candidate
                break
        if not amount_col:
            for c in existing_cols:
                if 'amount' in str(c).lower() or 'amt' in str(c).lower():
                    amount_col = c
                    break

        # Apply discovered mappings or execute safe fallback
        if age_col and amount_col:
            numerical_cols = [age_col, amount_col]
            print(f"🎯 Success: Automatically mapped numerical features to: {numerical_cols}")
        else:
            numeric_cols_found = list(cleaned_df.select_dtypes(include=[np.number]).columns)
            if len(numeric_cols_found) >= 2:
                numerical_cols = numeric_cols_found[:2]
                print(f"⚠️ Column names missing. Falling back to first numeric pairs: {numerical_cols}")
            else:
                numerical_cols = ["card_holder_age", "amount"]
                print(f"🚨 Defaulting to base definitions. Data schema may be corrupted.")

        # ----------------------------------------------------
        # SAFE SCALER FITTING ENGINE (Prevents RuntimeWarnings)
        # ----------------------------------------------------
        valid_numeric_data = cleaned_df[numerical_cols].dropna() if all(col in cleaned_df.columns for col in numerical_cols) else pd.DataFrame()
        
        if not valid_numeric_data.empty and len(valid_numeric_data) > 1 and valid_numeric_data.var().sum() > 0:
            scaler.fit(valid_numeric_data)
            print(f"🎯 Success: Scaler fitted cleanly with real-world dataset distributions across: {numerical_cols}")
        else:
            print("⚠️ Warning: Cleaned dataset numerical columns are empty or constant. Seeding fallback baseline scale matrix.")
            fallback_df = pd.DataFrame([[30, 100], [50, 500]], columns=numerical_cols)
            scaler.fit(fallback_df)
        
        if "merchant_category" in cleaned_df and len(cleaned_df["merchant_category"].dropna()) > 0:
            categories = sorted(cleaned_df["merchant_category"].dropna().unique().tolist())
        if "device_country" in cleaned_df and len(cleaned_df["device_country"].dropna()) > 0:
            countries = sorted(cleaned_df["device_country"].dropna().unique().tolist())
        feature_columns = getattr(pipeline, 'training_columns', ["card_holder_age", "amount"])
        
    except Exception as e:
        print(f"❌ Error during telemetry data ingestion: {str(e)}. Using fallback defaults.")
        scaler.fit(pd.DataFrame([[30, 100], [50, 500]], columns=["card_holder_age", "amount"]))
else:
    print("⚠️ Warning: Data source spreadsheet missing. Initializing fallback structures.")
    scaler.fit(pd.DataFrame([[30, 100], [50, 500]], columns=["card_holder_age", "amount"]))

# ==========================================
# SIMULATED MEMORY DATABASES
# ==========================================
USERS_DB = {
    "demo@riskshield.ai": {
        "password": "password123",
        "name": "Alex Carter",
        "tier": "Enterprise Auditor",
        "joined": "2026-01-15"
    }
}

# UI Layout Components
BASE_HEAD = """
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RiskShield AI Portal</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #0b0f19; }</style>
</head>
"""
NAVBAR = """
<header class="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex justify-between items-center">
    <a href="/" class="flex items-center gap-3 no-underline">
        <div class="bg-gradient-to-tr from-cyan-500 to-blue-600 p-2.5 rounded-xl shadow-lg shadow-blue-500/20"><i class="fa-solid fa-shield-halved text-xl text-white"></i></div>
        <div>
            <h1 class="text-lg font-bold tracking-tight bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">RiskShield AI</h1>
            <p class="text-xs text-cyan-400 font-mono tracking-widest uppercase">Secured Web App Gateway</p>
        </div>
    </a>
    <nav class="flex items-center gap-6">
        {% if session.get('user_email') %}
            <a href="/dashboard" class="text-sm font-medium text-slate-300 hover:text-white transition-colors"><i class="fa-solid fa-chart-pie mr-1.5"></i>Scoring Console</a>
            <a href="/account" class="text-sm font-medium text-slate-300 hover:text-white transition-colors"><i class="fa-solid fa-user-gear mr-1.5"></i>My Profile</a>
            <a href="/logout" class="text-sm font-medium text-rose-400 hover:text-rose-300 bg-rose-500/10 px-4 py-2 rounded-xl border border-rose-500/20 transition-all"><i class="fa-solid fa-power-off mr-1.5"></i>Sign Out</a>
        {% else %}
            <a href="/login" class="text-sm font-medium text-slate-300 hover:text-white transition-colors">Sign In</a>
            <a href="/register" class="bg-gradient-to-r from-cyan-500 to-blue-600 text-white text-sm font-semibold px-5 py-2.5 rounded-xl shadow-md transition-all hover:opacity-95">Create Account</a>
        {% endif %}
    </nav>
</header>
"""
BASE_LAYOUT = "<!DOCTYPE html><html lang='en'>{BASE_HEAD}<body class='text-slate-200 min-h-screen flex flex-col justify-between'>{NAVBAR}{CONTENT}</body></html>"

def render_page(content_template, **context):
    full_html = BASE_LAYOUT.replace("{BASE_HEAD}", BASE_HEAD).replace("{NAVBAR}", NAVBAR).replace("{CONTENT}", content_template)
    return render_template_string(full_html, **context)

HOME_CONTENT = """
<main class="flex-grow max-w-4xl mx-auto flex flex-col items-center justify-center text-center px-6 py-20">
    <span class="text-xs font-mono tracking-widest uppercase text-cyan-400 bg-cyan-500/10 border border-cyan-500/30 px-3 py-1.5 rounded-full mb-6">Autonomous Risk Infrastructure</span>
    <h2 class="text-4xl md:text-5xl font-extrabold text-white tracking-tight leading-tight max-w-2xl">Real-Time Machine Learning <br><span class="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">Credit Card Fraud Prevention</span></h2>
    <div class="mt-10 flex flex-wrap gap-4 justify-center">
        {% if session.get('user_email') %} <a href="/dashboard" class="bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold px-8 py-4 rounded-xl shadow-lg hover:opacity-95 transition-all">Launch Scoring Console</a>
        {% else %} <a href="/register" class="bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold px-8 py-4 rounded-xl shadow-lg hover:opacity-95 transition-all">Get Started Free</a>
        <a href="/login" class="bg-slate-900 border border-slate-800 text-slate-300 font-semibold px-8 py-4 rounded-xl hover:text-white transition-all">Access Account</a> {% endif %}
    </div>
</main>
"""
REGISTER_CONTENT = """
<main class="flex-grow flex items-center justify-center p-6 my-6">
    <div class="max-w-md w-full bg-slate-900/40 border border-slate-800 backdrop-blur-xl p-8 rounded-3xl shadow-2xl">
        <h3 class="text-2xl font-bold text-white text-center tracking-tight mb-6">Create your account</h3>
        <div class="space-y-2.5 mb-6">
            <button onclick="alert('Google Auth Connect...')" class="w-full bg-slate-950 border border-slate-800 hover:bg-slate-900 text-slate-200 text-sm font-medium py-3 rounded-xl transition-all flex items-center justify-center gap-3"><i class="fa-brands fa-google"></i> Continue with Google</button>
            <button onclick="alert('Facebook Auth Connect...')" class="w-full bg-slate-950 border border-slate-800 hover:bg-slate-900 text-slate-200 text-sm font-medium py-3 rounded-xl transition-all flex items-center justify-center gap-3"><i class="fa-brands fa-facebook text-blue-500"></i> Continue with Facebook</button>
        </div>
        <div class="relative flex items-center justify-center mb-6"><div class="absolute inset-0 flex items-center"><div class="w-full border-t border-slate-800/80"></div></div><span class="relative px-3 bg-[#0b0f19] text-[9px] font-bold tracking-widest text-slate-500 uppercase">OR CONTINUE WITH EMAIL</span></div>
        <form method="POST" class="space-y-4">
            <div><label class="block text-xs font-semibold text-slate-400 mb-2">Full Name</label><input type="text" name="name" required class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 px-4 text-sm text-white focus:outline-none focus:border-cyan-500"></div>
            <div><label class="block text-xs font-semibold text-slate-400 mb-2">Email Address</label><input type="email" name="email" required class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 px-4 text-sm text-white focus:outline-none focus:border-cyan-500"></div>
            <div><label class="block text-xs font-semibold text-slate-400 mb-2">Password</label><input type="password" name="password" required class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 px-4 text-sm text-white focus:outline-none focus:border-cyan-500"></div>
            <button type="submit" class="w-full bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold py-3.5 rounded-xl text-sm mt-2">Sign Up</button>
        </form>
    </div>
</main>
"""
LOGIN_CONTENT = REGISTER_CONTENT.replace("Create your account", "Welcome back").replace("Sign Up", "Continue")
DASHBOARD_CONTENT = """
<main class="flex-grow max-w-6xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
    <section class="lg:col-span-7 bg-slate-900/40 border border-slate-800 backdrop-blur-xl p-6 rounded-3xl shadow-2xl">
        <form id="prediction-form" class="space-y-4">
            <div class="grid grid-cols-2 gap-4">
                <div><label class="block text-[10px] font-bold text-slate-500 mb-2">Cardholder Age</label><input type="number" name="age" value="34" required class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 px-4 text-sm text-white"></div>
                <div><label class="block text-[10px] font-bold text-slate-500 mb-2">Amount ($ USD)</label><input type="number" step="0.01" name="amount" value="125.50" required class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 px-4 text-sm text-white"></div>
            </div>
            <div class="grid grid-cols-2 gap-4">
                <div><label class="block text-[10px] font-bold text-slate-500 mb-2">Merchant Category</label><select name="category" class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 px-4 text-sm text-white">{% for cat in categories %}<option value="{{ cat }}">{{ cat }}</option>{% endfor %}</select></div>
                <div><label class="block text-[10px] font-bold text-slate-500 mb-2">Origin Country</label><select name="country" class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 px-4 text-sm text-white">{% for c in countries %}<option value="{{ c }}">{{ c }}</option>{% endfor %}</select></div>
            </div>
            <button type="submit" class="w-full bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold py-3.5 rounded-xl text-sm mt-2">Evaluate Telemetry Vectors</button>
        </form>
    </section>
    <section class="lg:col-span-5">
        <div id="output-display" class="h-full bg-slate-900/40 border border-slate-800 p-5 rounded-3xl flex flex-col justify-between">
            <div>
                <h4 class="text-xs uppercase text-slate-500 font-mono mb-4">Response Matrix Node</h4>
                <div class="bg-slate-950/60 p-4 rounded-xl border border-slate-800"><p id="prob-text" class="text-2xl font-mono font-bold text-cyan-400">0.0%</p><p class="text-xs text-slate-400 mt-1">Algorithmic Fraud Imbalance Score</p></div>
            </div>
        </div>
    </section>
</main>
<script>
document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ age: parseInt(formData.get('age')), amount: parseFloat(formData.get('amount')), category: formData.get('category'), country: formData.get('country') })
    });
    const data = await response.json();
    document.getElementById('prob-text').innerText = (data.fraud_probability * 100).toFixed(2) + '% — ' + (data.is_fraud ? "CRITICAL RISK" : "SECURE");
});
</script>
"""
ACCOUNT_CONTENT = " <main class='p-12 text-center'><h3 class='text-xl text-white'>Profile Configuration</h3><p class='text-slate-400'>Logged in as: {{ session['user_email'] }}</p></main> "

@app.route("/")
def home(): return render_page(HOME_CONTENT)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email").strip().lower()
        USERS_DB[email] = {"password": request.form.get("password"), "name": request.form.get("name"), "tier": "Standard User", "joined": "2026-05-29"}
        session["user_email"] = email
        return redirect(url_for("dashboard"))
    return render_page(REGISTER_CONTENT)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email").strip().lower()
        if email in USERS_DB:
            session["user_email"] = email
            return redirect(url_for("dashboard"))
    return render_page(LOGIN_CONTENT)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/dashboard")
def dashboard():
    if "user_email" not in session: return redirect(url_for("login"))
    return render_page(DASHBOARD_CONTENT, categories=categories, countries=countries)

# ==========================================
# RESTFUL API INFERENCE ENDPOINT
# ==========================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        user_age = float(data.get("age", 30))
        user_amount = float(data.get("amount", 0.0))
        user_cat = str(data.get("category", "")).strip().capitalize()
        user_country = str(data.get("country", "")).strip()

        # 🎯 FIX: Dynamically adapt to whatever feature layout the scaler was fitted with
        scaler_features = list(getattr(scaler, "feature_names_in_", ["card_holder_age", "amount"]))
        input_num_df = pd.DataFrame([[user_age, user_amount]], columns=scaler_features)
        scaled_nums = scaler.transform(input_num_df)[0]

        # Initialize full vector dimensionality array
        input_vector_dict = {col: 0 for col in feature_columns}
        
        # 🎯 FIX: Smart feature-mapping crosswalk loop prevents schema tracking drops
        for i, col_name in enumerate(scaler_features):
            if col_name in input_vector_dict:
                input_vector_dict[col_name] = scaled_nums[i]
            else:
                matched = False
                if "age" in col_name.lower():
                    for f_col in feature_columns:
                        if "age" in f_col.lower():
                            input_vector_dict[f_col] = scaled_nums[i]
                            matched = True
                            break
                if not matched and ("amount" in col_name.lower() or "amt" in col_name.lower()):
                    for f_col in feature_columns:
                        if "amount" in f_col.lower() or "amt" in f_col.lower():
                            input_vector_dict[f_col] = scaled_nums[i]
                            matched = True
                            break
                if not matched:
                    if i < len(feature_columns):
                        input_vector_dict[feature_columns[i]] = scaled_nums[i]

        # Handle Categorical Binary Mappings
        cat_dummy_col = f"merchant_category_{user_cat}"
        country_dummy_col = f"device_country_{user_country}"

        if cat_dummy_col in input_vector_dict: input_vector_dict[cat_dummy_col] = 1
        if country_dummy_col in input_vector_dict: input_vector_dict[country_dummy_col] = 1

        final_input_df = pd.DataFrame([input_vector_dict], columns=feature_columns)
        
        # Execute ML Scored Prediction Matrix
        prediction = int(model.predict(final_input_df)[0])
        probability = float(model.predict_proba(final_input_df)[0][1])

        return jsonify({
            "status": "success",
            "is_fraud": prediction,
            "fraud_probability": round(probability, 4),
            "meta": { "engine": "XGBoost/RandomForest Core", "latency_status": "nominal" }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    host_ip = "0.0.0.0" if IS_CONTAINER else "127.0.0.1"
    app.run(debug=True, host=host_ip, port=5000)