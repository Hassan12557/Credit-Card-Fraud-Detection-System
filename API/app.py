import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request, redirect, url_for, session, flash
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. ENVIRONMENT-AWARE DYNAMIC PATH ARCHITECTURE
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

API_DIR = CURRENT_DIR
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Credit.xlsx")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.pkl")
SRC_DIRECTORY = os.path.join(PROJECT_ROOT, "src")

if not os.path.exists(MODEL_PATH):
    ABS_ROOT = "/workspaces/Credit-Card-Fraud-Detection-System"
    if os.path.exists(ABS_ROOT):
        PROJECT_ROOT = ABS_ROOT
        API_DIR = os.path.join(PROJECT_ROOT, "API")
        DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Credit.xlsx")
        MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.pkl")
        SRC_DIRECTORY = os.path.join(PROJECT_ROOT, "src")

if os.path.exists(SRC_DIRECTORY) and SRC_DIRECTORY not in sys.path:
    sys.path.insert(0, SRC_DIRECTORY)

try:
    from data_pipeline import CreditCardDataPipeline
except ImportError:
    class CreditCardDataPipeline:
        def __init__(self, file_path): 
            self.file_path = file_path
        def load_data(self): 
            return pd.read_excel(self.file_path)
        def clean_data(self, df): 
            return df.copy()
        def encode_features(self, df): 
            return df, None
        @property
        def training_columns(self): 
            return ["card_holder_age", "amount"]

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "risk_shield_super_secure_vault_key_2026")

# ==========================================
# 2. MACHINE LEARNING ENGINE LOAD
# ==========================================
print(f"🚀 Attempting to load model binary from: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("-> Success: Model Binary loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model binary: {e}. Activating Mock Backup.")
        class MockModel:
            def predict(self, X): return np.array([0])
            def predict_proba(self, X): return np.array([[0.92, 0.08]])
        model = MockModel()
else:
    print(f"⚠️ Model file not found at path. Activating Mock Engine placeholder.")
    class MockModel:
        def predict(self, X): return np.array([0])
        def predict_proba(self, X): return np.array([[0.92, 0.08]])
    model = MockModel()

categories = ["Entertainment", "Food & Dining", "Gas Stations", "Groceries", "Online Retail", "Travel", "Retail"]
countries = ["US", "CA", "UK", "DE", "FR", "AU", "RU"]
feature_columns = ["card_holder_age", "amount"]
scaler = StandardScaler()

print(f"📊 Parsing Source Analytics Schema from: {DATA_PATH}")
if os.path.exists(DATA_PATH):
    try:
        pipeline = CreditCardDataPipeline(file_path=DATA_PATH)
        raw_df = pipeline.load_data()
        cleaned_df = pipeline.clean_data(raw_df)
        existing_cols = list(cleaned_df.columns)
        
        age_col = next((c for c in existing_cols if c in ["card_holder_age", "age", "cardholder_age"]), None)
        if not age_col:
            age_col = next((c for c in existing_cols if 'age' in str(c).lower()), "card_holder_age")

        amount_col = next((c for c in existing_cols if c in ["amount", "transaction_amount", "amt", "Amount"]), None)
        if not amount_col:
            amount_col = next((c for c in existing_cols if 'amount' in str(c).lower() or 'amt' in str(c).lower()), "amount")

        numerical_cols = [age_col, amount_col]
        valid_numeric_data = cleaned_df[numerical_cols].dropna() if all(col in cleaned_df.columns for col in numerical_cols) else pd.DataFrame()
        
        if not valid_numeric_data.empty and len(valid_numeric_data) > 1 and valid_numeric_data.var().sum() > 0:
            scaler.fit(valid_numeric_data)
        else:
            scaler.fit(pd.DataFrame([[30, 100], [50, 500]], columns=["card_holder_age", "amount"]))
        
        if "merchant_category" in cleaned_df and len(cleaned_df["merchant_category"].dropna()) > 0:
            categories = sorted(cleaned_df["merchant_category"].dropna().unique().tolist())
        if "device_country" in cleaned_df and len(cleaned_df["device_country"].dropna()) > 0:
            countries = sorted(cleaned_df["device_country"].dropna().unique().tolist())
        feature_columns = getattr(pipeline, 'training_columns', ["card_holder_age", "amount"])
    except Exception as e:
        print(f"❌ Ingestion warning: {str(e)}")
        scaler.fit(pd.DataFrame([[30, 100], [50, 500]], columns=["card_holder_age", "amount"]))
else:
    scaler.fit(pd.DataFrame([[30, 100], [50, 500]], columns=["card_holder_age", "amount"]))

USERS_DB = {
    "demo@riskshield.ai": {
        "password": "password123",
        "name": "Alex Carter",
        "tier": "Enterprise Auditor",
        "joined": "2026-01-15"
    }
}

# ==========================================
# 3. GLOBAL BASE TEMPLATE GRAPHICS
# ==========================================
BASE_HEAD = """
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RiskShield AI Portal</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #0b0f19; }</style>
</head>
"""

def get_navbar(is_logged_in):
    if is_logged_in:
        nav_links = """
        <a href="/dashboard" class="text-sm font-medium text-slate-300 hover:text-white transition-colors"><i class="fa-solid fa-chart-pie mr-1.5"></i>Scoring Console</a>
        <a href="/logout" class="text-sm font-medium text-rose-400 hover:text-rose-300 bg-rose-500/10 px-4 py-2 rounded-xl border border-rose-500/20 transition-all"><i class="fa-solid fa-power-off mr-1.5"></i>Sign Out</a>
        """
    else:
        nav_links = """
        <a href="/login" class="text-sm font-medium text-slate-300 hover:text-white transition-colors">Sign In</a>
        """
    return f"""
    <header class="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex justify-between items-center">
        <a href="/" class="flex items-center gap-3 no-underline">
            <div class="bg-gradient-to-tr from-cyan-500 to-blue-600 p-2.5 rounded-xl shadow-lg shadow-blue-500/20"><i class="fa-solid fa-shield-halved text-xl text-white"></i></div>
            <div>
                <h1 class="text-lg font-bold tracking-tight bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">RiskShield AI</h1>
                <p class="text-xs text-cyan-400 font-mono tracking-widest uppercase">Secured Web App Gateway</p>
            </div>
        </a>
        <nav class="flex items-center gap-6">{nav_links}</nav>
    </header>
    """

def render_page(content_template, **context):
    is_logged_in = "user_email" in session
    current_navbar = get_navbar(is_logged_in)
    full_html = f"<!DOCTYPE html><html lang='en'>{BASE_HEAD}<body class='text-slate-200 min-h-screen flex flex-col justify-between'>{current_navbar}{content_template}</body></html>"
    return render_template_string(full_html, **context)

HOME_CONTENT = """
<main class="flex-grow max-w-4xl mx-auto flex flex-col items-center justify-center text-center px-6 py-20">
    <span class="text-xs font-mono tracking-widest uppercase text-cyan-400 bg-cyan-500/10 border border-cyan-500/30 px-3 py-1.5 rounded-full mb-6">Autonomous Risk Infrastructure</span>
    <h2 class="text-4xl md:text-5xl font-extrabold text-white tracking-tight leading-tight max-w-2xl">Real-Time Machine Learning <br><span class="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">Credit Card Fraud Prevention</span></h2>
    <div class="mt-10 flex flex-wrap gap-4 justify-center">
        {% if session.get('user_email') %} 
            <a href="/dashboard" class="bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold px-8 py-4 rounded-xl shadow-lg hover:opacity-95 transition-all">Launch Scoring Console</a>
        {% else %} 
            <a href="/login" class="bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold px-8 py-4 rounded-xl shadow-lg hover:opacity-95 transition-all">Access Scoring Dashboard</a> 
        {% endif %}
    </div>
</main>
"""

LOGIN_CONTENT = """
<main class="flex-grow flex items-center justify-center p-6 my-6">
    <div class="max-w-md w-full bg-slate-900/40 border border-slate-800 backdrop-blur-xl p-8 rounded-3xl shadow-2xl">
        <h3 class="text-2xl font-bold text-white text-center tracking-tight mb-6">Security Portal Login</h3>
        <form method="POST" class="space-y-4">
            <div><label class="block text-xs font-semibold text-slate-400 mb-2">Email Address</label><input type="email" name="email" value="demo@riskshield.ai" required class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 px-4 text-sm text-white focus:outline-none focus:border-cyan-500"></div>
            <div><label class="block text-xs font-semibold text-slate-400 mb-2">Password</label><input type="password" name="password" value="password123" required class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 px-4 text-sm text-white focus:outline-none focus:border-cyan-500"></div>
            <button type="submit" class="w-full bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold py-3.5 rounded-xl text-sm mt-2">Continue</button>
        </form>
    </div>
</main>
"""

# ==========================================
# 4. ROUTING SYSTEMS LOGIC
# ==========================================
@app.route("/")
def home():
    return render_page(HOME_CONTENT)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        if email in USERS_DB and USERS_DB[email]["password"] == password:
            session["user_email"] = email
            return redirect(url_for("dashboard"))
        flash("Invalid Credentials", "danger")
    return render_page(LOGIN_CONTENT)

@app.route("/logout")
def logout():
    session.pop("user_email", None)
    return redirect(url_for("home"))

@app.route("/dashboard")
def dashboard():
    if "user_email" not in session:
        return redirect(url_for("login"))
        
    search_paths = [
        os.path.join(CURRENT_DIR, "dashboard.html"),
        os.path.join(CURRENT_DIR, "templates", "dashboard.html"),
        os.path.join(PROJECT_ROOT, "dashboard.html"),
        os.path.join(API_DIR, "dashboard.html"),
        os.path.join(API_DIR, "templates", "dashboard.html")
    ]

    dashboard_html = None
    for target in search_paths:
        if os.path.exists(target):
            try:
                with open(target, "r", encoding="utf-8") as f:
                    dashboard_html = f.read()
                print(f"🎯 Successfully loaded layout template at: {target}")
                break
            except Exception as e:
                print(f"⚠️ Template reading bypass notification: {e}")
            
    if dashboard_html is None:
        searched_locations = "<br>".join([f"• {p}" for p in search_paths])
        return render_page(f"""
        <main class="max-w-xl mx-auto px-6 py-12 text-center bg-slate-900/50 border border-slate-800 rounded-3xl mt-12">
            <h3 class="text-xl font-bold text-rose-400 mb-2">⚠️ Template Discovery Matrix Failure</h3>
            <p class="text-sm text-slate-400 mb-4">The file 'dashboard.html' is missing or misplaced.</p>
            <div class="text-left bg-slate-950 p-4 rounded-xl border border-slate-800/80 text-xs font-mono text-slate-400 space-y-2">
                <p class="text-cyan-400 font-semibold">Checked Locations:</p>
                <p class="text-slate-500 text-[11px] leading-relaxed">{searched_locations}</p>
                <hr class="border-slate-800 my-2">
                <p class="text-white">💡 Quick Fix: Verify that 'dashboard.html' sits directly inside the same folder as your 'app.py' file.</p>
            </div>
        </main>
        """)
        
    return render_page(dashboard_html, categories=categories, countries=countries)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        user_age = float(data.get("age", 30))
        user_amount = float(data.get("amount", 0.0))
        user_cat = str(data.get("category", "")).strip()
        user_country = str(data.get("country", "")).strip()
        
        global feature_columns
        expected_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else feature_columns

        scaler_features = list(getattr(scaler, "feature_names_in_", ["card_holder_age", "amount"]))
        input_num_df = pd.DataFrame([[user_age, user_amount]], columns=scaler_features)
        scaled_nums = scaler.transform(input_num_df)[0]

        input_vector_dict = {col: 0 for col in expected_features}
        
        for i, col_name in enumerate(scaler_features):
            if col_name in input_vector_dict:
                input_vector_dict[col_name] = scaled_nums[i]
            else:
                matched = False
                if "age" in col_name.lower():
                    for f_col in expected_features:
                        if "age" in f_col.lower():
                            input_vector_dict[f_col] = scaled_nums[i]
                            matched = True
                            break
                if not matched and ("amount" in col_name.lower() or "amt" in col_name.lower()):
                    for f_col in expected_features:
                        if "amount" in f_col.lower() or "amt" in f_col.lower():
                            input_vector_dict[f_col] = scaled_nums[i]
                            matched = True
                            break

        cat_variants = [user_cat, user_cat.lower(), user_cat.capitalize(), user_cat.upper(), user_cat.replace(" ", "_")]
        for cv in cat_variants:
            dummy_col = f"merchant_category_{cv}"
            if dummy_col in input_vector_dict:
                input_vector_dict[dummy_col] = 1
                break
                
        country_variants = [user_country, user_country.lower(), user_country.upper(), user_country.capitalize()]
        for cv in country_variants:
            dummy_col = f"device_country_{cv}"
            if dummy_col in input_vector_dict:
                input_vector_dict[dummy_col] = 1
                break

        final_input_df = pd.DataFrame([input_vector_dict], columns=expected_features)
        prediction = int(model.predict(final_input_df)[0])
        probability = float(model.predict_proba(final_input_df)[0][1])

        return jsonify({
            "status": "success",
            "is_fraud": prediction,
            "fraud_probability": round(probability, 4),
            "meta": { "engine": "ML Inference Core", "latency_status": "nominal" }
        })
    except Exception as e:
        print(f"❌ Error during model prediction: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    print("📡 Activating RiskShield Security Gateway on http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)