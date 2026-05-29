import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request, redirect, url_for, session, flash
from sklearn.preprocessing import StandardScaler

# ==========================================
# DYNAMIC PATH RESOLUTION BREAKOUT
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_directory = os.path.join(project_root, "src")

if src_directory not in sys.path:
    sys.path.insert(0, src_directory)

from data_pipeline import CreditCardDataPipeline

app = Flask(__name__)
app.secret_key = "risk_shield_super_secure_vault_key_2026"  # Required for managing secure session cookies

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

# ==========================================
# GLOBAL MACHINE LEARNING ENGINE INITIALIZATION
# ==========================================
DATA_PATH = r"D:\Data Science Projects\Credit-Card-Fraud-Detection-System\data\raw\Credit.xlsx"
MODEL_PATH = r"D:\Data Science Projects\Credit-Card-Fraud-Detection-System\models\model.pkl"

print("🚀 Initializing Production GUI Inference Engine with Authentication Layers...")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("-> Model Binary loaded successfully.")
else:
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Please run model.py first!")

pipeline = CreditCardDataPipeline(file_path=DATA_PATH)
raw_df = pipeline.load_data()
cleaned_df = pipeline.clean_data(raw_df)
X_train_raw, _ = pipeline.encode_features(cleaned_df)

scaler = StandardScaler()
numerical_cols = ["card_holder_age", "amount"]
scaler.fit(cleaned_df[numerical_cols])

categories = sorted(cleaned_df["merchant_category"].dropna().unique().tolist())
countries = sorted(cleaned_df["device_country"].dropna().unique().tolist())
feature_columns = pipeline.training_columns

# ==========================================
# TEMPLATES LAYOUT MATRIX (PLAIN PYTHON STRINGS)
# ==========================================
BASE_HEAD = """
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RiskShield AI Portal</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #0b0f19; }
    </style>
</head>
"""

NAVBAR = """
<header class="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex justify-between items-center">
    <a href="/" class="flex items-center gap-3 no-underline">
        <div class="bg-gradient-to-tr from-cyan-500 to-blue-600 p-2.5 rounded-xl shadow-lg shadow-blue-500/20">
            <i class="fa-solid fa-shield-halved text-xl text-white"></i>
        </div>
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

BASE_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
{BASE_HEAD}
<body class="text-slate-200 min-h-screen flex flex-col justify-between">
    {NAVBAR}
    {CONTENT}
</body>
</html>
"""

def render_page(content_template, **context):
    """Safely merges HTML modules without f-string collisions and compiles via Jinja2."""
    full_html = BASE_LAYOUT.replace("{BASE_HEAD}", BASE_HEAD).replace("{NAVBAR}", NAVBAR).replace("{CONTENT}", content_template)
    return render_template_string(full_html, **context)

# ==========================================
# SEGMENT VIEW TEMPLATES (PLAIN PYTHON STRINGS)
# ==========================================
HOME_CONTENT = """
<main class="flex-grow max-w-4xl mx-auto flex flex-col items-center justify-center text-center px-6 py-20">
    <span class="text-xs font-mono tracking-widest uppercase text-cyan-400 bg-cyan-500/10 border border-cyan-500/30 px-3 py-1.5 rounded-full mb-6">Autonomous Risk Infrastructure</span>
    <h2 class="text-4xl md:text-5xl font-extrabold text-white tracking-tight leading-tight max-w-2xl">
        Real-Time Machine Learning <br><span class="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">Credit Card Fraud Prevention</span>
    </h2>
    <p class="text-slate-400 mt-6 max-w-xl text-base md:text-lg leading-relaxed">
        Empower your transaction flows with enterprise-grade predictive models. Analyze merchant profiles, spatial anomalies, and user velocity structures in micro-seconds.
    </p>
    <div class="mt-10 flex flex-wrap gap-4 justify-center">
        {% if session.get('user_email') %}
            <a href="/dashboard" class="bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold px-8 py-4 rounded-xl shadow-lg shadow-blue-500/20 hover:opacity-95 transition-all">Launch Scoring Console</a>
        {% else %}
            <a href="/register" class="bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold px-8 py-4 rounded-xl shadow-lg shadow-blue-500/20 hover:opacity-95 transition-all">Get Started Free</a>
            <a href="/login" class="bg-slate-900 border border-slate-800 text-slate-300 font-semibold px-8 py-4 rounded-xl hover:text-white transition-all">Access Account</a>
        {% endif %}
    </div>
</main>
<footer class="border-t border-slate-900 bg-slate-950/60 px-6 py-4 text-center text-xs text-slate-500">&copy; 2026 RiskShield AI Inc. All rights reserved.</footer>
"""

REGISTER_CONTENT = """
<main class="flex-grow flex items-center justify-center p-6 my-6">
    <div class="max-w-md w-full bg-slate-900/40 border border-slate-800 backdrop-blur-xl p-8 rounded-3xl shadow-2xl">
        <div class="mb-6 text-center">
            <h3 class="text-2xl font-bold text-white tracking-tight">Create your account</h3>
            <p class="text-xs text-slate-400 mt-1">Join RiskShield to begin evaluating transactions.</p>
        </div>
        
        <div class="space-y-2.5 mb-6">
            <button onclick="alert('OAuth Pipeline: Redirecting to Google Cloud Accounts Verification Node...')" class="w-full bg-slate-950 border border-slate-800 hover:bg-slate-900 text-slate-200 text-sm font-medium py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-3 cursor-pointer">
                <i class="fa-brands fa-google text-base text-white"></i> Continue with Google
            </button>
            <button onclick="alert('OAuth Pipeline: Redirecting to Facebook Identity Graph Node...')" class="w-full bg-slate-950 border border-slate-800 hover:bg-slate-900 text-slate-200 text-sm font-medium py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-3 cursor-pointer">
                <i class="fa-brands fa-facebook text-base text-blue-500"></i> Continue with Facebook
            </button>
        </div>

        <div class="relative flex items-center justify-center mb-6">
            <div class="absolute inset-0 flex items-center"><div class="w-full border-t border-slate-800/80"></div></div>
            <span class="relative px-3 bg-[#0b0f19] text-[9px] font-bold tracking-widest text-slate-500 uppercase">OR CONTINUE WITH EMAIL</span>
        </div>

        {% with messages = get_flashed_messages(category_filter=["error"]) %}
          {% if messages %}<p class="bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs p-3 rounded-xl mb-4 font-medium"><i class="fa-solid fa-circle-exclamation mr-1.5"></i>{{ messages[0] }}</p>{% endif %}
        {% endwith %}
        
        <form method="POST" class="space-y-4">
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Full Name</label>
                <input type="text" name="name" placeholder="John Doe" required class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 px-4 text-sm text-white focus:outline-none focus:border-cyan-500 transition-all">
            </div>
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Email Address</label>
                <input type="email" name="email" placeholder="name@company.com" required class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 px-4 text-sm text-white focus:outline-none focus:border-cyan-500 transition-all">
            </div>
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Password</label>
                <input type="password" name="password" placeholder="••••••••" required class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 px-4 text-sm text-white focus:outline-none focus:border-cyan-500 transition-all">
            </div>
            <button type="submit" class="w-full bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold py-3.5 rounded-xl shadow-lg shadow-blue-500/20 hover:opacity-95 transition-all mt-2 cursor-pointer text-sm">Sign Up</button>
        </form>
        <div class="text-center mt-6 text-xs text-slate-500">Already possess an identity? <a href="/login" class="text-cyan-400 hover:underline">Sign In</a></div>
    </div>
</main>
"""

LOGIN_CONTENT = """
<main class="flex-grow flex items-center justify-center p-6 my-6">
    <div class="max-w-md w-full bg-slate-900/40 border border-slate-800 backdrop-blur-xl p-8 rounded-3xl shadow-2xl">
        <div class="mb-6 text-center">
            <h3 class="text-2xl font-bold text-white tracking-tight">Welcome back</h3>
            <p class="text-xs text-slate-400 mt-1">Sign in to your account context to query operational models.</p>
        </div>
        
        <div class="space-y-2.5 mb-6">
            <button onclick="alert('OAuth Pipeline: Connecting to Google Gateway...')" class="w-full bg-slate-950 border border-slate-800 hover:bg-slate-900 text-slate-200 text-sm font-medium py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-3 cursor-pointer">
                <i class="fa-brands fa-google text-base text-white"></i> Continue with Google
            </button>
            <button onclick="alert('OAuth Pipeline: Connecting to Facebook Gateway...')" class="w-full bg-slate-950 border border-slate-800 hover:bg-slate-900 text-slate-200 text-sm font-medium py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-3 cursor-pointer">
                <i class="fa-brands fa-facebook text-base text-blue-500"></i> Continue with Facebook
            </button>
        </div>

        <div class="relative flex items-center justify-center mb-6">
            <div class="absolute inset-0 flex items-center"><div class="w-full border-t border-slate-800/80"></div></div>
            <span class="relative px-3 bg-[#0b0f19] text-[9px] font-bold tracking-widest text-slate-500 uppercase">OR CONTINUE WITH EMAIL</span>
        </div>

        {% with messages = get_flashed_messages(category_filter=["error"]) %}
          {% if messages %}<p class="bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs p-3 rounded-xl mb-4 font-medium"><i class="fa-solid fa-circle-exclamation mr-1.5"></i>{{ messages[0] }}</p>{% endif %}
        {% endwith %}
        
        <form method="POST" class="space-y-4">
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Email Address</label>
                <input type="email" name="email" placeholder="name@company.com" required class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 px-4 text-sm text-white focus:outline-none focus:border-cyan-500 transition-all">
            </div>
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Password</label>
                <input type="password" name="password" placeholder="••••••••" required class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 px-4 text-sm text-white focus:outline-none focus:border-cyan-500 transition-all">
            </div>
            <button type="submit" class="w-full bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold py-3.5 rounded-xl shadow-lg shadow-blue-500/20 hover:opacity-95 transition-all mt-2 cursor-pointer text-sm">Continue</button>
        </form>
        <div class="text-center mt-6 text-xs text-slate-500">Lacking an operational token? <a href="/register" class="text-cyan-400 hover:underline">Create Account</a></div>
    </div>
</main>
"""

ACCOUNT_CONTENT = """
<main class="flex-grow max-w-2xl w-full mx-auto p-6 md:p-12">
    <div class="bg-slate-900/40 border border-slate-800 backdrop-blur-xl p-8 rounded-3xl shadow-2xl">
        <div class="flex items-center gap-4 border-b border-slate-800 pb-6 mb-6">
            <div class="w-16 h-16 rounded-2xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center text-white text-2xl font-bold">
                {{ user_info['name'][0] }}
            </div>
            <div>
                <h3 class="text-xl font-bold text-white">{{ user_info['name'] }}</h3>
                <p class="text-xs text-cyan-400 font-mono tracking-wider">{{ user_info['tier'] }}</p>
            </div>
        </div>
        <div class="space-y-4">
            <div class="flex justify-between py-2 border-b border-slate-800/40">
                <span class="text-xs uppercase font-bold text-slate-500">System Identifier (Email)</span>
                <span class="text-sm font-mono text-slate-300">{{ session['user_email'] }}</span>
            </div>
            <div class="flex justify-between py-2 border-b border-slate-800/40">
                <span class="text-xs uppercase font-bold text-slate-500">Node Activation Date</span>
                <span class="text-sm font-mono text-slate-300">{{ user_info['joined'] }}</span>
            </div>
            <div class="flex justify-between py-2">
                <span class="text-xs uppercase font-bold text-slate-500">API Access State</span>
                <span class="text-xs bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-md font-mono font-bold">AUTHORIZED</span>
            </div>
        </div>
    </div>
</main>
"""

DASHBOARD_CONTENT = """
{% with messages = get_flashed_messages(category_filter=["success"]) %}
  {% if messages %}
  <div class="max-w-6xl w-full mx-auto px-6 mt-4">
      <div class="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs p-4 rounded-xl flex items-center gap-2">
          <i class="fa-solid fa-square-check text-base"></i>
          <span>{{ messages[0] }}</span>
      </div>
  </div>
  {% endif %}
{% endwith %}

<main class="flex-grow max-w-6xl w-full mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
    
    <section class="lg:col-span-7 bg-slate-900/40 border border-slate-800 backdrop-blur-xl p-6 rounded-3xl shadow-2xl">
        <div class="mb-6">
            <h2 class="text-lg font-bold text-white flex items-center gap-2">
                <i class="fa-solid fa-file-invoice-dollar text-cyan-400"></i> Analytics Workspace
            </h2>
            <p class="text-xs text-slate-400 mt-1">Log vector parameters below to query the operational model artifact.</p>
        </div>

        <form id="prediction-form" class="space-y-4">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label class="block text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-2">Cardholder Age</label>
                    <div class="relative">
                        <i class="fa-solid fa-user absolute left-4 top-3.5 text-slate-600 text-xs"></i>
                        <input type="number" name="age" min="18" max="110" placeholder="e.g. 34" required class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 pl-11 pr-4 text-sm text-white focus:outline-none focus:border-cyan-500 transition-all">
                    </div>
                </div>
                <div>
                    <label class="block text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-2">Amount ($ USD)</label>
                    <div class="relative">
                        <i class="fa-solid fa-dollar-sign absolute left-4 top-3.5 text-slate-600 text-xs"></i>
                        <input type="number" step="0.01" name="amount" min="0.01" placeholder="0.00" required class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 pl-11 pr-4 text-sm text-white focus:outline-none focus:border-cyan-500 transition-all">
                    </div>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label class="block text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-2">Merchant Category Mapping</label>
                    <div class="relative">
                        <i class="fa-solid fa-store absolute left-4 top-3.5 text-slate-600 text-xs"></i>
                        <select name="category" required class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 pl-11 pr-4 text-sm text-white appearance-none focus:outline-none focus:border-cyan-500 transition-all">
                            {% for cat in categories %}
                            <option value="{{ cat }}">{{ cat }}</option>
                            {% endfor %}
                        </select>
                        <i class="fa-solid fa-chevron-down absolute right-4 top-4 text-slate-600 pointer-events-none text-[10px]"></i>
                    </div>
                </div>
                <div>
                    <label class="block text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-2">Target Origin Country</label>
                    <div class="relative">
                        <i class="fa-solid fa-globe absolute left-4 top-3.5 text-slate-600 text-xs"></i>
                        <select name="country" required class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 pl-11 pr-4 text-sm text-white appearance-none focus:outline-none focus:border-cyan-500 transition-all">
                            {% for country in countries %}
                            <option value="{{ country }}">{{ country }}</option>
                            {% endfor %}
                        </select>
                        <i class="fa-solid fa-chevron-down absolute right-4 top-4 text-slate-600 pointer-events-none text-[10px]"></i>
                    </div>
                </div>
            </div>

            <button type="submit" class="w-full bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold py-3.5 rounded-xl shadow-lg shadow-blue-500/20 hover:opacity-90 active:scale-[0.99] transition-all flex items-center justify-center gap-2 cursor-pointer mt-2 text-sm">
                <i class="fa-solid fa-bolt"></i> Evaluate Telemetry Vectors
            </button>
        </form>
    </section>

    <section class="lg:col-span-5 flex flex-col justify-between">
        <div id="output-placeholder" class="h-full bg-slate-900/20 border border-dashed border-slate-800 p-8 rounded-3xl flex flex-col items-center justify-center text-center">
            <div class="w-12 h-12 rounded-xl bg-slate-900 flex items-center justify-center text-slate-600 mb-3 border border-slate-800">
                <i class="fa-solid fa-satellite-dish text-xl"></i>
            </div>
            <h3 class="text-sm font-semibold text-slate-400">Awaiting Data Feed Pipeline</h3>
            <p class="text-[11px] text-slate-500 mt-1 max-w-xs">Fill parameters and fire calculation to evaluate predictive outcomes.</p>
        </div>

        <div id="output-loader" class="hidden h-full bg-slate-900/40 border border-slate-800 backdrop-blur-xl p-8 rounded-3xl flex flex-col items-center justify-center text-center">
            <div class="w-10 h-10 rounded-full border-4 border-slate-800 border-t-cyan-500 animate-spin mb-4"></div>
            <h3 class="text-xs font-mono text-cyan-400 uppercase tracking-widest animate-pulse">Running Neural Classifiers...</h3>
        </div>

        <div id="output-display" class="hidden h-full bg-slate-900/40 border border-slate-800 backdrop-blur-xl p-5 rounded-3xl flex flex-col justify-between shadow-2xl transition-all duration-300">
            <div>
                <div class="flex justify-between items-center mb-4">
                    <span class="text-[10px] font-mono uppercase tracking-wider text-slate-500">Evaluation Response Node</span>
                    <span id="badge" class="px-2.5 py-0.5 rounded-full text-[10px] font-bold tracking-wide"></span>
                </div>

                <div id="verdict-card" class="p-4 rounded-xl border mb-4 flex items-start gap-3">
                    <div id="verdict-icon" class="p-2.5 rounded-lg text-white flex items-center justify-center text-sm"></div>
                    <div>
                        <h4 id="verdict-title" class="font-bold text-sm"></h4>
                        <p id="verdict-desc" class="text-[11px] text-slate-400 mt-0.5 leading-relaxed"></p>
                    </div>
                </div>

                <div class="bg-slate-950/60 border border-slate-800/80 p-4 rounded-xl space-y-3">
                    <div>
                        <div class="flex justify-between text-[11px] font-semibold text-slate-400 mb-1">
                            <span>Algorithmic Fraud Imbalance Score</span>
                            <span id="prob-text" class="font-mono font-bold text-white">0%</span>
                        </div>
                        <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                            <div id="prob-bar" class="h-full transition-all duration-500 ease-out" style="width: 0%"></div>
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-4 pt-2 border-t border-slate-800/60 text-center">
                        <div>
                            <span class="block text-[9px] uppercase font-bold text-slate-500 tracking-wider">Verdict</span>
                            <span id="stat-verdict" class="text-xs font-bold mt-0.5 block"></span>
                        </div>
                        <div>
                            <span class="block text-[9px] uppercase font-bold text-slate-500 tracking-wider">Telemetry Class</span>
                            <span id="stat-tier" class="text-xs font-bold mt-0.5 block"></span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="text-[9px] text-center font-mono text-slate-600 border-t border-slate-800/40 pt-3 mt-3">SHIELD CORE SYSTEM ENTRY // LATENCY MODULE CLEAR</div>
        </div>
    </section>
</main>
<footer class="border-t border-slate-900 bg-slate-950 px-6 py-4 text-center text-xs text-slate-500">&copy; 2026 RiskShield AI. Built with Flask, Jinja2 & Tailwind CSS templates.</footer>

<script>
    document.getElementById('prediction-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        const placeholder = document.getElementById('output-placeholder');
        const loader = document.getElementById('output-loader');
        const display = document.getElementById('output-display');
        
        placeholder.classList.add('hidden');
        display.classList.add('hidden');
        loader.classList.remove('hidden');

        const formData = new FormData(this);
        const payload = {
            age: parseInt(formData.get('age')),
            amount: parseFloat(formData.get('amount')),
            category: formData.get('category'),
            country: formData.get('country')
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            
            loader.classList.add('hidden');
            display.classList.remove('hidden');

            const probPercentage = (data.fraud_probability * 100).toFixed(1);
            document.getElementById('prob-text').innerText = probPercentage + '%';
            
            const probBar = document.getElementById('prob-bar');
            probBar.style.width = probPercentage + '%';

            const badge = document.getElementById('badge');
            const vCard = document.getElementById('verdict-card');
            const vIcon = document.getElementById('verdict-icon');
            const vTitle = document.getElementById('verdict-title');
            const vDesc = document.getElementById('verdict-desc');
            const sVerdict = document.getElementById('stat-verdict');
            const sTier = document.getElementById('stat-tier');

            if (data.is_fraud === 1) {
                badge.className = "px-2.5 py-0.5 rounded-full text-[10px] font-bold tracking-wide bg-rose-500/20 text-rose-400 border border-rose-500/30";
                badge.innerText = "CRITICAL ALERT";
                vCard.className = "p-4 rounded-xl border bg-rose-950/20 border-rose-500/20 flex items-start gap-3";
                vIcon.className = "p-2 rounded-lg bg-rose-500 text-white flex items-center justify-center text-xs shadow-md";
                vIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
                vTitle.className = "font-bold text-sm text-rose-400";
                vTitle.innerText = "Fraud Signatures Found";
                vDesc.innerText = "Anomalous heuristics detected. Automated rules suggest processing locks immediately.";
                probBar.className = "h-full bg-gradient-to-r from-orange-500 to-rose-500 transition-all duration-500 ease-out";
                sVerdict.className = "text-xs font-bold mt-0.5 block text-rose-400";
                sVerdict.innerText = "FRAUDULENT";
                sTier.className = "text-xs font-bold mt-0.5 block text-rose-500";
                sTier.innerText = "CRITICAL RISK";
            } else {
                badge.className = "px-2.5 py-0.5 rounded-full text-[10px] font-bold tracking-wide bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
                badge.innerText = "VERIFIED SECURE";
                vCard.className = "p-4 rounded-xl border bg-emerald-950/20 border-emerald-500/20 flex items-start gap-3";
                vIcon.className = "p-2 rounded-lg bg-emerald-500 text-white flex items-center justify-center text-xs shadow-md";
                vIcon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
                vTitle.className = "font-bold text-sm text-emerald-400";
                vTitle.innerText = "Clear Transaction Baseline";
                vDesc.innerText = "Vector values align perfectly with legitimate account user profiles.";
                probBar.className = "h-full bg-gradient-to-r from-blue-500 to-emerald-500 transition-all duration-500 ease-out";
                sVerdict.className = "text-xs font-bold mt-0.5 block text-emerald-400";
                sVerdict.innerText = "LEGITIMATE";
                sTier.className = "text-xs font-bold mt-0.5 block text-emerald-500";
                sTier.innerText = "MINIMAL";
            }
        } catch (err) {
            alert("Core Inference Failure.");
            loader.classList.add('hidden');
            placeholder.classList.remove('hidden');
        }
    });
</script>
"""

# ==========================================
# ROUTE GATEWAY: VISITOR CORE VIEWS
# ==========================================
@app.route("/")
def home():
    """Serves professional enterprise landing splash page."""
    return render_page(HOME_CONTENT)

# ==========================================
# ROUTE GATEWAY: AUTHENTICATION SUB-SYSTEM
# ==========================================
@app.route("/register", methods=["GET", "POST"])
def register():
    """Handles secure new account creation and triggers high-fidelity simulated transactional welcome email."""
    if request.method == "POST":
        full_name = request.form.get("name")
        email = request.form.get("email").strip().lower()
        password = request.form.get("password")

        if email in USERS_DB:
            flash("Account identity already registered. Choose another email or log in.", "error")
            return redirect(url_for("register"))

        USERS_DB[email] = {
            "password": password,
            "name": full_name,
            "tier": "Standard Sandbox User",
            "joined": "2026-05-29"
        }

        # ==========================================================
        # HIGH-FIDELITY TRANSACTIONAL EMAIL SERVICE EMULATION
        # ==========================================================
        print("\n" + "═"*80)
        print(" ✉️  [OUTBOUND SMTP TRANSACT RELAY] -> DISPATCHED VIA SENDGRID/AWS-SES NODES")
        print("═"*80)
        print(f" FROM: RiskShield Security Core <no-reply@riskshield.ai>")
        print(f" TO: {full_name} <{email}>")
        print(f" SUBJECT: Welcome to RiskShield AI — Verify Your Analytics Cluster Matrix")
        print("─"*80)
        print(f" Hi {full_name},")
        print("\n Thank you for choosing RiskShield AI for your transaction telemetry protection.")
        print(" Your account workspace configuration parameters have built successfully.")
        print("\n 🔗 CLICK TO CONFIRM WORKSPACE SECURITY CONSOLE ACCREDITATION:")
        print(f"    https://riskshield.ai/auth/verify-token?cluster=usr_93c202a8_production")
        print("\n 🔍 PROFILE DISPATCH LOGS:")
        print(f"   ├─ Associated Token Group: {email}")
        print("   ├─ Encryption Protocol: AES-GCM-256")
        print("   └─ Default Security Layer: ML Inference Endpoint Enabled")
        print("\n If you did not initiate this activation request, please lock your api credentials.")
        print("\n Stay secure,\n The RiskShield Core Engineering Team")
        print("═"*80 + "\n")

        flash("Account created successfully! Premium transactional onboarding email sent to your inbox.", "success")
        session["user_email"] = email
        return redirect(url_for("dashboard"))

    return render_page(REGISTER_CONTENT)

@app.route("/login", methods=["GET", "POST"])
def login():
    """Validates submitted session parameters against the virtual database store."""
    if request.method == "POST":
        email = request.form.get("email").strip().lower()
        password = request.form.get("password")

        if email in USERS_DB and USERS_DB[email]["password"] == password:
            session["user_email"] = email
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials provided. Please check fields and retry.", "error")
            return redirect(url_for("login"))

    return render_page(LOGIN_CONTENT)

@app.route("/logout")
def logout():
    """Clears session state arrays safely."""
    session.clear()
    return redirect(url_for("home"))

# ==========================================
# ROUTE GATEWAY: PROTECTED CORE SYSTEM ENCLAVES
# ==========================================
@app.route("/account")
def account():
    """Displays secure profile boundaries for verified operations sessions."""
    if "user_email" not in session:
        return redirect(url_for("login"))
    
    user_info = USERS_DB[session["user_email"]]
    return render_page(ACCOUNT_CONTENT, user_info=user_info)

@app.route("/dashboard")
def dashboard():
    """Serves the interactive live analysis interface console."""
    if "user_email" not in session:
        return redirect(url_for("login"))
    
    user_info = USERS_DB[session["user_email"]]
    return render_page(DASHBOARD_CONTENT, user_info=user_info, categories=categories, countries=countries)

@app.route("/predict", methods=["POST"])
def predict():
    """Accepts dynamic inference calls securely from authenticated dashboards."""
    if "user_email" not in session:
        return jsonify({"error": "Unauthorized context access"}), 401
    try:
        data = request.get_json()
        user_age = float(data["age"])
        user_amount = float(data["amount"])
        user_cat = str(data["category"]).strip().capitalize()
        user_country = str(data["country"]).strip()

        input_num_df = pd.DataFrame([[user_age, user_amount]], columns=numerical_cols)
        scaled_nums = scaler.transform(input_num_df)[0]

        input_vector_dict = {col: 0 for col in feature_columns}
        input_vector_dict["card_holder_age"] = scaled_nums[0]
        input_vector_dict["amount"] = scaled_nums[1]

        cat_dummy_col = f"merchant_category_{user_cat}"
        country_dummy_col = f"device_country_{user_country}"

        if cat_dummy_col in input_vector_dict:
            input_vector_dict[cat_dummy_col] = 1
        if country_dummy_col in input_vector_dict:
            input_vector_dict[country_dummy_col] = 1

        final_input_df = pd.DataFrame([input_vector_dict], columns=feature_columns)
        prediction = int(model.predict(final_input_df)[0])
        probability = float(model.predict_proba(final_input_df)[0][1])

        return jsonify({"is_fraud": prediction, "fraud_probability": probability})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n⚡ Production Enterprise App Engine Online! Launch Gateway:")
    print("➡️ http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)