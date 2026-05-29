import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from sklearn.preprocessing import StandardScaler

# ==========================================
# DYNAMIC PATH RESOLUTION BREAKOUT
# ==========================================
# Find the absolute path of app.py (inside D:\...\API)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root directory
project_root = os.path.dirname(current_dir)
# Map out the explicit path to the src folder
src_directory = os.path.join(project_root, "src")

# Insert the src folder into Python's module search path list
if src_directory not in sys.path:
    sys.path.insert(0, src_directory)

# Now Python can safely discover data_pipeline.py inside the src folder!
from data_pipeline import CreditCardDataPipeline

app = Flask(__name__)

# ==========================================
# GLOBAL SYSTEM INITIALIZATION
# ==========================================
DATA_PATH = r"D:\Data Science Projects\Credit-Card-Fraud-Detection-System\data\raw\Credit.xlsx"
MODEL_PATH = r"D:\Data Science Projects\Credit-Card-Fraud-Detection-System\models\model.pkl"

print("🚀 Initializing Production GUI Inference Engine...")

# ... Rest of the app.py code remains exactly the same as provided previously ...
# Load the trained model artifact
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("-> Model Binary loaded successfully.")
else:
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. Please run model.py first!"
    )

# Re-initialize a pipeline instance to extract training column alignment and scaling parameters
pipeline = CreditCardDataPipeline(file_path=DATA_PATH)
raw_df = pipeline.load_data()
cleaned_df = pipeline.clean_data(raw_df)
X_train_raw, _ = pipeline.encode_features(cleaned_df)

# Fit internal scaler to replicate exactly the training baseline distribution
scaler = StandardScaler()
numerical_cols = ["card_holder_age", "amount"]
scaler.fit(cleaned_df[numerical_cols])

# Dynamic categories list to populate dropdown menus in the UI
categories = sorted(
    cleaned_df["merchant_category"].dropna().unique().tolist()
)
countries = sorted(cleaned_df["device_country"].dropna().unique().tolist())
feature_columns = pipeline.training_columns

# ==========================================
# CORE JAVASCRIPT/TAILWIND REACT-LIKE UI TEMPLATE
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit Card Fraud Detection</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #0b0f19; }
    </style>
</head>
<body class="text-slate-200 min-h-screen flex flex-col justify-between">

    <header class="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex justify-between items-center">
        <div class="flex items-center gap-3">
            <div class="bg-gradient-to-tr from-cyan-500 to-blue-600 p-2.5 rounded-xl shadow-lg shadow-blue-500/20 animate-pulse">
                <i class="fa-solid fa-shield-halved text-xl text-white"></i>
            </div>
            <div>
                <h1 class="text-lg font-bold tracking-tight bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">RiskShield AI</h1>
                <p class="text-xs text-cyan-400 font-mono tracking-widest uppercase">Real-Time Scoring Engine</p>
            </div>
        </div>
        <div class="flex items-center gap-4 text-xs font-mono bg-slate-950/60 border border-slate-800 px-4 py-2 rounded-full">
            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
            <span class="text-slate-400">Core API Status:</span>
            <span class="text-emerald-400 font-bold">OPERATIONAL</span>
        </div>
    </header>

    <main class="flex-grow max-w-6xl w-full mx-auto p-4 md:p-8 grid grid-cols-1 lg:grid-cols-12 gap-8">

        <section class="lg:col-span-7 bg-slate-900/40 border border-slate-800 backdrop-blur-xl p-6 rounded-3xl shadow-2xl">
            <div class="mb-6">
                <h2 class="text-xl font-bold text-white flex items-center gap-2">
                    <i class="fa-solid fa-file-invoice-dollar text-cyan-400"></i> Transaction Parameters
                </h2>
                <p class="text-sm text-slate-400 mt-1">Provide live transaction details below to calculate structural risk telemetry vectors.</p>
            </div>

            <form id="prediction-form" class="space-y-6">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Cardholder Age Profile</label>
                        <div class="relative">
                            <i class="fa-solid fa-user absolute left-4 top-3.5 text-slate-500"></i>
                            <input type="number" name="age" min="18" max="110" placeholder="e.g. 35" required
                                class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 pl-11 pr-4 text-white focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all placeholder:text-slate-600">
                        </div>
                    </div>

                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Transaction Value ($ USD)</label>
                        <div class="relative">
                            <i class="fa-solid fa-dollar-sign absolute left-4 top-3.5 text-slate-500"></i>
                            <input type="number" step="0.01" name="amount" min="0.01" placeholder="0.00" required
                                class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 pl-11 pr-4 text-white focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all placeholder:text-slate-600">
                        </div>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Merchant Vector Segment</label>
                        <div class="relative">
                            <i class="fa-solid fa-store absolute left-4 top-3.5 text-slate-500"></i>
                            <select name="category" required
                                class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 pl-11 pr-4 text-white appearance-none focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all">
                                {% for cat in categories %}
                                <option value="{{ cat }}">{{ cat }}</option>
                                {% endfor %}
                            </select>
                            <i class="fa-solid fa-chevron-down absolute right-4 top-4 text-slate-500 pointer-events-none text-xs"></i>
                        </div>
                    </div>

                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Target Origin Location</label>
                        <div class="relative">
                            <i class="fa-solid fa-globe absolute left-4 top-3.5 text-slate-500"></i>
                            <select name="country" required
                                class="w-full bg-slate-950/50 border border-slate-800 rounded-xl py-3 pl-11 pr-4 text-white appearance-none focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all">
                                {% for country in countries %}
                                <option value="{{ country }}">{{ country }}</option>
                                {% endfor %}
                            </select>
                            <i class="fa-solid fa-chevron-down absolute right-4 top-4 text-slate-500 pointer-events-none text-xs"></i>
                        </div>
                    </div>
                </div>

                <button type="submit" id="submit-btn"
                    class="w-full bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold py-4 rounded-xl shadow-lg shadow-blue-500/20 hover:opacity-90 active:scale-[0.99] transition-all flex items-center justify-center gap-2 cursor-pointer">
                    <i class="fa-solid fa-bolt"></i> Run Risk Analysis Report
                </button>
            </form>
        </section>

        <section class="lg:col-span-5 flex flex-col justify-between">
            <div id="output-placeholder" class="h-full bg-slate-900/20 border border-dashed border-slate-800 p-8 rounded-3xl flex flex-col items-center justify-center text-center">
                <div class="w-16 h-16 rounded-2xl bg-slate-900 flex items-center justify-center text-slate-600 mb-4 border border-slate-800">
                    <i class="fa-solid fa-chart-line text-2xl"></i>
                </div>
                <h3 class="text-md font-semibold text-slate-400">Awaiting Live Feed Telemetry</h3>
                <p class="text-xs text-slate-500 mt-1 max-w-xs">Fill out the transaction parameters and click run to stream predictions from your trained machine learning model asset.</p>
            </div>

            <div id="output-loader" class="hidden h-full bg-slate-900/40 border border-slate-800 backdrop-blur-xl p-8 rounded-3xl flex flex-col items-center justify-center text-center">
                <div class="w-12 h-12 rounded-full border-4 border-slate-800 border-t-cyan-500 animate-spin mb-4"></div>
                <h3 class="text-md font-mono text-cyan-400 uppercase tracking-widest animate-pulse">Processing Vectors...</h3>
            </div>

            <div id="output-display" class="hidden h-full bg-slate-900/40 border border-slate-800 backdrop-blur-xl p-6 rounded-3xl flex flex-col justify-between shadow-2xl transition-all duration-500">
                <div>
                    <div class="flex justify-between items-center mb-6">
                        <span class="text-xs font-mono uppercase tracking-wider text-slate-400">Evaluation Node Response</span>
                        <span id="badge" class="px-3 py-1 rounded-full text-xs font-bold tracking-wide"></span>
                    </div>

                    <div id="verdict-card" class="p-5 rounded-2xl border mb-6 flex items-start gap-4">
                        <div id="verdict-icon" class="p-3 rounded-xl text-white flex items-center justify-center text-lg"></div>
                        <div>
                            <h4 id="verdict-title" class="font-bold text-md"></h4>
                            <p id="verdict-desc" class="text-xs text-slate-300 mt-1 leading-relaxed"></p>
                        </div>
                    </div>

                    <div class="bg-slate-950/60 border border-slate-800/80 p-5 rounded-2xl space-y-4">
                        <div>
                            <div class="flex justify-between text-xs font-semibold text-slate-400 mb-1">
                                <span>Algorithmic Fraud Probability Score</span>
                                <span id="prob-text" class="font-mono font-bold text-white">0%</span>
                            </div>
                            <div class="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                                <div id="prob-bar" class="h-full transition-all duration-700 ease-out" style="width: 0%"></div>
                            </div>
                        </div>

                        <div class="grid grid-cols-2 gap-4 pt-2 border-t border-slate-800/60 text-center">
                            <div>
                                <span class="block text-[10px] uppercase font-bold text-slate-500 tracking-wider">Classification Verdict</span>
                                <span id="stat-verdict" class="text-sm font-bold mt-0.5 block"></span>
                            </div>
                            <div>
                                <span class="block text-[10px] uppercase font-bold text-slate-500 tracking-wider">Risk Profile Tier</span>
                                <span id="stat-tier" class="text-sm font-bold mt-0.5 block"></span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="text-[10px] text-center font-mono text-slate-600 border-t border-slate-800/40 pt-4 mt-4">
                    SHIELD CORE VER. 1.0.4 // INFERENCE INTERVAL &lt; 14ms
                </div>
            </div>
        </section>
    </main>

    <footer class="border-t border-slate-900 bg-slate-950 px-6 py-4 text-center text-xs text-slate-500">
        &copy; 2026 Credit Card Fraud Analytics System Framework Portfolio Project. Built with Flask & Tailwind CSS.
    </footer>

    <script>
        document.getElementById('prediction-form').addEventListener('submit', async function(e) {
            e.preventDefault();

            const placeholder = document.getElementById('output-placeholder');
            const loader = document.getElementById('output-loader');
            const display = document.getElementById('output-display');

            // Step 1: Transition UI to React-Like Loading Animation State
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
                // Step 2: Stream request payload to Flask REST endpoint
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();
                loader.classList.add('hidden');
                display.classList.remove('hidden');

                // Step 3: Parse metrics and update the UI container components dynamically
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
                    // Update layout config to Red (High Alert Fraud Warning Trigger)
                    badge.className = "px-3 py-1 rounded-full text-xs font-bold tracking-wide bg-rose-500/20 text-rose-400 border border-rose-500/30";
                    badge.innerText = "FRAUD DETECTED";

                    vCard.className = "p-5 rounded-2xl border bg-rose-950/20 border-rose-500/20 flex items-start gap-4 animate-shake";
                    vIcon.className = "p-3 rounded-xl bg-rose-500 text-white flex items-center justify-center text-lg shadow-lg shadow-rose-500/20";
                    vIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';

                    vTitle.className = "font-bold text-md text-rose-400";
                    vTitle.innerText = "High-Risk Fraud Profile Flagged";
                    vDesc.innerText = "This transaction breaks expected spending heuristics. The engine recommends immediate account authorization freeze actions.";

                    probBar.className = "h-full bg-gradient-to-r from-orange-500 to-rose-500 transition-all duration-700 ease-out";
                    sVerdict.className = "text-sm font-bold mt-0.5 block text-rose-400";
                    sVerdict.innerText = "FRAUDULENT";
                    sTier.className = "text-sm font-bold mt-0.5 block text-red-500";
                    sTier.innerText = "CRITICAL RISK";
                } else {
                    // Update layout config to Green (Safe Authorized Transaction Clearance)
                    badge.className = "px-3 py-1 rounded-full text-xs font-bold tracking-wide bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
                    badge.innerText = "CLEARED SECURE";

                    vCard.className = "p-5 rounded-2xl border bg-emerald-950/20 border-emerald-500/20 flex items-start gap-4";
                    vIcon.className = "p-3 rounded-xl bg-emerald-500 text-white flex items-center justify-center text-lg shadow-lg shadow-emerald-500/20";
                    vIcon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';

                    vTitle.className = "font-bold text-md text-emerald-400";
                    vTitle.innerText = "Transaction Cleared Securely";
                    vDesc.innerText = "The transaction aligns with safe purchasing models. Behavior metrics fall safely within authentic cardholder bounds.";

                    probBar.className = "h-full bg-gradient-to-r from-blue-500 to-emerald-500 transition-all duration-700 ease-out";
                    sVerdict.className = "text-sm font-bold mt-0.5 block text-emerald-400";
                    sVerdict.innerText = "LEGITIMATE";
                    sTier.className = "text-sm font-bold mt-0.5 block text-emerald-500";
                    sTier.innerText = "MINIMAL";
                }

            } catch (err) {
                console.error("Inference pipeline crash:", err);
                alert("API Endpoint Error. Check console script runtime logs.");
                loader.classList.add('hidden');
                placeholder.classList.remove('hidden');
            }
        });
    </script>
</body>
</html>
"""


# ==========================================
# REST API INFERENCE ENDPOINTS
# ==========================================
@app.route("/")
def index():
    """Serves the dashboard app page with dropdown contents dynamically injected."""
    return render_template_string(
        HTML_TEMPLATE, categories=categories, countries=countries
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Accepts JSON payloads asynchronously, scales features, maps dummies, and infers risk classification."""
    try:
        data = request.get_json()

        # Extract parameters sent from UI
        user_age = float(data["age"])
        user_amount = float(data["amount"])
        user_cat = str(data["category"]).strip().capitalize()
        user_country = str(data["country"]).strip()

        # Step A: Package continuous metrics and scale them using fitted parameters
        input_num_df = pd.DataFrame(
            [[user_age, user_amount]], columns=numerical_cols
        )
        scaled_nums = scaler.transform(input_num_df)[0]

        # Step B: Reconstruct a clean base layout dictionary mapping all features initialized to zero
        input_vector_dict = {col: 0 for col in feature_columns}

        # Step C: Populate continuous metrics fields
        input_vector_dict["card_holder_age"] = scaled_nums[0]
        input_vector_dict["amount"] = scaled_nums[1]

        # Step D: Handle hot-encoded flag indexes dynamically (with prefix strings from training)
        cat_dummy_col = f"merchant_category_{user_cat}"
        country_dummy_col = f"device_country_{user_country}"

        if cat_dummy_col in input_vector_dict:
            input_vector_dict[cat_dummy_col] = 1
        if country_dummy_col in input_vector_dict:
            input_vector_dict[country_dummy_col] = 1

        # Step E: Convert the dictionary into a standardized matrix row order format
        final_input_df = pd.DataFrame([input_vector_dict], columns=feature_columns)

        # Step F: Run inference predictions through the serialized model
        prediction = int(model.predict(final_input_df)[0])
        probability = float(model.predict_proba(final_input_df)[0][1])

        # Return response as JSON payload
        return jsonify(
            {"is_fraud": prediction, "fraud_probability": probability}
        )

    except Exception as e:
        print(f"ERROR: Inference pipeline encountered an anomaly: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ==========================================
# GUI WEB APPLICATION RUNNER
# ==========================================
if __name__ == "__main__":
    print("\n⚡ Web Service Online! Open your browser and navigate to:")
    print("➡️ http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)