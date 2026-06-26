import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# 1. Define UI Categorical Mapping for "Brand"
# Maps user-facing dropdown selections to the numerical labels your model expects
BRAND_MAPPING = {
    "Asus": 0,
    "Acer": 1,
    "Dell": 2,
    "HP": 3,
    "Lenovo": 4,
    "Apple": 5
}

# 2. Load the uploaded GradientBoostingRegressor model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'gradient_pkl.pkl')

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    print("SUCCESS: GradientBoostingRegressor successfully operational.")
except Exception as e:
    model = None
    print(f"ERROR: Could not load the pickle file model. Details: {e}")


# 3. Embedded HTML & CSS Frontend UI
COMBINED_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laptop Valuation Engine</title>
    
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    
    <style>
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(8, 10, 15) 100%);
        }
        .glass-panel {
            background: rgba(30, 41, 59, 0.45);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        select option {
            background-color: #0f172a;
            color: #e2e8f0;
        }
    </style>
</head>
<body class="min-h-screen text-slate-100 flex items-center justify-center p-4 sm:p-8 selection:bg-indigo-500 selection:text-white">

    <div class="w-full max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-12 gap-8 items-stretch">
        
        <div class="md:col-span-5 flex flex-col justify-between p-2">
            <div>
                <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-400/20 text-indigo-400 text-xs font-semibold uppercase tracking-wider mb-6">
                    <span class="w-2 h-2 rounded-full bg-indigo-400 animate-pulse"></span> Gradient Boosting System
                </div>
                <h1 class="text-4xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-200 to-indigo-300 bg-clip-text text-transparent leading-tight">
                    Laptop Price <br>Prediction Engine.
                </h1>
                <p class="mt-4 text-sm text-slate-400 leading-relaxed max-w-xs">
                    Input hardware metric architecture specifications to evaluate smart retail price indexes using historical market data curves instantly.
                </p>
            </div>

            <div class="mt-8 md:mt-0">
                {% if prediction %}
                <div class="glass-panel rounded-2xl p-6 border-l-4 border-l-emerald-500 shadow-xl transition-all duration-300">
                    <p class="text-xs uppercase font-bold tracking-widest text-slate-400">Estimated Fair Market Value</p>
                    <h2 class="text-4xl font-black text-emerald-400 mt-2 tracking-tight">{{ prediction }}</h2>
                    <p class="text-xs text-slate-500 mt-2">Valuation derived directly via Scikit-Learn Model parameters.</p>
                </div>
                {% elif error_msg %}
                <div class="glass-panel rounded-2xl p-6 border-l-4 border-l-rose-500 bg-rose-950/20 shadow-xl">
                    <p class="text-xs uppercase font-bold tracking-widest text-rose-400">Processing Error</p>
                    <p class="text-sm text-slate-300 mt-1">{{ error_msg }}</p>
                </div>
                {% else %}
                <div class="glass-panel rounded-2xl p-6 border-dashed border-2 border-slate-700 flex items-center justify-center text-center py-10 opacity-60">
                    <div>
                        <svg class="w-8 h-8 text-slate-500 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"></path>
                        </svg>
                        <p class="text-xs text-slate-400">Fill attributes and tap Predict Price to generate pricing insights</p>
                    </div>
                </div>
                {% endif %}
            </div>
        </div>

        <div class="md:col-span-7 glass-panel rounded-3xl p-6 sm:p-8 shadow-2xl">
            <form action="/predict" method="POST" class="space-y-5">
                
                <div>
                    <label for="Brand" class="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">Device Manufacturer Brand Label</label>
                    <select name="Brand" id="Brand" required 
                            class="w-full bg-slate-900/60 border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition cursor-pointer">
                        {% for brand in brands %}
                        <option value="{{ brand }}" {% if previous_inputs and previous_inputs.get('Brand') == brand %}selected{% endif %}>{{ brand }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="grid grid-cols-1 sm:grid-cols-2 gap-5">
                    <div>
                        <label for="Processor_Speed" class="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">Processor Speed (GHz)</label>
                        <input type="number" step="0.01" name="Processor_Speed" id="Processor_Speed" required placeholder="e.g. 2.8"
                               value="{{ previous_inputs.get('Processor_Speed', '') if previous_inputs else '' }}"
                               class="w-full bg-slate-900/60 border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition">
                    </div>

                    <div>
                        <label for="RAM_Size" class="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">RAM Size (GB)</label>
                        <input type="number" step="1" name="RAM_Size" id="RAM_Size" required placeholder="e.g. 16"
                               value="{{ previous_inputs.get('RAM_Size', '') if previous_inputs else '' }}"
                               class="w-full bg-slate-900/60 border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition">
                    </div>
                </div>

                <div class="grid grid-cols-1 sm:grid-cols-2 gap-5">
                    <div>
                        <label for="Storage_Capacity" class="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">Storage Capacity (GB)</label>
                        <input type="number" step="1" name="Storage_Capacity" id="Storage_Capacity" required placeholder="e.g. 512"
                               value="{{ previous_inputs.get('Storage_Capacity', '') if previous_inputs else '' }}"
                               class="w-full bg-slate-900/60 border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition">
                    </div>

                    <div>
                        <label tabfor="Screen_Size" class="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">Screen Size (Inches)</label>
                        <input type="number" step="0.1" name="Screen_Size" id="Screen_Size" required placeholder="e.g. 15.6"
                               value="{{ previous_inputs.get('Screen_Size', '') if previous_inputs else '' }}"
                               class="w-full bg-slate-900/60 border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition">
                    </div>
                </div>

                <div>
                    <label for="Weight" class="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">Chassis Weight (kg)</label>
                    <input type="number" step="0.01" name="Weight" id="Weight" required placeholder="e.g. 1.65"
                           value="{{ previous_inputs.get('Weight', '') if previous_inputs else '' }}"
                           class="w-full bg-slate-900/60 border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition">
                </div>

                <button type="submit" 
                        class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm py-4 rounded-xl shadow-lg shadow-indigo-600/20 active:scale-[0.99] transition duration-150 mt-4 cursor-pointer flex items-center justify-center gap-2">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                    </svg>
                    Predict Price Matrix
                </button>

            </form>
        </div>

    </div>

</body>
</html>
"""


@app.route('/', methods=['GET'])
def index():
    return render_template_string(COMBINED_TEMPLATE, brands=list(BRAND_MAPPING.keys()), prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(COMBINED_TEMPLATE, brands=list(BRAND_MAPPING.keys()), 
                                      error_msg="Model pkl file is missing or unreadable on the server side.")
    
    try:
        # 1. Fetch form variables
        brand_selected = request.form.get('Brand')
        processor_speed = float(request.form.get('Processor_Speed'))
        ram_size = float(request.form.get('RAM_Size'))
        storage_capacity = float(request.form.get('Storage_Capacity'))
        screen_size = float(request.form.get('Screen_Size'))
        weight = float(request.form.get('Weight'))

        # 2. Apply categorical label encoding mapping matrix
        brand_encoded = BRAND_MAPPING.get(brand_selected, 0)

        # 3. Create input structure to match exact feature sequences
        # Sequential Format: [Brand, Processor_Speed, RAM_Size, Storage_Capacity, Screen_Size, Weight]
        input_features = np.array([[
            brand_encoded, 
            processor_speed, 
            ram_size, 
            storage_capacity, 
            screen_size, 
            weight
        ]])

        # 4. Generate prediction using loaded model pipeline
        predicted_value = model.predict(input_features)[0]
        formatted_price = f"${predicted_value:,.2f}"

        return render_template_string(
            COMBINED_TEMPLATE, 
            brands=list(BRAND_MAPPING.keys()), 
            prediction=formatted_price,
            previous_inputs=request.form
        )

    except Exception as e:
        return render_template_string(
            COMBINED_TEMPLATE, 
            brands=list(BRAND_MAPPING.keys()), 
            error_msg=f"Inference Failure: {str(e)}"
        )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
