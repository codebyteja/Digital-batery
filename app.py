"""
=============================================================================
app.py — Flask Backend for EV Battery Digital Twin Dashboard
=============================================================================
This is the main web server that:
  - Serves the single-page dashboard (index.html)
  - Provides REST API endpoints for predictions, advisor, cost estimation,
    chatbot, and PDF report upload
  - Loads pre-trained ML models (soh_model.pkl, rul_model.pkl)
  - Reads historical battery data from processed_battery_data.csv

Endpoints:
  GET  /                           → Main dashboard page
  POST /predict                    → SoH & RUL prediction
  GET  /battery_history/<id>       → Historical SoH trend data
  POST /advisor                    → Charging advisor recommendations
  POST /cost                       → Cost impact estimator
  POST /chatbot                    → Rule-based chatbot
  POST /upload_report              → PDF report parser
  GET  /sample_reports/<filename>  → Download sample report PDFs
  GET  /metrics                    → Model accuracy metrics

Usage:
  python app.py
  Then open http://localhost:5000
=============================================================================
"""

import os
import re
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, send_from_directory

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload

# Paths to data and models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "processed_battery_data.csv")
SOH_MODEL_PATH = os.path.join(BASE_DIR, "models", "soh_model.pkl")
RUL_MODEL_PATH = os.path.join(BASE_DIR, "models", "rul_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
SAMPLE_REPORTS_DIR = os.path.join(BASE_DIR, "sample_reports")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load models and data at startup (cached in memory for fast predictions)
# ---------------------------------------------------------------------------
soh_model = None
rul_model = None
battery_data = None

# Session store for chatbot context (simple in-memory, single-user demo)
session_state = {
    "last_soh": None,
    "last_rul": None,
    "last_status": None,
}


def load_resources():
    """Load ML models and battery data into memory at startup."""
    global soh_model, rul_model, battery_data

    # Load SoH prediction model
    if os.path.exists(SOH_MODEL_PATH):
        soh_model = joblib.load(SOH_MODEL_PATH)
        print(f"[OK] SoH model loaded from {SOH_MODEL_PATH}")
    else:
        print(f"[WARNING] SoH model not found at {SOH_MODEL_PATH}")

    # Load RUL prediction model
    if os.path.exists(RUL_MODEL_PATH):
        rul_model = joblib.load(RUL_MODEL_PATH)
        print(f"[OK] RUL model loaded from {RUL_MODEL_PATH}")
    else:
        print(f"[WARNING] RUL model not found at {RUL_MODEL_PATH}")

    # Load battery CSV data for historical charts
    if os.path.exists(DATA_FILE):
        battery_data = pd.read_csv(DATA_FILE)
        print(f"[OK] Battery data loaded: {len(battery_data)} rows")
    else:
        print(f"[WARNING] Battery data not found at {DATA_FILE}")


# Initialize resources when module is imported by Gunicorn
load_resources()

# ---------------------------------------------------------------------------
# Route: Main Dashboard Page
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Render the main single-page dashboard."""
    # Pass available battery IDs to the template for the dropdown
    battery_ids = []
    if battery_data is not None:
        battery_ids = sorted(battery_data["battery_id"].unique().tolist())
    return render_template("index.html", battery_ids=battery_ids)


# ---------------------------------------------------------------------------
# Route: SoH & RUL Prediction
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict State of Health and Remaining Useful Life.
    
    Input (JSON):
      { voltage, current, temperature, cycle_number }
    
    Output (JSON):
      { soh, rul_cycles, rul_years, status, status_class }
    
    Status thresholds:
      SoH > 85% → "Healthy"
      SoH 60-85% → "Monitor"
      SoH < 60% → "Service Recommended"
    """
    if soh_model is None or rul_model is None:
        return jsonify({"error": "Models not loaded. Run train_model.py first."}), 500

    try:
        data = request.get_json()
        voltage = float(data["voltage"])
        current = float(data["current"])
        temperature = float(data["temperature"])
        cycle_number = int(data["cycle_number"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    # --- Predict SoH ---
    # SoH model features: [voltage, current, temperature, cycle_number]
    soh_input = np.array([[voltage, current, temperature, cycle_number]])
    soh_pred = float(soh_model.predict(soh_input)[0])
    soh_pred = round(max(0, min(100, soh_pred)), 1)  # Clamp to 0-100%

    # --- Predict RUL ---
    # RUL model features: [voltage, current, temperature, cycle_number, SoH]
    rul_input = np.array([[voltage, current, temperature, cycle_number, soh_pred]])
    rul_pred = float(rul_model.predict(rul_input)[0])
    rul_pred = round(max(0, rul_pred), 0)  # No negative RUL

    # Convert RUL from cycles to estimated years (assuming ~300 cycles/year)
    rul_years = round(rul_pred / 300, 1)

    # --- Determine status ---
    if soh_pred > 85:
        status = "Healthy"
        status_class = "healthy"
    elif soh_pred >= 60:
        status = "Monitor"
        status_class = "monitor"
    else:
        status = "Service Recommended"
        status_class = "service"

    # Update session state for chatbot
    session_state["last_soh"] = soh_pred
    session_state["last_rul"] = rul_pred
    session_state["last_status"] = status

    return jsonify({
        "soh": soh_pred,
        "rul_cycles": int(rul_pred),
        "rul_years": rul_years,
        "status": status,
        "status_class": status_class,
    })


# ---------------------------------------------------------------------------
# Route: Battery History (for Chart.js line chart)
# ---------------------------------------------------------------------------
@app.route("/battery_history/<battery_id>")
def battery_history(battery_id):
    """
    Return historical SoH trend data for a specific battery.
    
    Output (JSON):
      { cycles: [...], soh_values: [...], battery_id: "B0005" }
    """
    if battery_data is None:
        return jsonify({"error": "No battery data available."}), 500

    # Filter for the requested battery
    mask = battery_data["battery_id"] == battery_id
    filtered = battery_data[mask].sort_values("cycle_number")

    if filtered.empty:
        return jsonify({"error": f"No data found for battery {battery_id}"}), 404

    return jsonify({
        "battery_id": battery_id,
        "cycles": filtered["cycle_number"].tolist(),
        "soh_values": filtered["SoH"].tolist(),
        "voltage_values": filtered["voltage"].tolist(),
        "temperature_values": filtered["temperature"].tolist(),
    })


# ---------------------------------------------------------------------------
# Route: Charging Advisor
# ---------------------------------------------------------------------------
@app.route("/advisor", methods=["POST"])
def advisor():
    """
    Recommend optimal charging strategy based on daily driving distance.
    
    Input (JSON):
      { daily_distance_km }
    
    Output (JSON):
      { recommended_soc_range, comparison_table }
    
    Logic (rule-based, no ML):
      - Short daily drives (<50 km) → 20%-80% SoC range is ideal
      - Medium drives (50-150 km) → 15%-85% range
      - Long drives (>150 km) → 10%-90% range (need more capacity)
    """
    try:
        data = request.get_json()
        daily_km = float(data["daily_distance_km"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    # Determine recommended SoC range based on daily usage
    if daily_km < 50:
        soc_min, soc_max = 20, 80
        usage_level = "Light"
    elif daily_km < 150:
        soc_min, soc_max = 15, 85
        usage_level = "Moderate"
    else:
        soc_min, soc_max = 10, 90
        usage_level = "Heavy"

    # Charging method comparison table
    # These are realistic estimates based on EV industry data
    comparison = [
        {
            "method": "DC Fast Charging (Level 3)",
            "degradation_percent_per_year": round(2.5 + daily_km * 0.005, 1),
            "charging_time": "20-40 min (20% → 80%)",
            "estimated_lifespan_years": round(max(5, 12 - daily_km * 0.02), 1),
            "recommendation": "Use sparingly — only for long trips",
        },
        {
            "method": "Normal AC Charging (Level 2)",
            "degradation_percent_per_year": round(1.5 + daily_km * 0.002, 1),
            "charging_time": "4-8 hours (20% → 80%)",
            "estimated_lifespan_years": round(max(7, 15 - daily_km * 0.015), 1),
            "recommendation": "Best daily option — balanced speed and longevity",
        },
        {
            "method": "Slow Trickle Charging (Level 1)",
            "degradation_percent_per_year": round(0.8 + daily_km * 0.001, 1),
            "charging_time": "12-20 hours (20% → 80%)",
            "estimated_lifespan_years": round(max(9, 18 - daily_km * 0.01), 1),
            "recommendation": "Gentlest on battery — ideal for overnight charging",
        },
    ]

    return jsonify({
        "daily_distance_km": daily_km,
        "usage_level": usage_level,
        "recommended_soc_min": soc_min,
        "recommended_soc_max": soc_max,
        "comparison": comparison,
    })


# ---------------------------------------------------------------------------
# Route: Cost Estimator
# ---------------------------------------------------------------------------
@app.route("/cost", methods=["POST"])
def cost():
    """
    Estimate degradation cost impact and potential savings.
    
    Input (JSON):
      { replacement_cost, current_soh }
    
    Output (JSON):
      { cost_impact, savings_3yr, current_habit_cost, optimized_cost }
    
    Formula:
      cost_impact = replacement_cost * (100 - current_soh) / 100
      
      If following advisor recommendations, degradation slows by ~30%,
      leading to proportional cost savings over 3 years.
    """
    try:
        data = request.get_json()
        replacement_cost = float(data["replacement_cost"])
        current_soh = float(data["current_soh"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    # Current degradation cost
    degradation_pct = (100 - current_soh) / 100
    cost_impact = round(replacement_cost * degradation_pct, 2)

    # Project 3-year costs under current vs optimized habits
    # Assume ~3% SoH loss per year under current habits
    # Optimized habits reduce this to ~1.8% per year (40% improvement)
    current_annual_loss = 3.0
    optimized_annual_loss = 1.8

    current_3yr_soh = max(0, current_soh - (current_annual_loss * 3))
    optimized_3yr_soh = max(0, current_soh - (optimized_annual_loss * 3))

    current_3yr_cost = round(replacement_cost * (100 - current_3yr_soh) / 100, 2)
    optimized_3yr_cost = round(replacement_cost * (100 - optimized_3yr_soh) / 100, 2)
    savings_3yr = round(current_3yr_cost - optimized_3yr_cost, 2)

    return jsonify({
        "current_soh": current_soh,
        "replacement_cost": replacement_cost,
        "cost_impact": cost_impact,
        "current_3yr_cost": current_3yr_cost,
        "optimized_3yr_cost": optimized_3yr_cost,
        "savings_3yr": savings_3yr,
        "current_3yr_soh": round(current_3yr_soh, 1),
        "optimized_3yr_soh": round(optimized_3yr_soh, 1),
    })


# ---------------------------------------------------------------------------
# Route: Chatbot (Rule-Based)
# ---------------------------------------------------------------------------
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    msg = data.get("message", "").lower()
    
    # We now get context explicitly from the frontend, but fallback to session state
    context = data.get("context", {})
    soh = context.get("soh") or session_state.get("last_soh")
    rul = context.get("rul") or session_state.get("last_rul")
    defect = context.get("defect_result")
    
    status = session_state.get("last_status", "Unknown")
    
    if soh is not None:
        rul_val = int(rul) if rul is not None else "unknown"
        ctx_str = f"Your battery's SoH is {soh}% with ~{rul_val} cycles remaining ({status})."
    else:
        ctx_str = "I don't have your health prediction yet."

    if any(kw in msg for kw in ["hello", "hi", "hey", "help"]):
        reply = f"👋 Hello! I'm your PRO EV Battery Assistant. {ctx_str} I can analyze your defect scans, estimate degradation cost, or give maintenance tips!"
        
    elif any(kw in msg for kw in ["soh", "state of health", "healthy", "health"]):
        if soh is not None:
            if soh > 85:
                reply = f"✅ {ctx_str} It's in excellent condition!"
            elif soh >= 60:
                reply = f"⚠️ {ctx_str} Aging but functional. Optimize charging to slow degradation."
            else:
                reply = f"🔴 {ctx_str} Significant degradation detected. Schedule a check."
        else:
            reply = "I can estimate your Battery Health (SoH) if you enter the car's parameters on the Dashboard."
            
    elif any(kw in msg for kw in ["rul", "remaining", "lifespan", "cycles", "life"]):
        if rul is not None:
            years = round(rul / 300, 1)
            reply = f"⏱️ You have approx {int(rul)} cycles left (about {years} years at average usage). "
            if int(rul) < 200:
                reply += "Consider preparing for a replacement soon."
            else:
                reply += "Plenty of life remaining!"
        else:
            reply = "I don't have your Remaining Useful Life yet. Predict it on the Dashboard first."

    elif any(kw in msg for kw in ["defect", "damage", "image", "scan", "xray", "x-ray"]):
        if "how to" in msg or ("what is" in msg and "scan" in msg):
            reply = "To scan for damage, scroll to the 'Visual Defect Scanner' section on the Dashboard. You can upload an X-Ray, plain image, or a ZIP file with multiple images. My AI Vision model will instantly analyze them for physical defects!"
        elif defect:
            n = defect.get("n_damaged", 0)
            good = defect.get("n_good", 0)
            sev = defect.get("severity", "Unknown")
            total = n + good
            
            if "percent" in msg or "percentage" in msg:
                if total > 0:
                    pct = round((n / total) * 100, 1)
                    reply = f"Based on the {total} images scanned, {pct}% of the cells are damaged."
                else:
                    reply = "I couldn't calculate a percentage because no valid cells were found in the last scan."
            elif n > 0:
                reply = f"⚠️ Based on the images uploaded, I detected {n} damaged cell(s) out of {total}. Severity is rated as: {sev}. You should schedule a service immediately!"
            else:
                reply = "✅ Good news! The uploaded images showed 0 damaged batteries. They all look healthy!"
        else:
            reply = "Upload X-ray or normal images of your battery cells in the Dashboard, and my AI Vision model will scan them for physical defects."

    elif any(kw in msg for kw in ["temperature", "temp", "hot", "cold"]):
        if context.get("temperature"):
            reply = f"🌡️ The temperature from your current data is {context['temperature']} °C. Optimal operating temperature for EV batteries is between 20-30°C."
        else:
            reply = "I don't see any temperature data right now. If you upload a PDF or enter it manually on the Dashboard, I can read it!"

    elif any(kw in msg for kw in ["voltage", "volts", "v"]):
        if context.get("voltage"):
            reply = f"⚡ The voltage from your current data is {context['voltage']} V. Most EVs operate between 350V and 800V depending on the model architecture."
        else:
            reply = "I don't have your voltage data right now. You can enter it on the Dashboard!"

    elif any(kw in msg for kw in ["current", "amps", "amperage", "ampere"]):
        if context.get("current"):
            reply = f"🔌 The current flow from your data is {context['current']} A. Higher sustained currents generate more heat and can accelerate degradation."
        else:
            reply = "I don't have your current data right now. You can enter it on the Dashboard!"

    elif any(kw in msg for kw in ["cost", "money", "price", "expensive", "save", "estimator"]):
        reply = "💰 Battery replacement costs $5,000-$15,000. Use our 'Cost Estimator' tab to calculate exactly how much degradation is costing you based on your car's price."

    elif any(kw in msg for kw in ["extend", "prolong", "last longer", "improve", "tips", "service"]):
        reply = "🌟 Tips to extend life & reduce services:\n• Keep charge between 20%-80%\n• Avoid frequent DC fast charging\n• Use Visual Defect Scanning at annual services."

    elif any(kw in msg for kw in ["about", "project", "digital twin", "what is"]):
        reply = "🔬 This is a PRO AI Digital Twin! It uses Machine Learning (trained on NASA data) for health prediction, and Computer Vision for physical defect scanning."

    else:
        reply = f"🤔 I'm not sure. {ctx_str} Try asking about: 'Show SoH', 'Check battery life', 'How much damaged?', or 'Cost estimator'."

    return jsonify({"reply": reply})


# ---------------------------------------------------------------------------
# Route: Universal File Upload (handles .zip, .mat, .csv, .pdf)
# ---------------------------------------------------------------------------
@app.route("/upload_universal", methods=["POST"])
def upload_universal():
    """
    Accept ANY file type and process accordingly:
      .zip  -> extract, find .mat/.csv inside, process
      .mat  -> process directly as battery cycle data
      .csv  -> read if columns match expected structure
      .pdf  -> extract labeled values with regex (existing logic)

    After extraction, VERIFY data against realistic ranges:
      voltage:     2.0 - 4.3 V
      current:     0.0 - 10.0 A
      temperature: -10 - 60 C
      cycle_number: 1 - 10000

    Returns JSON with extracted values, verification status, and any errors.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    filename = file.filename.lower()
    extracted = {}
    file_type = "unknown"
    message = ""

    try:
        # ── Determine file type and process ──
        if filename.endswith(".pdf"):
            file_type = "pdf"
            extracted, message = _process_pdf(file)

        elif filename.endswith(".csv"):
            file_type = "csv"
            extracted, message = _process_csv(file)

        elif filename.endswith(".mat"):
            file_type = "mat"
            extracted, message = _process_mat(file)

        elif filename.endswith(".zip"):
            file_type = "zip"
            extracted, message = _process_zip(file)

        else:
            return jsonify({
                "error": f"Unsupported file type: {os.path.splitext(filename)[1]}. "
                         "Accepted: .zip, .mat, .csv, .pdf"
            }), 400

        # ── Verify extracted data ──
        verification = _verify_data(extracted)

        return jsonify({
            "file_type": file_type,
            "filename": file.filename,
            "message": message,
            "extracted": extracted,
            "verification": verification,
        })

    except Exception as e:
        import traceback
        err_msg = f"Error processing file {file.filename}: {str(e)}\n{traceback.format_exc()}"
        print("\n" + "="*60)
        print("  UPLOAD UNIVERSAL ERROR")
        print("="*60)
        print(err_msg)
        print("="*60 + "\n")
        return jsonify({"error": f"Failed during universal processing. Reason: {str(e)}"}), 500


def _process_pdf(file):
    """Extract battery parameters from a PDF using pdfplumber + regex."""
    import pdfplumber

    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    if not text.strip():
        return {}, "Could not auto-read this PDF -- please enter values manually."

    patterns = {
        "voltage": r"Voltage[:\s]+(\d+\.?\d*)",
        "current": r"Current[:\s]+(\d+\.?\d*)",
        "temperature": r"Temperature[:\s]+(\d+\.?\d*)",
        "cycle_number": r"Cycle\s*Count[:\s]+(\d+\.?\d*)",
        "capacity": r"Capacity[:\s]+(\d+\.?\d*)",
    }

    extracted = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted[field] = float(match.group(1))

    if not extracted:
        return {}, "Could not auto-read this PDF -- please enter values manually."

    return extracted, f"Extracted {len(extracted)} values from PDF report."


def _process_csv(file):
    """Read a CSV file and extract battery parameters from the first/last row."""
    import io

    try:
        content = file.read().decode("utf-8", errors="ignore")
        df = pd.read_csv(io.StringIO(content))

        # Normalize column names to lowercase
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        extracted = {}

        # Map expected columns
        col_map = {
            "voltage": ["voltage", "voltage_measured", "v"],
            "current": ["current", "current_measured", "i"],
            "temperature": ["temperature", "temperature_measured", "temp", "t"],
            "cycle_number": ["cycle_number", "cycle", "cycle_count", "cycles"],
            "capacity": ["capacity", "cap"],
            "soh": ["soh", "state_of_health"],
        }

        for field, aliases in col_map.items():
            for alias in aliases:
                if alias in df.columns:
                    # Use the last row (most recent cycle)
                    val = df[alias].dropna().iloc[-1] if len(df[alias].dropna()) > 0 else None
                    if val is not None:
                        extracted[field] = float(val)
                    break

    except Exception as e:
        import traceback
        err_trace = traceback.format_exc()
        print(f"\n[ERROR in _process_csv]\n{err_trace}")
        return {}, f"CSV processing failed: {str(e)}"

    if not extracted:
        return {}, f"CSV loaded ({len(df)} rows) but no matching columns found."

    return extracted, f"Extracted values from CSV ({len(df)} rows, last cycle)."


def _process_mat(file):
    """Load a .mat file and extract averaged discharge cycle data."""
    from scipy.io import loadmat
    import tempfile

    # Save to temp file (loadmat needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    try:
        mat_data = loadmat(tmp_path)
        META_KEYS = {"__header__", "__version__", "__globals__"}
        data_keys = [k for k in mat_data.keys() if k not in META_KEYS]

        if not data_keys:
            return {}, "No usable data found in .mat file."

        battery_key = data_keys[0]

        # Try to extract cycle data from NASA-format struct
        try:
            battery_struct = mat_data[battery_key]
            try:
                cycles = battery_struct["cycle"][0, 0][0]
            except (IndexError, KeyError):
                cycles = battery_struct["cycle"][0][0][0]

            # Collect averages from all discharge cycles
            voltages, currents, temps, capacities = [], [], [], []
            for cycle in cycles:
                try:
                    cycle_type = str(cycle["type"][0]).lower()
                    if "discharge" not in cycle_type:
                        continue
                    data = cycle["data"][0, 0]
                    voltages.append(float(np.mean(data["Voltage_measured"].flatten())))
                    currents.append(float(np.mean(np.abs(data["Current_measured"].flatten()))))
                    temps.append(float(np.mean(data["Temperature_measured"].flatten())))
                    try:
                        capacities.append(float(data["Capacity"].flatten()[0]))
                    except (KeyError, IndexError):
                        pass
                except Exception:
                    continue

            if voltages:
                extracted = {
                    "voltage": round(voltages[-1], 4),
                    "current": round(currents[-1], 4),
                    "temperature": round(temps[-1], 2),
                    "cycle_number": len(voltages),
                }
                if capacities:
                    extracted["capacity"] = round(capacities[-1], 4)
                return extracted, f"Extracted data from {len(voltages)} discharge cycles (battery: {battery_key})."

        except (KeyError, IndexError, TypeError) as e:
            import traceback
            err_trace = traceback.format_exc()
            print(f"\n[ERROR in _process_mat structure parse]\n{err_trace}")
            return {}, f"Could not parse .mat file structure: {str(e)}"

        return {}, "Could not parse .mat file structure -- unsupported format."
    except Exception as e:
        import traceback
        err_trace = traceback.format_exc()
        print(f"\n[ERROR in _process_mat load]\n{err_trace}")
        return {}, f"Failed to load .mat file: {str(e)}"

    finally:
        os.unlink(tmp_path)


def _process_zip(file):
    """Extract a .zip, search for .mat/.csv files inside, process the first one found."""
    import zipfile
    import tempfile
    
    # Save zip to disk instead of memory for large files
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_zip = tmp.name
    file.save(tmp_zip)

    try:
        if not zipfile.is_zipfile(tmp_zip):
            os.unlink(tmp_zip)
            return {}, "File is not a valid ZIP archive."

        with zipfile.ZipFile(tmp_zip, "r") as zf:
            # List all files in the zip
            all_files = zf.namelist()

            # Find .mat and .csv files
            mat_files = [f for f in all_files if f.lower().endswith(".mat") and not f.startswith("__")]
            csv_files = [f for f in all_files if f.lower().endswith(".csv") and not f.startswith("__")]

            if not mat_files and not csv_files:
                return {}, f"ZIP contains {len(all_files)} files but no .mat or .csv files found."

            # Prefer .mat files, fall back to .csv
            if mat_files:
                target_file = mat_files[0]
                with zf.open(target_file) as f:
                    from scipy.io import loadmat
                    import tempfile as tf

                    with tf.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name

                    try:
                        # Re-use the mat processing logic
                        from werkzeug.datastructures import FileStorage
                        with open(tmp_path, "rb") as mat_f:
                            # Create a file-like wrapper
                            class FakeFile:
                                def __init__(self, fobj):
                                    self._f = fobj
                                def read(self):
                                    return self._f.read()
                            extracted, msg = _process_mat(FakeFile(mat_f))
                            return extracted, f"ZIP: found {len(mat_files)} .mat file(s). {msg}"
                    finally:
                        os.unlink(tmp_path)

            elif csv_files:
                target_file = csv_files[0]
                with zf.open(target_file) as f:
                    content = f.read()

                    class FakeCSVFile:
                        def __init__(self, data):
                            self._data = data
                        def read(self):
                            return self._data

                    extracted, msg = _process_csv(FakeCSVFile(content))
                    return extracted, f"ZIP: found {len(csv_files)} .csv file(s). {msg}"
    except Exception as e:
        import traceback
        err_trace = traceback.format_exc()
        print(f"\n[ERROR in _process_zip]\n{err_trace}")
        return {}, f"ZIP processing failed: {str(e)}"
    finally:
        os.unlink(tmp_zip)

    return {}, "Could not process any files from the ZIP."


def _verify_data(extracted):
    """
    Verify extracted values are within realistic physical ranges.

    Returns dict with:
      passed: bool - True if all present values are within range
      checks: list of { field, value, valid, reason }
    """
    RANGES = {
        "voltage":     (0.0, 10.0, "V",  "Expected 0.0 - 10.0 V"),
        "current":     (-50.0, 50.0, "A",  "Expected -50.0 - 50.0 A"),
        "temperature": (-50.0, 100.0, "C", "Expected -50 to 100 C"),
        "cycle_number": (0, 20000, "",    "Expected 0 - 20,000"),
    }

    checks = []
    all_present_passed = True
    required_found = 0

    for field, (lo, hi, unit, reason) in RANGES.items():
        if field in extracted:
            val = extracted[field]
            valid = lo <= val <= hi
            checks.append({
                "field": field,
                "value": val,
                "unit": unit,
                "valid": valid,
                "reason": "OK" if valid else reason,
            })
            if not valid:
                all_present_passed = False
            required_found += 1
        else:
            checks.append({
                "field": field,
                "value": None,
                "unit": "",
                "valid": True,  # Missing is allowed as long as total >= 3
                "reason": "Missing (will use default value)",
            })

    return {
        "passed": all_present_passed and required_found >= 3,
        "checks": checks,
        "fields_found": required_found,
    }


# ---------------------------------------------------------------------------
# Route: Upload & Parse PDF Report (legacy endpoint, kept for compatibility)
# ---------------------------------------------------------------------------
@app.route("/upload_report", methods=["POST"])
def upload_report():
    """
    Accept a PDF file upload, extract battery parameters using regex.
    
    Looks for patterns like:
      "Voltage: 3.82 V"
      "Current: 1.15 A"
      "Temperature: 28.5 C"
      "Cycle Count: 45"
      "Capacity: 1.78 Ah"
    
    Returns extracted values as JSON for auto-filling the prediction form.
    """
    try:
        import pdfplumber
    except ImportError:
        return jsonify({"error": "pdfplumber not installed. Run: pip install pdfplumber"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted."}), 400

    try:
        # Extract text from all pages of the PDF
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            return jsonify({
                "message": "Could not auto-read this PDF -- please enter values manually.",
                "extracted": {}
            })

        # Regex patterns to find labeled values
        # Each pattern looks for the label followed by a number
        patterns = {
            "voltage": r"Voltage[:\s]+(\d+\.?\d*)",
            "current": r"Current[:\s]+(\d+\.?\d*)",
            "temperature": r"Temperature[:\s]+(\d+\.?\d*)",
            "cycle_number": r"Cycle\s*Count[:\s]+(\d+\.?\d*)",
            "capacity": r"Capacity[:\s]+(\d+\.?\d*)",
        }

        extracted = {}
        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted[field] = float(match.group(1))

        if not extracted:
            return jsonify({
                "message": "Could not auto-read this PDF -- please enter values manually.",
                "extracted": {}
            })

        return jsonify({
            "message": f"Successfully extracted {len(extracted)} values from the report.",
            "extracted": extracted,
        })

    except Exception as e:
        return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500



# ---------------------------------------------------------------------------
# Route: Serve Sample Report PDFs for Download
# ---------------------------------------------------------------------------
@app.route("/sample_reports/<filename>")
def serve_sample_report(filename):
    """Allow users to download sample battery report PDFs."""
    return send_from_directory(SAMPLE_REPORTS_DIR, filename, as_attachment=True)


# ---------------------------------------------------------------------------
# Route: List available sample reports
# ---------------------------------------------------------------------------
@app.route("/sample_reports_list")
def sample_reports_list():
    """Return list of available sample report filenames."""
    if os.path.exists(SAMPLE_REPORTS_DIR):
        files = [f for f in os.listdir(SAMPLE_REPORTS_DIR) if f.endswith(".pdf")]
        return jsonify({"reports": sorted(files)})
    return jsonify({"reports": []})


# ---------------------------------------------------------------------------
# Route: Model Metrics (for About page)
# ---------------------------------------------------------------------------
@app.route("/metrics")
def metrics():
    """Return model evaluation metrics (R², RMSE) for display."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return jsonify(json.load(f))
    return jsonify({
        "soh_model": {"r2": "N/A", "rmse": "N/A"},
        "rul_model": {"r2": "N/A", "rmse": "N/A"},
    })

# ---------------------------------------------------------------------------
# Route: Defect Detection
# ---------------------------------------------------------------------------
@app.route("/detect_defect", methods=["POST"])
def detect_defect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    allowed_ext = (".png", ".jpg", ".jpeg", ".pdf", ".zip")
    if not file.filename.lower().endswith(allowed_ext):
        return jsonify({"error": "Accepted formats: JPG, PNG, PDF, ZIP"}), 400

    import tempfile
    suffix = os.path.splitext(file.filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        import sys
        sys.path.append(BASE_DIR)
        from models.defect_classifier import predict_defect
        res = predict_defect(tmp_path)
        return jsonify({
            "result":        res.get("result"),
            "confidence":    res.get("confidence"),
            "severity":      res.get("severity", ""),
            "is_demo":       res.get("is_demo"),
            "image_mode":    res.get("image_mode"),
            "bulk_info":     res.get("bulk_info", ""),
            "breakdown":     res.get("breakdown", []),
            "composite_b64": res.get("composite_b64"),
            "n_damaged":     res.get("n_damaged", 0),
            "n_good":        res.get("n_good", 0),
            "action":        res.get("action", ""),
        })
    except Exception as e:
        import traceback
        print(f"\n[ERROR in detect_defect]\n{traceback.format_exc()}")
        return jsonify({"error": f"Defect detection failed: {str(e)}"}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  EV Battery Digital Twin — Starting Server")
    print("=" * 60)

    load_resources()

    print("\n[WEB] Dashboard: http://localhost:5000")
    print("=" * 60)

    # debug=True for development — auto-reloads on code changes
    # Set debug=False for production/demo
    app.run(debug=True, host="0.0.0.0", port=5000)
