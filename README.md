# 🔋 AI-Powered Digital Twin for EV Battery Health Monitoring

A full-stack web application that uses Machine Learning to predict EV battery **State of Health (SoH)** and **Remaining Useful Life (RUL)**, built with Flask, scikit-learn, and Chart.js.

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3+-000000?style=flat&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-f7931e?style=flat&logo=scikit-learn&logoColor=white)

---

## 📋 Features

| Feature | Description |
|---------|-------------|
| **SoH Prediction** | Predicts battery State of Health (%) using Random Forest |
| **RUL Prediction** | Estimates Remaining Useful Life in cycles and years |
| **SoH Trend Chart** | Interactive line chart showing degradation over cycles |
| **Charging Advisor** | Personalized charging recommendations based on driving habits |
| **Cost Estimator** | Financial impact analysis of battery degradation |
| **PDF Report Upload** | Auto-extracts battery data from service report PDFs |
| **AI Chatbot** | Context-aware battery health assistant (rule-based) |
| **Sample Reports** | Pre-generated demo PDFs for testing the upload feature |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** installed
- **pip** package manager

### Installation Steps

```bash
# 1. Navigate to the project folder
cd "digital batery"

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Load and process battery data
#    (Place NASA .mat files in /raw_data folder first, or it uses synthetic fallback)
python load_data.py

# 4. Train the ML models
python train_model.py

# 5. Generate sample PDF reports for demo
python generate_sample_reports.py

# 6. Start the web server
python app.py

# 7. Open in your browser
#    http://localhost:5000
```

---

## 📁 Project Structure

```
digital batery/
├── app.py                      # Flask backend — all API routes
├── load_data.py                # NASA .mat file loader + synthetic fallback
├── train_model.py              # ML model training (SoH + RUL)
├── generate_sample_reports.py  # Creates demo PDF service reports
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── templates/
│   └── index.html              # Single-page dashboard (5 tabs)
│
├── static/
│   ├── css/
│   │   └── style.css           # Complete design system
│   └── js/
│       └── main.js             # Frontend logic
│
├── models/                     # (Generated after training)
│   ├── soh_model.pkl           # Trained SoH prediction model
│   ├── rul_model.pkl           # Trained RUL prediction model
│   └── metrics.json            # Model evaluation metrics
│
├── data/                       # (Generated after load_data.py)
│   └── processed_battery_data.csv
│
├── raw_data/                   # (Optional) NASA .mat files go here
│   ├── B0005.mat
│   ├── B0006.mat
│   ├── B0007.mat
│   └── B0018.mat
│
└── sample_reports/             # (Generated after generate_sample_reports.py)
    ├── battery_report_healthy.pdf
    ├── battery_report_moderate.pdf
    └── battery_report_degraded.pdf
```

---

## 🔬 How It Works

### Data Pipeline
1. **load_data.py** reads NASA Li-ion Battery Dataset (.mat files) or generates realistic synthetic data
2. Extracts per-cycle: voltage, current, temperature, capacity
3. Calculates **SoH = (current_capacity / initial_capacity) × 100**
4. Saves as `processed_battery_data.csv`

### ML Models
- **SoH Model**: RandomForestRegressor  
  - Input: `[voltage, current, temperature, cycle_number]`  
  - Output: `SoH (%)`

- **RUL Model**: RandomForestRegressor  
  - Input: `[voltage, current, temperature, cycle_number, SoH]`  
  - Output: `RUL (cycles remaining until SoH ≤ 80%)`

### API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Main dashboard page |
| POST | `/predict` | SoH & RUL prediction |
| GET | `/battery_history/<id>` | Historical SoH trend data |
| POST | `/advisor` | Charging recommendations |
| POST | `/cost` | Cost impact calculation |
| POST | `/chatbot` | Rule-based chatbot |
| POST | `/upload_report` | PDF report parsing |
| GET | `/sample_reports/<file>` | Download sample PDFs |
| GET | `/metrics` | Model accuracy metrics |

---

## 🎓 For Viva/Presentation

### Key Concepts to Explain
- **Digital Twin**: A virtual replica of the physical battery, updated with real-time sensor data
- **SoH (State of Health)**: Percentage of remaining capacity vs. original capacity
- **RUL (Remaining Useful Life)**: Estimated cycles/years before end-of-life (SoH ≤ 80%)
- **Random Forest**: Ensemble ML algorithm using multiple decision trees for robust predictions
- **NASA Dataset**: Real Li-ion battery cycle-aging data from NASA Ames Prognostics Center

### Proposed Future Extensions
- **YOLO**: Visible defect detection (cracks, swelling)
- **U-Net**: Damage region pixel-level segmentation
- **CNN**: Internal defect classification from X-ray images

---

## 📊 Dataset

**NASA Li-ion Battery Aging Dataset** (Kaggle)  
Contains charge/discharge cycle data from batteries B0005, B0006, B0007, B0018 tested under controlled conditions until end-of-life.

If `.mat` files are not provided, the system automatically generates **synthetic data** with realistic degradation curves, so the app always works.

---

## 📝 License

This project is developed for educational/academic purposes as a college major project.
