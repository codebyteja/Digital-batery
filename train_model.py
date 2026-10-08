"""
=============================================================================
train_model.py — ML Model Training for SoH & RUL Prediction
=============================================================================
This script:
  1. Loads processed_battery_data.csv
  2. Engineers features including RUL (Remaining Useful Life)
  3. Trains two RandomForestRegressor models:
       - SoH Model: predicts State of Health (%)
       - RUL Model: predicts Remaining Useful Life (in cycles)
  4. Evaluates both models (R² score, RMSE)
  5. Saves trained models as .pkl files and metrics as JSON

End-of-Life (EOL) Threshold:
  A battery is considered at end-of-life when SoH drops to 80%.
  RUL = (number of cycles remaining until SoH reaches 80%)

Usage:
  python train_model.py
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "processed_battery_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
SOH_MODEL_PATH = os.path.join(MODEL_DIR, "soh_model.pkl")
RUL_MODEL_PATH = os.path.join(MODEL_DIR, "rul_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# End-of-life threshold: battery needs replacement when SoH < 80%
EOL_THRESHOLD = 80.0


def compute_rul(df):
    """
    Compute Remaining Useful Life (RUL) for each row.
    
    For each battery, RUL at cycle N = (cycle at which SoH first drops to ≤80%) - N.
    If the battery never reaches 80% in the data, we extrapolate linearly
    from the degradation rate to estimate when it would.
    
    Args:
        df: DataFrame with columns [battery_id, cycle_number, SoH]
    
    Returns:
        DataFrame with added 'RUL' column (in cycles)
    """
    df = df.copy()
    df["RUL"] = 0.0  # Initialize

    for battery_id in df["battery_id"].unique():
        mask = df["battery_id"] == battery_id
        battery_df = df[mask].sort_values("cycle_number")

        # Find the cycle where SoH first drops to or below EOL threshold
        eol_rows = battery_df[battery_df["SoH"] <= EOL_THRESHOLD]

        if len(eol_rows) > 0:
            # Battery reaches EOL in the data
            eol_cycle = eol_rows["cycle_number"].iloc[0]
        else:
            # Extrapolate: estimate when SoH would reach 80%
            # Use the degradation rate from first to last cycle
            first_soh = battery_df["SoH"].iloc[0]
            last_soh = battery_df["SoH"].iloc[-1]
            last_cycle = battery_df["cycle_number"].iloc[-1]
            first_cycle = battery_df["cycle_number"].iloc[0]

            if first_soh > last_soh:
                # Degradation rate per cycle
                rate = (first_soh - last_soh) / (last_cycle - first_cycle)
                # Estimated cycles from start until SoH = 80%
                cycles_to_eol = (first_soh - EOL_THRESHOLD) / rate
                eol_cycle = int(first_cycle + cycles_to_eol)
            else:
                # No degradation detected — set a large default
                eol_cycle = last_cycle + 500

        # RUL = eol_cycle - current_cycle (clamped to ≥ 0)
        df.loc[mask, "RUL"] = (eol_cycle - battery_df["cycle_number"]).clip(lower=0).values

    return df


def train_soh_model(df):
    """
    Train RandomForest model to predict State of Health (SoH).
    
    Features: voltage, current, temperature, cycle_number
    Target:   SoH (%)
    
    Returns:
        (trained_model, metrics_dict)
    """
    print("\n--- Training SoH Prediction Model ---")

    # Define features and target
    feature_cols = ["voltage", "current", "temperature", "cycle_number"]
    X = df[feature_cols].values
    y = df["SoH"].values

    # 80-20 train-test split with fixed random state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # RandomForestRegressor with tuned hyperparameters
    # n_estimators=150 gives good accuracy without excessive training time
    # max_depth=20 prevents overfitting while capturing degradation patterns
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )

    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"  R² Score:  {r2:.4f}")
    print(f"  RMSE:      {rmse:.4f}")

    metrics = {"r2": round(r2, 4), "rmse": round(rmse, 4)}
    return model, metrics


def train_rul_model(df):
    """
    Train RandomForest model to predict Remaining Useful Life (RUL).
    
    Features: voltage, current, temperature, cycle_number, SoH
    Target:   RUL (remaining cycles until SoH ≤ 80%)
    
    Returns:
        (trained_model, metrics_dict)
    """
    print("\n--- Training RUL Prediction Model ---")

    # RUL model uses SoH as an additional feature (it's a strong predictor)
    feature_cols = ["voltage", "current", "temperature", "cycle_number", "SoH"]
    X = df[feature_cols].values
    y = df["RUL"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"  R² Score:  {r2:.4f}")
    print(f"  RMSE:      {rmse:.4f}")

    metrics = {"r2": round(r2, 4), "rmse": round(rmse, 4)}
    return model, metrics


def main():
    """Main entry point: load data, engineer features, train, save."""
    print("=" * 60)
    print("  EV Battery Digital Twin — Model Training")
    print("=" * 60)

    # Ensure model output directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Step 1: Load processed data
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] {DATA_FILE} not found!")
        print("  Run `python load_data.py` first to generate the dataset.")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"[INFO] Loaded {len(df)} rows from {DATA_FILE}")

    # Step 2: Compute RUL (Remaining Useful Life) for each row
    print("[INFO] Computing RUL for each cycle ...")
    df = compute_rul(df)

    # Step 3: Train SoH prediction model
    soh_model, soh_metrics = train_soh_model(df)

    # Step 4: Train RUL prediction model
    rul_model, rul_metrics = train_rul_model(df)

    # Step 5: Save trained models
    joblib.dump(soh_model, SOH_MODEL_PATH)
    print(f"\n[SAVED] SoH model -> {SOH_MODEL_PATH}")

    joblib.dump(rul_model, RUL_MODEL_PATH)
    print(f"[SAVED] RUL model -> {RUL_MODEL_PATH}")

    # Step 6: Save metrics for the About page
    all_metrics = {
        "soh_model": soh_metrics,
        "rul_model": rul_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[SAVED] Metrics -> {METRICS_PATH}")

    print("\n" + "=" * 60)
    print("  Training complete! Models ready for deployment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
