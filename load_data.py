"""
=============================================================================
load_data.py — NASA Li-ion Battery Dataset Loader & Preprocessor
=============================================================================
This script:
  1. Recursively scans the /raw_data folder (all subfolders, any depth)
     to find every .mat file automatically — no hardcoded paths needed
  2. Extracts charge/discharge cycle data (voltage, current, temperature, capacity)
  3. Calculates State of Health (SoH) per cycle
  4. Falls back to realistic synthetic data if no .mat files are found
  5. Saves everything as a clean CSV: data/processed_battery_data.csv

HOW TO USE:
  - Drop any NASA battery dataset (extracted zip, any folder structure) into /raw_data
  - Run:  python load_data.py
  - It will automatically find and process every .mat file, no matter how
    deeply nested. If none are found, synthetic data is generated instead.

SoH Formula:
  SoH = (current_cycle_capacity / initial_capacity) * 100

Usage:
  python load_data.py
=============================================================================
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Use the directory where this script lives as the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "processed_battery_data.csv")


# ---------------------------------------------------------------------------
# Step 1: Recursively discover all .mat files
# ---------------------------------------------------------------------------
def discover_mat_files(root_dir):
    """
    Recursively walk the entire root_dir tree and return a list of
    absolute paths to every file ending in .mat.

    Uses os.walk() so it works regardless of how deeply nested the files
    are inside subfolders. For example, all of these are found:
      raw_data/B0005.mat
      raw_data/NASA_dataset/B0005.mat
      raw_data/extracted_zip/folder1/folder2/B0018.mat

    Args:
        root_dir: Absolute path to the top-level search directory.

    Returns:
        list[str]: Sorted list of absolute paths to .mat files found.
    """
    mat_files = []

    if not os.path.isdir(root_dir):
        # raw_data folder doesn't exist yet — will be created later
        return mat_files

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".mat"):
                full_path = os.path.join(dirpath, filename)
                mat_files.append(full_path)

    # Sort for consistent ordering across runs
    mat_files.sort()
    return mat_files


# ---------------------------------------------------------------------------
# Step 2: Extract battery ID from a .mat file
# ---------------------------------------------------------------------------
def extract_battery_id(mat_data, filepath):
    """
    Figure out the battery ID key inside a loaded .mat dict.

    The NASA dataset uses keys like 'B0005', 'B0006', etc. for the main
    struct.  scipy.io.loadmat also adds metadata keys ('__header__',
    '__version__', '__globals__') which we skip.

    Strategy:
      1. Try the filename stem first (e.g. B0005.mat -> 'B0005').
      2. If that key isn't in the data, search for any key that looks
         like a battery identifier (starts with 'B' followed by digits).
      3. Fall back to the first non-metadata key.

    Args:
        mat_data:  Dict returned by scipy.io.loadmat().
        filepath:  Path to the .mat file (used to derive the filename).

    Returns:
        str or None: The battery ID key, or None if nothing usable found.
    """
    # Keys added by loadmat that are not battery data
    META_KEYS = {"__header__", "__version__", "__globals__"}
    data_keys = [k for k in mat_data.keys() if k not in META_KEYS]

    if not data_keys:
        return None

    # Strategy 1 — match by filename stem
    stem = os.path.splitext(os.path.basename(filepath))[0]
    if stem in data_keys:
        return stem

    # Strategy 2 — look for a key starting with 'B' + digits (e.g. B0005)
    for key in data_keys:
        if len(key) >= 2 and key[0] == "B" and key[1:].isdigit():
            return key

    # Strategy 3 — just use the first non-meta key
    return data_keys[0]


# ---------------------------------------------------------------------------
# Step 3: Process a single .mat file into rows
# ---------------------------------------------------------------------------
def process_single_mat(filepath, loadmat_fn):
    """
    Load one .mat file and extract per-discharge-cycle summary rows.

    The NASA .mat structure is typically:
        mat_data[battery_id]  ->  struct with field 'cycle'
        cycle[i]              ->  struct with 'type' and 'data'
        type                  ->  'charge' or 'discharge'
        data                  ->  Voltage_measured, Current_measured,
                                  Temperature_measured, (Capacity)

    This function is wrapped in broad exception handling so a single
    corrupted file never crashes the entire pipeline.

    Args:
        filepath:    Absolute path to the .mat file.
        loadmat_fn:  Reference to scipy.io.loadmat (passed in to avoid
                     repeated imports).

    Returns:
        (battery_id, list[dict]) — the ID string and a list of row dicts,
        or (None, []) on failure.
    """
    rows = []

    try:
        mat_data = loadmat_fn(filepath)
    except Exception as e:
        print(f"  [WARNING] Could not read {filepath}: {e}")
        return None, []

    battery_id = extract_battery_id(mat_data, filepath)
    if battery_id is None:
        print(f"  [WARNING] No usable battery key in {filepath} -- skipping.")
        return None, []

    # -----------------------------------------------------------------
    # Navigate the nested MATLAB struct
    # -----------------------------------------------------------------
    try:
        battery_struct = mat_data[battery_id]
        # The 'cycle' field may be nested in different ways depending on
        # how the .mat file was saved.  Try the most common layout first.
        try:
            cycles = battery_struct["cycle"][0, 0][0]
        except (IndexError, KeyError):
            cycles = battery_struct["cycle"][0][0][0]
    except (KeyError, IndexError, TypeError) as e:
        print(f"  [WARNING] Unexpected structure in {filepath} "
              f"(key='{battery_id}'): {e} -- skipping.")
        return battery_id, []

    initial_capacity = None
    cycle_counter = 0

    for cycle in cycles:
        try:
            # We only care about discharge cycles for capacity tracking
            cycle_type = str(cycle["type"][0]).lower()
            if "discharge" not in cycle_type:
                continue

            cycle_counter += 1
            data = cycle["data"][0, 0]

            # Extract measured arrays and take the mean per cycle
            voltage = float(np.mean(data["Voltage_measured"].flatten()))
            current = float(np.mean(np.abs(data["Current_measured"].flatten())))
            temperature = float(np.mean(data["Temperature_measured"].flatten()))

            # Capacity may be stored at the data level or the cycle level
            capacity = None
            for source in (data, cycle):
                try:
                    cap_val = float(source["Capacity"].flatten()[0])
                    if cap_val > 0:
                        capacity = cap_val
                        break
                except (KeyError, IndexError, TypeError, ValueError):
                    continue

            if capacity is None or capacity <= 0:
                continue

            # Track initial capacity for SoH calculation
            if initial_capacity is None:
                initial_capacity = capacity

            # SoH = (current capacity / initial capacity) * 100
            soh = (capacity / initial_capacity) * 100.0

            rows.append({
                "battery_id": battery_id,
                "cycle_number": cycle_counter,
                "voltage": round(voltage, 4),
                "current": round(current, 4),
                "temperature": round(temperature, 2),
                "capacity": round(capacity, 4),
                "SoH": round(soh, 2),
            })

        except Exception:
            # Skip any individual malformed cycle without crashing
            continue

    return battery_id, rows


# ---------------------------------------------------------------------------
# Step 4: Master loader — discover, load, process
# ---------------------------------------------------------------------------
def load_mat_files():
    """
    Robustly discover and load all .mat files under /raw_data.

    Workflow:
      1. Recursively scan /raw_data for *.mat files using os.walk().
      2. Attempt to load each one; skip any that are corrupted/unreadable.
      3. Return a combined DataFrame, or None if nothing was loaded.

    This function is designed to NEVER crash.  Any individual file error
    is caught, logged, and skipped.

    Returns:
        pd.DataFrame or None
    """
    # Check if scipy is available
    try:
        from scipy.io import loadmat
    except ImportError:
        print("[WARNING] scipy is not installed. Cannot load .mat files.")
        print("          Install with: pip install scipy")
        return None

    # --- Discover ---
    mat_paths = discover_mat_files(RAW_DATA_DIR)

    if not mat_paths:
        # No .mat files found anywhere in the tree
        return None

    print(f"[SCAN] Found {len(mat_paths)} .mat file(s) under {RAW_DATA_DIR}:")
    for p in mat_paths:
        # Print path relative to RAW_DATA_DIR for readability
        rel = os.path.relpath(p, RAW_DATA_DIR)
        print(f"       - {rel}")

    # --- Load & Process ---
    all_rows = []
    loaded_ids = []     # Battery IDs that yielded at least one row
    skipped_files = []  # Files that failed to load

    for filepath in mat_paths:
        rel = os.path.relpath(filepath, RAW_DATA_DIR)
        print(f"\n[LOADING] {rel} ...")

        battery_id, rows = process_single_mat(filepath, loadmat)

        if rows:
            all_rows.extend(rows)
            loaded_ids.append(battery_id)
            print(f"  [OK] Extracted {len(rows)} discharge cycles "
                  f"(battery: {battery_id})")
        else:
            skipped_files.append(rel)
            reason = "no usable data" if battery_id else "unreadable"
            print(f"  [SKIPPED] {rel} -- {reason}")

    # --- Summary ---
    if loaded_ids:
        unique_ids = sorted(set(loaded_ids))
        print(f"\nFound and processed {len(unique_ids)} battery file(s): "
              f"{', '.join(unique_ids)}")
    if skipped_files:
        print(f"[WARNING] Skipped {len(skipped_files)} file(s): "
              f"{', '.join(skipped_files)}")

    if not all_rows:
        return None

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Synthetic data fallback
# ---------------------------------------------------------------------------
def generate_synthetic_data():
    """
    Generate realistic synthetic Li-ion battery degradation data.

    This ensures the app always has data to work with, even without the
    NASA .mat files. The synthetic data mimics real degradation patterns:
      - Capacity starts near rated and degrades with a slight knee effect
      - Voltage, current, temperature vary realistically per cycle
      - Each battery has slightly different degradation rates

    Returns:
        pd.DataFrame: Synthetic battery data with same columns as real data.
    """
    print("[INFO] Generating realistic synthetic battery data ...")
    np.random.seed(42)  # Reproducibility

    all_rows = []

    # Each battery has a different initial capacity and degradation rate
    battery_configs = {
        "B0005": {"init_cap": 1.86, "deg_rate": 0.0018, "total_cycles": 168},
        "B0006": {"init_cap": 1.87, "deg_rate": 0.0020, "total_cycles": 168},
        "B0007": {"init_cap": 1.85, "deg_rate": 0.0022, "total_cycles": 168},
        "B0018": {"init_cap": 1.85, "deg_rate": 0.0019, "total_cycles": 132},
    }

    for battery_id, cfg in battery_configs.items():
        init_cap = cfg["init_cap"]
        deg_rate = cfg["deg_rate"]
        total_cycles = cfg["total_cycles"]

        for cycle in range(1, total_cycles + 1):
            # ---------------------------------------------------------------
            # Simulate capacity fade with a mild "knee" effect
            # Capacity = init_cap * (1 - deg_rate * cycle - knee_effect)
            # The "knee" accelerates degradation after ~70% of life
            # ---------------------------------------------------------------
            linear_fade = deg_rate * cycle
            knee_factor = 0.00001 * (cycle ** 1.5) if cycle > total_cycles * 0.7 else 0
            capacity = init_cap * (1 - linear_fade - knee_factor)
            capacity = max(capacity, init_cap * 0.5)  # Floor at 50% of original

            soh = (capacity / init_cap) * 100.0

            # Simulate realistic sensor readings with noise
            voltage = 3.5 + 0.3 * np.random.randn() + 0.2 * (soh / 100)
            voltage = np.clip(voltage, 2.5, 4.2)

            current = 1.0 + 0.3 * np.random.randn()
            current = np.clip(current, 0.5, 2.5)

            temperature = 25.0 + 5.0 * np.random.randn() + 0.05 * cycle
            temperature = np.clip(temperature, 20.0, 50.0)

            all_rows.append({
                "battery_id": battery_id,
                "cycle_number": cycle,
                "voltage": round(voltage, 4),
                "current": round(current, 4),
                "temperature": round(temperature, 2),
                "capacity": round(capacity, 4),
                "SoH": round(soh, 2),
            })

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    """Main entry point: load real data or fall back to synthetic."""
    print("=" * 60)
    print("  EV Battery Digital Twin -- Data Loader")
    print("=" * 60)

    # Ensure directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # ── Try loading real NASA .mat files ──
    print(f"\n[SCAN] Searching for .mat files in: {RAW_DATA_DIR}")
    print(f"       (including ALL subfolders, recursively)\n")

    df = load_mat_files()

    if df is not None and len(df) > 0:
        batteries = sorted(df["battery_id"].unique().tolist())
        print(f"\n[SUCCESS] Loaded {len(df)} rows from {len(batteries)} "
              f"real NASA battery file(s).")
    else:
        # ── Fallback to synthetic data ──
        print("\nNo .mat files found -- using synthetic data instead\n")
        df = generate_synthetic_data()
        batteries = sorted(df["battery_id"].unique().tolist())
        print(f"[SUCCESS] Generated {len(df)} rows of synthetic data.")

    # ── Save to CSV ──
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[SAVED] {OUTPUT_FILE}")

    # ── Final summary ──
    print("\n" + "-" * 60)
    print("  DATA SUMMARY")
    print("-" * 60)
    print(f"  Batteries : {batteries}")
    print(f"  Total rows: {len(df)}")
    print(f"  Columns   : {df.columns.tolist()}")
    print(f"  SoH range : {df['SoH'].min():.1f}% - {df['SoH'].max():.1f}%")
    print(f"  Cycles    : {df['cycle_number'].min()} - {df['cycle_number'].max()}")
    print("-" * 60)
    print(f"\nFound and processed {len(batteries)} battery files: "
          f"{', '.join(batteries)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
