import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

"""
Title: Fruit Gaze Latency Extraction from Raw Eye-Tracking Data
Author: Olivia Lecomte
Affiliation: University of Fribourg
Date: 2025-07-11

Description:
    This script loops over annotated gaze CSVs in participant folders to compute latency-based spatial features.
    It processes gaze data when at least one 'Fruit' object is detected in the scene.

    Specifically, it calculates:
        - Mean and median latency to first fixation on each unique 'Fruit'
        - Standard deviation, min, and max latency
        - Percentage of fruits never fixated (% unseen)
        - Total number of unique fruits and how many were fixated

    The output is one row per participant recording, with folder name and HASH as metadata.

Expected Folder Structure:
    - final_csv/
        └── Filtered_<Experiment>_<Datetime-Hash>/
            └── annotated_gaze_<...>.csv

Dependencies:
    - pandas
    - numpy
    - os
    - re
    - tqdm

Usage:
    Run directly as a script.
"""

# --- IMPORTS ---
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.paths import CLEANED_DIR, EXTRACTED_FEATURES_DIR

# --- CONFIGURATION ---
root_folder = CLEANED_DIR
output_csv_path = EXTRACTED_FEATURES_DIR / "latency.csv"


# --- HELPER FUNCTIONS ---
def prep_data(path):
    """
    Load and filter eye-tracking data to retain relevant trials involving fruit-related objects.

    The function retains only rows where the event name starts with 'FN' and ends with
    'Start' (active Fruit Ninja game time),
    the 'Object_ID_0' is present, and any 'Class_i' column indicates a 'Fruit' object.

    Args:
        path (str): File path to the CSV file containing the raw data.

    Returns:
        pd.DataFrame: Filtered dataframe containing only relevant fruit-related events.
    """
    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.str.strip()

    # Mask to include only rows with at least one 'Fruit' object
    fruit_mask = np.zeros(len(data), dtype=bool)
    for i in range(8):
        if f"Class_{i}" in data.columns:
            fruit_mask |= data[f"Class_{i}"] == "Fruit"

    df = (
        data[
            data["event_name"].str.startswith("FN")
            & data["event_name"].str.endswith("Start")
            & data["Object_ID_0"].notna()
            & fruit_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    # Convert timestamp to ms once for performance
    df["timestamp_ms"] = df["timestamp [ns]"] / 1e6

    return df


def analyze_fruit_gaze_latency(df, max_objects=8):
    """
    Analyze gaze latency to each unique fruit object in an eye-tracking dataset.

    For each fruit object in each trial, this function computes the time delay (in milliseconds)
    between the moment the fruit first appears (based on the current row's timestamp) and
    the first moment the participant's gaze enters the fruit's bounding box.

    Only the first entry into the bounding box is counted per unique (recording id, object id) pair.

    Args:
        df (pd.DataFrame): Eye-tracking dataframe containing columns such as:
                           - 'recording id', 'timestamp_ms', 'gaze x [px]', 'gaze y [px]',
                           - 'Class_i', 'Object_ID_i', 'Center_X_i', 'Center_Y_i', 'Width_i', 'Height_i' for i in 0 to max_objects-1
        max_objects (int, optional): Maximum number of object slots per frame (default is 8).

    Returns:
        dict: A dictionary containing latency statistics:
            - 'mean_latency' (float): Mean gaze latency to fruit objects.
            - 'median_latency' (float): Median gaze latency.
            - 'std_latency' (float): Standard deviation of gaze latency.
            - 'min_latency' (float): Minimum gaze latency.
            - 'max_latency' (float): Maximum gaze latency.
            - 'percent_unseen_fruits' (float): Percentage of fruits never fixated on.
            - 'n_fruits' (int): Total number of unique fruit instances tracked.
            - 'n_seen' (int): Number of fruits with at least one gaze inside the bounding box.
    """
    latencies = []
    tracked_fruits = set()

    for row in df.itertuples(index=False):
        rec_id = row[1]  # 'recording id'
        timestamp_ms = row[-1]  # 'timestamp_ms'

        for i in range(max_objects):
            try:
                cls = getattr(row, f"Class_{i}")
                if cls != "Fruit":
                    continue

                obj_id = getattr(row, f"Object_ID_{i}")
                cx = getattr(row, f"Center_X_{i}")
                cy = getattr(row, f"Center_Y_{i}")
                w = getattr(row, f"Width_{i}")
                h = getattr(row, f"Height_{i}")
            except AttributeError:
                continue

            if (
                pd.isna(obj_id)
                or pd.isna(cx)
                or pd.isna(cy)
                or pd.isna(w)
                or pd.isna(h)
            ):
                continue

            fruit_key = (rec_id, obj_id)
            if fruit_key in tracked_fruits:
                continue
            tracked_fruits.add(fruit_key)

            # Get all rows for the same recording
            trial_rows = df[df["recording id"] == rec_id].sort_values("timestamp_ms")

            # Fruit appearance timestamp
            t_appear = timestamp_ms

            # Find how long this fruit is visible
            # i.e., rows where the same Object_ID_i is present for this rec_id
            visible_rows = trial_rows[trial_rows[f"Object_ID_{i}"] == obj_id]
            if visible_rows.empty:
                continue  # Shouldn't happen, but just in case

            t_disappear = visible_rows["timestamp_ms"].max()

            # Limit to time window when fruit is visible
            search_window = trial_rows[
                (trial_rows["timestamp_ms"] >= t_appear)
                & (trial_rows["timestamp_ms"] <= t_disappear)
            ]

            # Check gaze inside bounding box during that window
            inside_bbox = search_window[
                (np.abs(search_window["gaze x [px]"] - cx) <= w / 2)
                & (np.abs(search_window["gaze y [px]"] - cy) <= h / 2)
            ]

            if not inside_bbox.empty:
                t_enter = inside_bbox["timestamp_ms"].iloc[0]
                latency = t_enter - t_appear
                latencies.append(latency)
            else:
                latencies.append(np.nan)

    latencies = np.array(latencies)
    seen_mask = ~np.isnan(latencies)
    seen_latencies = latencies[seen_mask]
    percent_unseen = (
        100 * (1 - seen_mask.sum() / len(latencies)) if len(latencies) > 0 else 100.0
    )

    return {
        "mean_latency": np.mean(seen_latencies) if seen_latencies.size > 0 else None,
        "median_latency": (
            np.median(seen_latencies) if seen_latencies.size > 0 else None
        ),
        "std_latency": np.std(seen_latencies) if seen_latencies.size > 0 else None,
        "min_latency": np.min(seen_latencies) if seen_latencies.size > 0 else None,
        "max_latency": np.max(seen_latencies) if seen_latencies.size > 0 else None,
        "percent_unseen_fruits": percent_unseen,
        "n_fruits": len(latencies),
        "n_seen": seen_mask.sum(),
    }


# --- MAIN LOOP ---
results = []
folders = [f for f in os.listdir(root_folder) if not f.startswith("ERROR_")]

print(f"🔍 Found {len(folders)} folders to process")
for folder in tqdm(folders, desc="Processing folders"):
    folder_path = os.path.join(root_folder, folder)
    if not os.path.isdir(folder_path):
        continue

    gaze_files = [
        f
        for f in os.listdir(folder_path)
        if f.startswith("annotated_gaze_") and f.endswith(".csv")
    ]
    if not gaze_files:
        print(f"⚠️  No gaze file in folder: {folder}")
        continue

    gaze_file = gaze_files[0]  # Use first match
    gaze_path = os.path.join(folder_path, gaze_file)

    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", gaze_file)
    if not match:
        print(f"⚠️  Filename format not matched: {gaze_file}")
        continue

    hash_str = match.group(1)
    participant_id = folder.split("_")[1] if "_" in folder else folder

    try:
        df = prep_data(gaze_path)
        if df.empty:
            print(f"⚠️  No valid data in {gaze_file}")
            continue

        result = analyze_fruit_gaze_latency(df)
        result["HASH"] = hash_str
        result["folder"] = folder
        result["participant"] = participant_id
        results.append(result)

    except Exception as e:
        print(f"❌ Error processing {gaze_file}: {e}")

# --- SAVE RESULTS ---
if results:
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv_path, index=False)
    print(
        f"\n✅ Saved output to: {output_csv_path}, total={len(df_results)} participant recordings"
    )
else:
    print("\n⚠️ No valid data processed.")
    print("\n⚠️ No valid data processed.")
    print("\n⚠️ No valid data processed.")
    print("\n⚠️ No valid data processed.")
    print("\n⚠️ No valid data processed.")
    print("\n⚠️ No valid data processed.")
    print("\n⚠️ No valid data processed.")
    print("\n⚠️ No valid data processed.")
