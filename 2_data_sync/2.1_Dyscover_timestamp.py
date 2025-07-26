import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

"""
Title: Addition of World Timestamp for Filtered Eye-Tracking CSVs
Author: Niloufar Chamani, edited by Olivia Lecomte
Affiliation: University of Fribourg
Date: 2025-05-22
Description:
    This script processes filtered CSV files generated from object detection in eye-tracking videos.
    It extracts the experiment name and date-time-hash from each filename, finds the corresponding
    `world_timestamps.csv` file in a matching folder structure, and adds the timestamp column
    into the filtered CSV if the frame counts match.

    Files with errors (e.g., mismatched lengths) are renamed with a prefix "ERROR_" for further inspection.

    The folder structure is expected to be:
        - filtered_csv/Filtered_<Experiment>_<Datetime-Hash>_*.csv
        - raw_videos/<Experiment>/<Datetime-Hash>/world_timestamps.csv

Usage:
    Run directly as a script.

Dependencies:
    - pandas
    - os
    - re

Notes:
    - Assumes all `world_timestamps.csv` files align 1:1 with their corresponding filtered CSV.
    - Prints mismatch warnings when no match or row length inconsistency occurs.
"""

# --- IMPORTS ---
import os
import re

import pandas as pd

from utils.paths import FILTERED_DIR, RAW_DATA_DIR

# --- CONFIGURATION ---
filtered_csv_dir = FILTERED_DIR
raw_videos_dir = RAW_DATA_DIR
output_dir = filtered_csv_dir

# Regex to extract full datetime and hash (yyyy-mm-dd_hh-mm-ss-hhhhhhhh)
datetime_pattern = re.compile(
    r"Filtered_(.*?)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-[a-zA-Z0-9]{8})_.*\.csv"
)

# Regex to extract subject and hash (subxxx-hhhhhhhh)
subject_pattern = re.compile(r"Filtered_(.*?)_(sub\d{3}-[a-zA-Z0-9]{8})_.*\.csv")

# --- MAIN LOOP ---
for i, filtered_file in enumerate(os.listdir(filtered_csv_dir)):
    if not filtered_file.endswith(".csv"):
        continue

    match = datetime_pattern.match(filtered_file) or subject_pattern.match(
        filtered_file
    )
    if not match:
        print(f"❌ Skipping: {filtered_file} (doesn't match pattern)")
        continue

    experiment_name, date_time = match.groups()

    # Build path to the corresponding world_timestamps.csv
    timestamp_file = os.path.join(
        raw_videos_dir, experiment_name, date_time, "world_timestamps.csv"
    )

    if not os.path.exists(timestamp_file):
        print(f"❌ Missing timestamps for {filtered_file} at {timestamp_file}")
        continue

    filtered_path = os.path.join(filtered_csv_dir, filtered_file)
    dyscover = pd.read_csv(filtered_path)
    worldtimestamps = pd.read_csv(timestamp_file)

    if len(dyscover) != len(worldtimestamps):
        print(
            f"❌ Length mismatch for {filtered_file}: {len(dyscover)} vs {len(worldtimestamps)}"
        )

        # Rename the original mismatched file
        old_path = os.path.join(filtered_csv_dir, filtered_file)
        new_name = f"ERROR_{filtered_file}"
        new_path = os.path.join(filtered_csv_dir, new_name)
        os.rename(old_path, new_path)

        print(f"❌ Renamed to {new_name} due to length mismatch.")
        continue

    print(f"Processing file {i+1}")

    # Merge timestamps
    dyscover["timestamp [ns]"] = worldtimestamps["timestamp [ns]"]

    # --- SAVE CSV ---
    output_path = os.path.join(output_dir, filtered_file)
    dyscover.to_csv(output_path, index=False)
    print(f"✅ Updated and saved: {filtered_file}")
    print("--------------------------------------------------")

print("All CSV files processed.")
