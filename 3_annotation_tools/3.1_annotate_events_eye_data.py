import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

"""
Title: Annotation of Eye-Tracking CSVs with Event Labels
Author: Niloufar Chamani, edited by Olivia Lecomte
Affiliation: University of Fribourg
Date: 2025-06-19
Description:
    This script processes filtered CSV files and annotates them with event labels
    based on corresponding event data in raw_data folders.

    It extracts the experiment name and date-time-hash from each filename, finds the corresponding
    event files in the matching raw_data/<Experiment>/<Datetime-Hash> folder,
    and annotates each dataset with the appropriate event label.

    Files with errors (e.g., missing event files) are renamed with a prefix "ERROR_" for inspection.

    The folder structure is expected to be:
        - filtered_csv/Filtered_<Experiment>_<Datetime-Hash>_*.csv
        - raw_data/<Experiment>/<Datetime-Hash>/events.csv, gaze.csv, blinks.csv, etc.
        - annotated_csv/ (output goes here)

Usage:
    Run directly as a script.

Dependencies:
    - pandas
    - os
    - re
"""

# --- IMPORTS ---
import os
import re

import pandas as pd

from utils.paths import ANNOTATED_DIR, FILTERED_DIR, RAW_DATA_DIR

# --- CONFIGURATION ---
filtered_csv_dir = FILTERED_DIR
raw_data_dir = RAW_DATA_DIR
annotated_csv_dir = ANNOTATED_DIR

# Create output folder if it does not exist
os.makedirs(annotated_csv_dir, exist_ok=True)


# --- HELPER FUNCTION ---
def find_event_for_timestamp(timestamp, events_sorted):
    event_timestamps = events_sorted["timestamp [ns]"]
    idx = (
        event_timestamps.searchsorted(timestamp, side="right") - 1
    )  # last event ≤ timestamp

    if idx >= 0:
        return events_sorted.iloc[idx]["name"]
    else:
        return ""


# Regex to extract full datetime and hash (yyyy-mm-dd_hh-mm-ss-hhhhhhhh)
datetime_pattern = re.compile(
    r"Filtered_(.*?)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-[a-zA-Z0-9]{8})_.*\.csv"
)

# Regex to extract subject and hash (subxxx-hhhhhhhh)
subject_pattern = re.compile(r"Filtered_(.*?)_(sub\d{3}-[a-zA-Z0-9]{8})_.*\.csv")

# List of event files to annotate
event_files = [
    "blinks.csv",
    "events.csv",
    "fixations.csv",
    "gaze.csv",
    "imu.csv",
    "saccades.csv",
]

# --- MAIN LOOP ---
for i, filtered_file in enumerate(os.listdir(filtered_csv_dir)):
    if not filtered_file.endswith(".csv") or filtered_file.startswith("ERROR_"):
        continue

    match = datetime_pattern.match(filtered_file) or subject_pattern.match(
        filtered_file
    )
    if not match:
        print(f"❌ Skipping: {filtered_file} (doesn't match pattern)")
        continue

    experiment_name, date_time = match.groups()

    # Path to corresponding raw_data folder
    raw_folder = os.path.join(raw_data_dir, experiment_name, date_time)

    if not os.path.exists(raw_folder):
        print(f"❌ Missing raw_data folder for {filtered_file} at {raw_folder}")

        # Rename the original file
        old_path = os.path.join(filtered_csv_dir, filtered_file)
        new_name = f"ERROR_{filtered_file}"
        new_path = os.path.join(filtered_csv_dir, new_name)
        os.rename(old_path, new_path)

        print(f"❌ Renamed to {new_name} due to missing raw_data folder.")
        continue

    # ---- Process the file ----
    try:
        # Load event file
        events = pd.read_csv(os.path.join(raw_folder, "events.csv"))
        events_sorted = events.sort_values(by="timestamp [ns]").reset_index(drop=True)

        # ---- Annotate the filtered CSV itself ----
        filtered_path = os.path.join(filtered_csv_dir, filtered_file)
        filtered_df = pd.read_csv(filtered_path, low_memory=False)

        # Determine timestamp column
        if "timestamp [ns]" in filtered_df.columns:
            filtered_timestamp_column = "timestamp [ns]"
        elif "Timestamp(ms)" in filtered_df.columns:
            # Convert to ns
            filtered_df["timestamp [ns]"] = filtered_df["Timestamp(ms)"] * 1_000_000
            filtered_timestamp_column = "timestamp [ns]"
        else:
            raise ValueError("❌ Filtered CSV is missing timestamp column")

        # Annotate filtered CSV
        filtered_df["event_name"] = filtered_df[filtered_timestamp_column].apply(
            lambda x: find_event_for_timestamp(x, events_sorted)
        )

        # Save annotated filtered CSV
        annotated_name = f"annotated_filtered_{filtered_file}"
        output_path = os.path.join(annotated_csv_dir, annotated_name)
        filtered_df.to_csv(output_path, index=False)

        # ---- Annotate each event file ----
        for data_file in event_files:
            data_path = os.path.join(raw_folder, data_file)

            if not os.path.exists(data_path):
                print(f"❌ Missing {data_file} for {filtered_file}")
                raise FileNotFoundError(f"Missing {data_file}")

            data = pd.read_csv(data_path)

            # Determine timestamp column
            if "start timestamp [ns]" in data.columns:
                timestamp_column = "start timestamp [ns]"
            elif "timestamp [ns]" in data.columns:
                timestamp_column = "timestamp [ns]"
            else:
                raise ValueError(f"❌ {data_file} is missing timestamp column")

            # Annotate
            data["event_name"] = data[timestamp_column].apply(
                lambda x: find_event_for_timestamp(x, events_sorted)
            )

            # --- SAVE CSV ---
            annotated_name = (
                f"annotated_{os.path.splitext(data_file)[0]}_{filtered_file}"
            )
            output_path = os.path.join(annotated_csv_dir, annotated_name)
            data.to_csv(output_path, index=False)

        print(f"✅ Processed file {i+1}: {filtered_file}")
        print("--------------------------------------------------")

    except Exception as e:
        # Rename problematic file
        old_path = os.path.join(filtered_csv_dir, filtered_file)
        new_name = f"ERROR_{filtered_file}"
        new_path = os.path.join(filtered_csv_dir, new_name)
        os.rename(old_path, new_path)

        print(f"❌ ERROR processing {filtered_file}: {e}")
        print(f"❌ Renamed to {new_name}")
        print("--------------------------------------------------")

print("All CSV files processed.")
