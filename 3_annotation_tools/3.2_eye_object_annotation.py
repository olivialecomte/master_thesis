import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

"""
Title: Add Object Positions to Annotated Eye-Tracking CSVs
Author: Niloufar Chamani, edited by Olivia Lecomte
Affiliation: University of Fribourg
Date: 2025-06-19
Description:
    This script adds object positions (from filtered CSV) to annotated eye-tracking datasets,
    matching by HASH in the filenames.

    It loops over all annotated CSVs:
        - annotated_gaze_*.csv
        - annotated_imu_*.csv
        - annotated_saccades_*.csv
        - annotated_fixations_*.csv
        - annotated_blinks_*.csv

    It finds the matching filtered CSV:
        - annotated_filtered_*.csv

    Then:
        - adds object positions
        - saves to output folder: final_csv/<HASH>/

    The folder structure is expected to be:
        - annotated_csv/
            - annotated_gaze_*.csv
            - annotated_imu_*.csv
            - annotated_saccades_*.csv
            - annotated_fixations_*.csv
            - annotated_blinks_*.csv
            - annotated_filtered_*.csv

    Output:
        - final_csv/<HASH>/annotated_gaze_<HASH>.csv
        - final_csv/<HASH>/annotated_imu_<HASH>.csv
        ...

Usage:
    Run directly as a script.

Dependencies:
    - pandas
    - numpy
    - os
    - re
"""

# --- IMPORTS ---
import os
import re

import numpy as np
import pandas as pd
import torch

from utils.paths import ANNOTATED_DIR, CLEANED_DIR

# --- CONFIGURATION ---
base_path = ANNOTATED_DIR
output_base = CLEANED_DIR


# --- HELPER FUNCTIONS ---
def add_object_positions_torch_batch(
    annotated_data,
    object_positions,
    annotated_timestamp_field,
    object_timestamp_field,
    threshold_ns,
    device="cuda",
    batch_size=10000,
):
    # Convert timestamps to PyTorch tensors
    annotated_timestamps = torch.tensor(
        annotated_data[annotated_timestamp_field].values, device=device
    )  # (N,)
    object_timestamps = torch.tensor(
        object_positions[object_timestamp_field].values, device=device
    )  # (M,)

    N = len(annotated_timestamps)
    M = len(object_timestamps)

    # Prepare output
    nearest_indices_all = []
    valid_mask_all = []

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)

        batch_timestamps = annotated_timestamps[start_idx:end_idx].view(-1, 1)  # (B, 1)
        abs_diff = torch.abs(batch_timestamps - object_timestamps.view(1, -1))  # (B, M)

        nearest_indices = torch.argmin(abs_diff, dim=1)  # (B,)
        nearest_deltas = abs_diff[
            torch.arange(len(batch_timestamps), device=device), nearest_indices
        ]
        valid_mask = nearest_deltas <= threshold_ns

        nearest_indices_all.append(nearest_indices.cpu())
        valid_mask_all.append(valid_mask.cpu())

    # Concatenate all batches
    nearest_indices_cpu = torch.cat(nearest_indices_all).numpy()
    valid_mask_cpu = torch.cat(valid_mask_all).numpy()

    # Initialize columns
    object_columns = object_positions.columns.drop(object_timestamp_field)
    for col in object_columns:
        annotated_data[col] = (
            np.nan if object_positions[col].dtype.kind in "ifc" else None
        )

    # Fill in data
    for col in object_columns:
        annotated_data.loc[valid_mask_cpu, col] = object_positions.iloc[
            nearest_indices_cpu[valid_mask_cpu]
        ][col].values

    return annotated_data


def add_object_positions_to_all(base_path, output_base, threshold_ns=42e6):
    # Setup regex to extract HASH
    datetime_pattern = re.compile(
        r"annotated_(?:gaze|imu|saccades|fixations|blinks|filtered)_(.+)\.csv"
    )

    # Files to process: list of (prefix, timestamp_field)
    file_types = [
        ("annotated_gaze_", "timestamp [ns]"),
        ("annotated_imu_", "timestamp [ns]"),
        ("annotated_saccades_", "start timestamp [ns]"),
        ("annotated_fixations_", "start timestamp [ns]"),
        ("annotated_blinks_", "start timestamp [ns]"),
    ]

    all_files = os.listdir(base_path)

    for prefix, timestamp_column in file_types:
        # Process all matching files
        for file in all_files:
            if not file.startswith(prefix) or not file.endswith(".csv"):
                continue

            # Extract HASH
            match = datetime_pattern.match(file)
            if not match:
                print(f"❌ Skipping (no match): {file}")
                continue

            hash_part = match.group(1)

            # Find matching filtered CSV
            filtered_name = f"annotated_filtered_{hash_part}.csv"
            filtered_path = os.path.join(base_path, filtered_name)
            if not os.path.exists(filtered_path):
                print(
                    f"❌ Missing filtered CSV for: {file} → expected: {filtered_name}"
                )
                continue

            # Load both
            file_path = os.path.join(base_path, file)
            annotated_data = pd.read_csv(file_path, low_memory=False)
            object_positions = pd.read_csv(filtered_path, low_memory=False)

            print(f"✅ Processing: {file} using '{timestamp_column}'")
            # print(f"✅ Using object positions: {filtered_name}")

            # Add object positions
            updated_data = add_object_positions_torch_batch(
                annotated_data,
                object_positions,
                timestamp_column,
                "timestamp [ns]",
                threshold_ns,
                device="cuda",
                batch_size=10000,
            )

            # Create output folder for this HASH
            participant_folder = os.path.join(output_base, hash_part)
            os.makedirs(participant_folder, exist_ok=True)

            # Save to output folder
            output_path = os.path.join(participant_folder, file)
            updated_data.to_csv(output_path, index=False)

            print("--------------------------------------------------")


# --- RUN ---
if __name__ == "__main__":
    print("Saving final versions to: final_csv/<HASH>/")
    print("--------------------------------------------------")
    add_object_positions_to_all(base_path, output_base)
    print("All files processed.")
