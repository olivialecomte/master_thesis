import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

"""
This script concatenates CSV files for each participant across multiple segments.
First, rename files that need to be concatenated to a common prefix (here, using participant ID). Pass those IDs in the `participants` list.
The files are concatented by type (e.g., annotated_blinks, annotated_fixations, etc.) and saved in the first segment's folder.
"""

# --- IMPORTS ---
import os

import pandas as pd

from utils.paths import CLEANED_DIR

# --- CONFIGURATION ---
root_folder = CLEANED_DIR
participants = ["1", "311", "600"]

# File types to process
file_types = [
    "annotated_blinks",
    "annotated_fixations",
    "annotated_gaze",
    "annotated_imu",
    "annotated_saccades",
]

# --- Loop over participants ---
for p_id in participants:
    print(f"\nProcessing participant {p_id}...")

    # Find all folders for this participant
    part_folders = [f for f in os.listdir(root_folder) if f.startswith(p_id)]
    part_folders.sort()  # ensure order

    if not part_folders:
        print(f"!! No folders found for participant {p_id}")
        continue

    # First folder to target output folder
    first_folder_name = part_folders[0]
    first_folder_path = os.path.join(root_folder, first_folder_name)

    for ftype in file_types:
        print(f"  Processing type: {ftype}")

        all_dfs = []

        # Loop over folders (segments)
        for pfolder in part_folders:
            full_folder_path = os.path.join(root_folder, pfolder)

            # Find matching file
            matching_files = [
                f
                for f in os.listdir(full_folder_path)
                if f.startswith(ftype) and f.endswith(".csv")
            ]

            if not matching_files:
                print(f"    !! No file found for type {ftype} in {pfolder}")
                continue

            # We expect 1 file per type per folder
            fpath = os.path.join(full_folder_path, matching_files[0])

            try:
                df = pd.read_csv(fpath)
                all_dfs.append(df)
                print(f"    + Read {fpath}, shape={df.shape}")
            except Exception as e:
                print(f"    !! Error reading {fpath}: {e}")

        # --- After looping through segments ---
        if all_dfs:
            concat_df = pd.concat(all_dfs, ignore_index=True)

            # Use first folder’s filename exactly
            example_file = [
                f for f in os.listdir(first_folder_path) if f.startswith(ftype)
            ][0]
            out_path = os.path.join(first_folder_path, example_file)

            # Save — overwrite
            concat_df.to_csv(out_path, index=False)
            print(f"    --> Overwritten: {out_path}, shape={concat_df.shape}")
        else:
            print(f"    !! No data to concatenate for {ftype}")

print("\nAll done!")
