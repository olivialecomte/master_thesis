import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

"""
Title: Object Tracking ID Assignment and CSV Cleanup
Author: Niloufar Chamani, edited by Olivia Lecomte
Affiliation: University of Fribourg
Date: 2025-05-21
Description:
    This script processes YOLOv11 object detection outputs from video frames.
    It loops through all CSV files in a folder, assigns consistent object IDs across frames
    by tracking spatial proximity, removes unreliable short-lived detections,
    and clears unassigned detections.

Usage:
    Run directly as a script.

Dependencies:
    - pandas
    - os

Notes:
    - Assumes CSV format where each frame has up to 8 objects.
    - Uses a proximity threshold of 40px for both X and Y axes.
"""
# --- IMPORTS ---
import os

import pandas as pd

from utils.paths import INFERENCE_CSV_DIR

# --- CONFIGURATION ---
csv_path = INFERENCE_CSV_DIR
max_objects = 8
x_threshold = 40.0
y_threshold = 40.0

# Column groups
id_columns = [f"Object_ID_{i}" for i in range(max_objects)]
class_columns = [f"Class_{i}" for i in range(max_objects)]
x_columns = [f"Center_X_{i}" for i in range(max_objects)]
y_columns = [f"Center_Y_{i}" for i in range(max_objects)]
width_columns = [f"Width_{i}" for i in range(max_objects)]
height_columns = [f"Height_{i}" for i in range(max_objects)]
conf_columns = [f"Confidence_{i}" for i in range(max_objects)]

# --- MAIN LOOP ---
for object_file in os.listdir(csv_path):
    if not object_file.endswith(".csv"):
        continue

    file_path = os.path.join(csv_path, object_file)
    print(f"Processing: {object_file}")
    df = pd.read_csv(file_path, low_memory=False)

    # Assign IDs backwards
    for index in range(len(df)):
        for i in range(max_objects):
            if not pd.isna(df.at[index, id_columns[i]]):
                current_id = df.at[index, id_columns[i]]
                current_position = (
                    df.at[index, x_columns[i]],
                    df.at[index, y_columns[i]],
                )

                lookback_index = index - 1
                while lookback_index >= 0:
                    found_any_related = False
                    for j in range(max_objects):
                        if pd.isna(df.at[lookback_index, id_columns[j]]):
                            prev_position = (
                                df.at[lookback_index, x_columns[j]],
                                df.at[lookback_index, y_columns[j]],
                            )
                            x_diff = abs(current_position[0] - prev_position[0])
                            y_diff = abs(current_position[1] - prev_position[1])

                            if x_diff < x_threshold and y_diff < y_threshold:
                                df.at[lookback_index, id_columns[j]] = current_id
                                current_position = prev_position
                                found_any_related = True

                    if not found_any_related:
                        break
                    lookback_index -= 1

    # Remove objects that exist in fewer than 5 timestamps
    object_counts = {col: df[col].value_counts() for col in id_columns}
    for i, col in enumerate(id_columns):
        ids_to_remove = object_counts[col][object_counts[col] < 5].index
        for obj_id in ids_to_remove:
            indices_to_clear = df[df[col] == obj_id].index
            for index in indices_to_clear:
                df.at[index, id_columns[i]] = None
                df.at[index, class_columns[i]] = None
                df.at[index, x_columns[i]] = None
                df.at[index, y_columns[i]] = None
                df.at[index, width_columns[i]] = None
                df.at[index, height_columns[i]] = None
                df.at[index, conf_columns[i]] = None

    # Clear data for unassigned objects
    for index in range(len(df)):
        for i in range(max_objects):
            if pd.isna(df.at[index, id_columns[i]]):
                df.at[index, class_columns[i]] = None
                df.at[index, x_columns[i]] = None
                df.at[index, y_columns[i]] = None
                df.at[index, width_columns[i]] = None
                df.at[index, height_columns[i]] = None
                df.at[index, conf_columns[i]] = None

    # --- SAVE CSV ---
    df.to_csv(file_path, index=False)
    print(f"✅ Finished: {object_file}")
    print("--------------------------------------------------")

print("All CSV files processed.")
