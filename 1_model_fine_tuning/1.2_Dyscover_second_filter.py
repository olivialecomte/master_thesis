import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

"""
Title: Filter Short-Lived Object Detections from YOLOv11 CSV Outputs
Author: Niloufar Chamani, edited by Olivia Lecomte
Affiliation: University of Fribourg
Date: 2025-05-21
Description:
    This script loops through all YOLOv11 tracking CSV files in a directory.
    It removes objects (by ID) that appear in fewer than 5 timestamps and
    saves the cleaned data to new files prefixed with "Filtered_".

Usage:
    Run directly as a script.

Dependencies:
    - pandas
    - os

Notes:
    - Assumes each frame contains up to 8 object slots.
    - Keeps only objects with sufficient temporal consistency.
"""

# --- IMPORTS ---
import os

import pandas as pd

from utils.paths import FILTERED_DIR, INFERENCE_CSV_DIR

# --- CONFIGURATION ---
csv_path = INFERENCE_CSV_DIR
output_path = FILTERED_DIR
num_objects = 8

# Column templates
id_columns = [f"Object_ID_{i}" for i in range(num_objects)]
class_columns = [f"Class_{i}" for i in range(num_objects)]
x_columns = [f"Center_X_{i}" for i in range(num_objects)]
y_columns = [f"Center_Y_{i}" for i in range(num_objects)]
width_columns = [f"Width_{i}" for i in range(num_objects)]
height_columns = [f"Height_{i}" for i in range(num_objects)]
conf_columns = [f"Confidence_{i}" for i in range(num_objects)]

# --- MAIN LOOP ---
for object_file in os.listdir(csv_path):
    if not object_file.endswith(".csv"):
        continue

    file_path = os.path.join(csv_path, object_file)
    print(f"Processing: {object_file}")
    df = pd.read_csv(file_path, low_memory=False)

    # Count ID appearances in each ID column
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

    # --- SAVE CSV ---
    output_file = os.path.join(output_path, "Filtered_" + object_file)
    df.to_csv(output_file, index=False)
    print(f"✅ Saved filtered file: {output_file}")
    print("--------------------------------------------------")

print("All CSV files processed and filtered.")
