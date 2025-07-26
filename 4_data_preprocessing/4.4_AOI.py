import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

"""
Title: Fixation Distance and AOI Coverage Feature Extraction from Raw Gaze Data
Author: Olivia Lecomte
Affiliation: University of Fribourg
Date: 2025-07-10
Description:
    This script loops over raw gaze CSV files in participant folders to
    compute fixation-based spatial features.
    It only processes gaze data when at least one 'Fruit' object is detected
    in the scene.
    The script extracts fixation distances to the nearest Area of Interest
    (AOI) object and calculates the percentage of fixations outside AOI
    bounding boxes.
    Specifically, it calculates:
        - Mean and median distance to the center of the nearest AOI object
        - Mean and median distance to the nearest object of class 'Fruit'
        - Percentage of fixations outside AOI bounding boxes

    The folder structure is expected to be:
        - final_csv/Filtered_<Experiment>_<Datetime-Hash>/<recording folder>/gaze.csv

    Each gaze CSV must include object detection fields (e.g., Class_0 to
    Class_7, Center_X_0 to Center_X_7, etc.).

Dependencies:
    - pandas
    - numpy
    - os
    - re
    - tqdm

Usage:
    Run directly as a script.
"""

"""
Title: Fixation Distance and AOI Coverage Extraction
Author: Olivia Lecomte
Description:
    Loops through gaze CSVs in final_csv folders and extracts:
        - Fixation distances to nearest AOI
        - Distances to nearest "Fruit"
        - % fixations outside AOI bounding boxes
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
output_csv_path = EXTRACTED_FEATURES_DIR / "fixation_distance_aoi.csv"
relevant_classes = ["Fruit", "Bomb"]

# Auto-discover participants (exclude ERROR_)
all_folders = os.listdir(root_folder)
participants = sorted(
    set(f.split("_")[0] for f in all_folders if not f.startswith("ERROR_"))
)


# --- HELPER FUNCTIONS ---
def build_fixations_from_gaze(df):
    """
    Convert raw gaze data into a list of fixation events.

    Each fixation contains start and end timestamps, and a list of x and y gaze
    positions.

    Args:
        df (pd.DataFrame): Gaze data with columns including 'fixation id',
            'timestamp [ns]', 'gaze x [px]', and 'gaze y [px]'.

    Returns:
        List[Dict]: A list of fixation dictionaries with keys 'start_time',
            'end_time', 'x', and 'y'.
    """
    fixation_data = []
    grouped = df.dropna(subset=["fixation id"]).groupby("fixation id")
    for fix_id, group in grouped:
        fixation = {
            "x": group["gaze x [px]"].tolist(),
            "y": group["gaze y [px]"].tolist(),
            "indices": group.index.tolist(),
        }
        fixation_data.append(fixation)
    return fixation_data


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

    return df


def compute_nearest_item(df, relevant_classes, prefix):
    """
    Compute the distance from gaze to the nearest relevant object for each row.

    For each potential object slot (0–7), if the object class is in `relevant_classes`,
    calculates the Euclidean distance from the gaze point to the object's center,
    identifies whether the gaze falls within the object's bounding box,
    and stores the nearest object's ID, class, and bounding box status.

    Adds the following columns to the dataframe:
        - f"gaze_to_nearest_{prefix}"
        - f"nearest_{prefix}_id"
        - f"nearest_{prefix}_class"
        - f"gaze_inside_{prefix}_bbox"

    Args:
        df (pd.DataFrame): Eye-tracking dataframe with gaze coordinates and object metadata.
        relevant_classes (list[str]): List of object class names considered relevant.
        prefix (str): Prefix used in the new column names (e.g., 'fruit' or 'target').

    Returns:
        pd.DataFrame: The input dataframe with new columns indicating the nearest object's properties.
    """
    distance_df = pd.DataFrame(index=df.index)
    object_id_df = pd.DataFrame(index=df.index)
    class_df = pd.DataFrame(index=df.index)
    inside_bbox_df = pd.DataFrame(index=df.index)

    for i in range(8):
        if f"Class_{i}" not in df.columns:
            continue

        mask = df[f"Class_{i}"].isin(relevant_classes)
        dx = df["gaze x [px]"] - df[f"Center_X_{i}"]
        dy = df["gaze y [px]"] - df[f"Center_Y_{i}"]
        distance = np.sqrt(dx**2 + dy**2)
        distance[~mask] = np.nan

        distance_df[f"distance_{i}"] = distance
        object_id_df[f"object_id_{i}"] = (
            df[f"Object_ID_{i}"].where(mask, np.nan).astype("float64")
        )
        class_df[f"class_{i}"] = df[f"Class_{i}"].where(mask, np.nan).astype("object")

        half_width = df[f"Width_{i}"] / 2
        half_height = df[f"Height_{i}"] / 2
        inside_x = dx.abs() <= half_width
        inside_y = dy.abs() <= half_height
        inside_bbox = (inside_x & inside_y) & mask
        inside_bbox_df[f"inside_bbox_{i}"] = inside_bbox

    min_distances = distance_df.min(axis=1)
    valid_rows = min_distances.notna()
    valid_distance_values = distance_df.loc[valid_rows].to_numpy()

    valid_mask = ~np.isnan(valid_distance_values)
    safe_distances = np.where(valid_mask, valid_distance_values, np.inf)
    min_distance_idx = safe_distances.argmin(axis=1)

    object_id_values = object_id_df.loc[valid_rows].to_numpy()
    class_values = class_df.loc[valid_rows].to_numpy()
    inside_bbox_values = inside_bbox_df.loc[valid_rows].to_numpy()

    df[f"gaze_to_nearest_{prefix}"] = min_distances
    df[f"nearest_{prefix}_id"] = pd.Series(index=df.index, dtype="float64")
    df[f"nearest_{prefix}_class"] = pd.Series(index=df.index, dtype="object")
    df[f"gaze_inside_{prefix}_bbox"] = False

    df.loc[valid_rows, f"nearest_{prefix}_id"] = object_id_values[
        np.arange(valid_rows.sum()), min_distance_idx
    ]
    df.loc[valid_rows, f"nearest_{prefix}_class"] = class_values[
        np.arange(valid_rows.sum()), min_distance_idx
    ]
    df.loc[valid_rows, f"gaze_inside_{prefix}_bbox"] = inside_bbox_values[
        np.arange(valid_rows.sum()), min_distance_idx
    ]

    return df


# --- MAIN LOOP ---
results = []
folders = [f for f in os.listdir(root_folder) if not f.startswith("ERROR_")]
print(f"🔍 Found {len(folders)} folders to process")

for folder in tqdm(folders, desc="Processing folders"):
    folder_path = os.path.join(root_folder, folder)
    if not os.path.isdir(folder_path):
        continue

    # Find gaze file inside folder
    gaze_files = [
        f
        for f in os.listdir(folder_path)
        if f.startswith("annotated_gaze_") and f.endswith(".csv")
    ]
    if not gaze_files:
        print(f"⚠️  No gaze file in folder: {folder}")
        continue

    gaze_file = gaze_files[0]  # Use first if multiple
    gaze_path = os.path.join(folder_path, gaze_file)

    # Extract simplified HASH (datetime only) from filename
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", gaze_file)
    if not match:
        print(f"⚠️  Filename format not matched: {gaze_file}")
        continue

    hash_str = match.group(1)

    try:
        df = prep_data(gaze_path)
        if df.empty:
            print(f"⚠️  No valid data in {gaze_file}")
            continue

        df = compute_nearest_item(df, relevant_classes, prefix="object")
        fixations = build_fixations_from_gaze(df)

        fix_dists = []
        fruit_dists = []
        outside_count = 0

        for fix in fixations:
            idx = fix["indices"]
            dist_vals = df.loc[idx, "gaze_to_nearest_object"].dropna()
            class_vals = df.loc[idx, "nearest_object_class"]
            inside_vals = df.loc[idx, "gaze_inside_object_bbox"]

            if not dist_vals.empty:
                fix_dists.append(dist_vals.mean())

                fruit_mask = class_vals == "Fruit"
                fruit_d = dist_vals[fruit_mask]
                if not fruit_d.empty:
                    fruit_dists.append(fruit_d.mean())

                if not inside_vals.any():
                    outside_count += 1

        total_fix = len(fixations)
        outside_ratio = outside_count / total_fix if total_fix > 0 else np.nan

        if not fix_dists:
            print(f"⚠️ No valid fixations in {gaze_file}")
            continue

        results.append(
            {
                "filename": gaze_file,
                "folder": folder,
                "HASH": hash_str,
                "mean_fix_dist_to_object": np.mean(fix_dists),
                "median_fix_dist_to_object": np.median(fix_dists),
                "mean_fix_dist_to_fruit": (
                    np.mean(fruit_dists) if fruit_dists else np.nan
                ),
                "median_fix_dist_to_fruit": (
                    np.median(fruit_dists) if fruit_dists else np.nan
                ),
                "percent_fixations_outside_bbox": (
                    100 * outside_ratio if outside_ratio is not np.nan else np.nan
                ),
            }
        )

    except Exception as e:
        print(f"❌ Error processing {gaze_file}: {e}")

# --- SAVE RESULTS ---
if results:
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv_path, index=False)
    print(
        f"\n✅ Saved output to: {output_csv_path}, total={len(df_results)} recordings"
    )
else:
    print("\n⚠️ No valid data processed.")
