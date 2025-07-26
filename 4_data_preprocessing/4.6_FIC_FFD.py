import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

"""
Title: FID and FFD Feature Extraction from Annotated Gaze Data
Author: Olivia Lecomte
Affiliation: University of Fribourg
Date: 2025-07-10
Description:
    This script processes annotated gaze CSVs to compute spatial fixation features
    including Fixation Intersection Coefficient (FIC) and Fixation Fractal Dimension (FFD).

    It loops through participant folders, extracts gaze data from filtered FN trials,
    reconstructs fixation sequences, and computes FIC and FFD per participant.

    Results are saved to a CSV for downstream modeling.

    The folder structure is expected to be:
        - final_csv/Filtered_<Experiment>_<Datetime-Hash>_<UID>_yolo11/annotated_gaze_Filtered__<Experiment>_<Datetime-Hash>_<UID>_yolo11.csv

    Dependencies:
        - pandas
        - numpy
        - shapely
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
from shapely.geometry import LineString
from tqdm import tqdm

from utils.paths import CLEANED_DIR, EXTRACTED_FEATURES_DIR

# --- CONFIGURATION ---
root_folder = CLEANED_DIR
output_csv_path = EXTRACTED_FEATURES_DIR / "fic_ffd_features.csv"


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
        group = group.sort_values("timestamp [ns]")
        fixation = {
            "start_time": group["timestamp [ns]"].min(),
            "end_time": group["timestamp [ns]"].max(),
            "x": group["gaze x [px]"].tolist(),
            "y": group["gaze y [px]"].tolist(),
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


def compute_fic(fixations):
    """
    Compute the Fixation Intersection Coefficient (FIC), measuring how often fixation
    paths cross. As described in the paper by Vajs et al. (2022).

    Args:
        fixations (List[Dict]): List of fixation dictionaries, each with 'x' and 'y'
        coordinate lists.

    Returns:
        Tuple[float, float]: Mean and standard deviation of intersection counts across
        fixations.
    """
    intersection_counts = []
    for fix in tqdm(fixations, desc="Computing FIC"):
        points = list(zip(fix["x"], fix["y"]))
        if len(points) < 3:
            intersection_counts.append(0)
            continue
        count = 0
        for i in range(len(points)):
            for j in range(i + 2, len(points) - 1):
                seg1 = LineString([points[i], points[i + 1]])
                seg2 = LineString([points[j], points[j + 1]])
                if seg1.crosses(seg2):
                    count += 1
        intersection_counts.append(count)
    return np.mean(intersection_counts), np.std(intersection_counts)


def fractal_dimension(Z):
    # https://stackoverflow.com/questions/44793221/python-fractal-box-count-fractal-dimension
    """
    Estimate the fractal dimension of a 2D binary image using box-counting.

    Args:
        Z (np.ndarray): 2D binary array (True for fixation locations, False
        elsewhere).

    Returns:
        float: Estimated fractal dimension.
    """

    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k),
            axis=1,
        )
        return len(np.where((S > 0) & (S < k * k))[0])

    Z = Z > 0
    p = min(Z.shape)
    n = 2 ** np.floor(np.log2(p))
    sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)
    counts = [boxcount(Z, size) for size in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def compute_ffd(fixations, grid_size=64):
    """
    Compute the Fixation Fractal Dimension (FFD) for each fixation event.

    The FFD quantifies spatial complexity of fixations based on their
    distribution in a discretized 2D grid.

    Args:
        fixations (List[Dict]): List of fixation dictionaries, each with 'x' and 'y'
            coordinate lists.
        grid_size (int): Resolution of the 2D grid used for fractal dimension estimation.

    Returns:
        float: Mean fractal dimension across all fixations.
    """
    fd_list = []
    for fix in tqdm(fixations, desc="Computing FFD"):
        if len(fix["x"]) < 5:
            fd_list.append(0)
            continue
        x = np.array(fix["x"])
        y = np.array(fix["y"])
        x_norm = ((x - x.min()) / (x.max() - x.min() + 1e-5) * (grid_size - 1)).astype(
            int
        )
        y_norm = ((y - y.min()) / (y.max() - y.min() + 1e-5) * (grid_size - 1)).astype(
            int
        )
        grid = np.zeros((grid_size, grid_size), dtype=bool)
        grid[y_norm, x_norm] = 1
        fd = fractal_dimension(grid)
        fd_list.append(fd)
    return np.mean(fd_list)


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

        fixations = build_fixations_from_gaze(df)
        fic, fic_std = compute_fic(fixations)
        ffd = compute_ffd(fixations)

        print(f"  ✓ FIC: {fic:.2f} ± {fic_std:.2f}, FFD: {ffd:.2f}")

        results.append(
            {
                "filename": gaze_file,
                "folder": folder,
                "HASH": hash_str,
                "FIC_mean": fic,
                "FIC_std": fic_std,
                "FFD": ffd,
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
