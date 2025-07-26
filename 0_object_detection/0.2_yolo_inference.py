import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

"""
Title: YOLO Video Inference and Object Tracking CSV Export
Author: Niloufar Chamani, edited by Olivia Lecomte
Affiliation: University of Fribourg
Date: 2025-05-21
Description:
    This script performs object detection and tracking on videos using a trained YOLO model.
    For each video, it outputs:
        - An annotated video with bounding boxes and object IDs
        - A CSV file containing per-frame object data including ID, class, position, size, confidence, and RGB color

Usage:
    1. Set the correct paths for:
        - model_path (trained YOLOv11 weights)
        - video_folder (raw data folders including input videos)
        - output_folder (annotated videos)
        - csv_folder (inference results in CSV format)
    2. Run the script:
        python yolo_inference_script.py

Dependencies:
    - Python >= 3.8
    - OpenCV (cv2)
    - torch
    - numpy
    - ultralytics
    - csv
    - os

Notes:
    - RGB color is sampled at the center of the bounding box.
    - If a bounding box center falls out of frame bounds, RGB values are left empty.
    - Object ID assignment is handled by YOLOv11's built-in tracker.
    - Output videos are encoded in MP4v format, using original video resolution and frame rate.
    - All paths should be absolute or adjusted based on your environment.
"""
# --- IMPORTS ---
import csv
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from utils.paths import (
    INFERENCE_CSV_DIR,
    INFERENCE_VIDEOS_DIR,
    MODELS_DIR,
    RAW_DATA_DIR,
)

# --- CONFIGURATION ---
model_path = MODELS_DIR / "roboflow_2c_combined_et_10000_4300_3500/weights/best.pt"
video_folder = RAW_DATA_DIR
output_folder = INFERENCE_VIDEOS_DIR
csv_folder = INFERENCE_CSV_DIR

# Check if folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO(model_path)
model.to(device)


# --- INFERENCE FUNCTION ---
def inference(model=model, video_path=None, output_path=None, csv_path=None):
    """complete inference and CSV generation for a video

    Args:
        model: YOLO model to use for inference
        video_path (str): full path to the video file
    """
    # Video capture and output setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        raise RuntimeError("Error: Could not open video.")

    # Get original video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Original width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Original height
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
    out = cv2.VideoWriter(
        output_path, fourcc, fps, (width, height)
    )  # Use original resolution

    # CSV file setup
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = ["Timestamp(ms)"]
        max_objects = 8
        for i in range(max_objects):
            header.extend(
                [
                    f"Object_ID_{i}",
                    f"Class_{i}",
                    f"Center_X_{i}",
                    f"Center_Y_{i}",
                    f"Width_{i}",
                    f"Height_{i}",
                    f"Confidence_{i}",
                    f"R_{i}",
                    f"G_{i}",
                    f"B_{i}",
                ]
            )
        writer.writerow(header)

        previous_timestamp = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                if previous_timestamp is not None:
                    print(f"Interval: {timestamp - previous_timestamp} ms")
                previous_timestamp = timestamp

                # Object detection and tracking on GPU
                results = model.track(frame, persist=True, device=device)

                # Visualization (plot detections and return as numpy array)
                frame_ = results[0].plot()

                # CSV file
                row_data = [timestamp]
                object_count = 0

                # Iterate over detected objects
                if hasattr(results[0], "boxes") and results[0].boxes is not None:
                    for box in results[0].boxes:
                        if object_count >= max_objects:
                            break  # Limit to max_objects

                        # Extract bounding box details and confidence
                        xywh = (
                            box.xywh.cpu().numpy().flatten()
                        )  # Center X, Center Y, Width, Height
                        conf = box.conf.cpu().item()  # Confidence score
                        cls = int(box.cls.cpu().item())  # Class index
                        obj_id = (
                            int(box.id.cpu().item()) if box.id is not None else ""
                        )  # Object ID (if available)

                        # Get RGB at center of bounding box
                        center_x, center_y = int(xywh[0]), int(xywh[1])
                        if (
                            0 <= center_x < frame.shape[1]
                            and 0 <= center_y < frame.shape[0]
                        ):
                            b, g, r = frame[center_y, center_x]
                        else:
                            r, g, b = "", "", ""  # Out of bounds fallback

                        # Append object details to the row
                        row_data.extend(
                            [
                                obj_id,
                                model.names[cls],
                                xywh[0],
                                xywh[1],
                                xywh[2],
                                xywh[3],
                                conf,
                                r,
                                g,
                                b,
                            ]
                        )
                        object_count += 1

                # Fill remaining columns for undetected objects
                while len(row_data) < len(header):
                    row_data.extend(["", "", "", "", "", "", "", "", "", ""])

                # Write the ryow to the CSV
                writer.writerow(row_data)

                # Ensure frame_ is a valid numpy array for OpenCV operations
                frame_ = frame_.astype(np.uint8)  # Ensure correct data type
                frame_ = np.ascontiguousarray(frame_)  # Ensure proper memory alignment

                # Write the frame to output with the original resolution
                out.write(frame_)

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()


# --- MAIN LOOP ---

if __name__ == "__main__":
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: The model file '{model_path}' does not exist.")
        raise RuntimeError("Error: model path does not exist.")

    video_paths = []
    for root, dirs, files in os.walk(video_folder):
        for file in files:
            if file.endswith(".mp4"):
                full_path = os.path.join(root, file)
                video_paths.append(full_path)

    if not video_paths:
        print(f"No .mp4 files found in '{video_folder}'.")
        raise RuntimeError("Error: no mp4 files in video folder.")

    for i, video_path in enumerate(video_paths):
        try:
            print(f"Processing video {i + 1} of {len(video_paths)}")

            # Extract subfolder names and original video name
            path_parts = os.path.normpath(video_path).split(os.sep)
            if len(path_parts) < 3:
                print(f"Skipping video due to unexpected path structure: {video_path}")
                continue

            primary_folder = path_parts[-3]
            secondary_folder = path_parts[-2]
            video_filename = os.path.splitext(os.path.basename(video_path))[0]

            output_filename = (
                f"{primary_folder}_{secondary_folder}_{video_filename}_yolo11.mp4"
            )
            csv_filename = (
                f"{primary_folder}_{secondary_folder}_{video_filename}_yolo11.csv"
            )

            output_path = os.path.join(output_folder, output_filename)
            csv_path = os.path.join(csv_folder, csv_filename)

            inference(
                model=model,
                video_path=video_path,
                output_path=output_path,
                csv_path=csv_path,
            )

            print("Inference completed and CSV file generated.")
            print(f"Output video saved to: {output_path}")
            print(f"CSV file saved to: {csv_path}")
            print("--------------------------------------------------")

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            continue

    print("All videos processed.")
    print("--------------------------------------------------")
