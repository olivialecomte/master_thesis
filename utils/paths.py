from pathlib import Path

# --- Base Directories ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "trained_models"

# --- Raw Data ---
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_ZIPS = {
    "adults": RAW_DATA_DIR / "Adult_Spring.zip",
    "dyscover": RAW_DATA_DIR / "DysCover.zip",
    "fruit_ninja": RAW_DATA_DIR / "Fruit_Ninja.zip",
}

# --- Processed Data ---
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATED_DIR = PROCESSED_DIR / "annotated"
CLEANED_DIR = PROCESSED_DIR / "cleaned"
EXTRACTED_FEATURES_DIR = PROCESSED_DIR / "extracted_features"
FILTERED_DIR = PROCESSED_DIR / "filtered_csv"
INFERENCE_CSV_DIR = PROCESSED_DIR / "inference_csv"
INFERENCE_VIDEOS_DIR = DATA_DIR / "inference_videos"
RAN_DIR = PROCESSED_DIR / "RAN"

# --- External Datasets ---
EXTERNAL_DIR = DATA_DIR / "external"
OBJECT_DETECTION_DIR = EXTERNAL_DIR / "object_detection"
DATASET_10000A_DIR = OBJECT_DETECTION_DIR / "dataset10000A"
ROBOFLOW_COMBINED_DIR = OBJECT_DETECTION_DIR / "roboflow_2c_combined_et"


# --- Ensure key directories exist (optional) ---
def ensure_dirs():
    for path in [DATA_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    print(f"All necessary directories have been ensured at: {PROJECT_ROOT}")
