# 🧪 Master Thesis – No Reading Required: Using Gamified Eye-Tracking to Predict RAN Scores and Screen for Developmental Dyslexia

## 📚 Abstract
Eye-tracking combined with machine learning shows strong potential as a diagnostic 
tool for developmental dyslexia, particularly when decoupled from reading-based 
assessments.  However,  few  studies  have  explored  the  use  of  gamified,  non-reading  tasks 
suitable for pre-readers. This pilot study investigates whether eye-tracking metrics recorded 
during  gameplay  (Fruit  Ninja)  can  predict  Rapid  Automatized  Naming  scores  and  classify 
individuals  into  good  versus  poor  performance  groups.  A  total  of  23  eye-tracking  features 
were  extracted  from  57  participants  (children  and  adults)  and  analyzed  using  correlation, 
regression,  and  classification  models.  Correlation and  regression  analyses  identified several 
significant predictors of Rapid Automatized Naming performance, particularly features 
related to blink behaviours, saccade length, and gaze complexity, while classification models 
yielding  the  highest  accuracy  with  fixation-based  features.  These  findings  suggest  that 
oculomotor  behaviour  during  a  non-reading  task  contains  information  relevant  to  reading 
ability  and  may  reflect  magnocellular processing  differences.  This approach shows  promise 
for  early, literacy-independent  risk  detection,  but  further  research  with  larger  and  clinically 
diverse samples is needed to validate and extend these results. 

---

## 🎯 Project Goals
- Train an object detection model capable of tracking fruits and bombs from scene video of FruitNinja
- Extract 23 eye-tracking features from data
- Build a machine learning pipeline to screen for dyslexia without reading-based tasks
- Evaluate feature importance with Random Forest Feature Importance Ranking
- Interpretable AI with SHAP analysis
- Correlate eye-tracking features to RAN score
- Predict RAN performance using regression/classification

---

## 🚀 Usage

This project is organized into 6 pipeline stages (`0` to `5`). Each folder contains scripts that must be run **in numerical order**. Most steps include both `.py` and `.ipynb` files. The only exception is the first step, which is intended to be run in **Google Colab**.

```bash
# Step 0: Object detection in Google Colab
# Run this notebook in Colab (not locally)
0_object_detection/0.1_yolov8_train.ipynb

# Steps 1–5: Run locally in order
# Each folder may contain multiple .py and .ipynb files

python 1_model_fine-tuning/*.py
python 2_data_sync/*.py
python 3_annotation_tools/*.py
python 4_data_preprocessing/*.py
jupyter notebook 4_data_preprocessing/*.ipynb
jupyter notebook 5_analysis/*.ipynb
```
Paths are managed in `utils.paths.py`

---

## 🗂️ Project Structure
master_thesis/
├── 0_object_detection/ # YOLOv8s training and inference scripts
├── 1_model_fine-tuning/ # Scripts to clean inference
├── 2_data_sync/ # Script to synchronise inference data with eye-tracking data
├── 3_annotation_tools/ # Scripts to annotate eye-tracking data
├── 4_data_preprocessing/ # Scripts to concatenate split files, extract RAN scores, and complete feature extraction
├── 5_analysis/ # Descriptive stats, RF Feature Importance Ranking / SHAP, regression, classification, colourmaps, feature sets, plotting names
├── data/ # (Empty in repo – see Data section below)
├── trained_models/ # Trained model weights and performance stats
└── utils/ # Path config
---

## 🧠 Features Extracted

- Age (only for prediction task) 
- Fixation Fractal Dimension (FFD) 
- Maximum Latency 
- Mean Blink Duration 
- Mean Fixation Intersection Coefficient (FIC) 
- Mean Fixation Distance to Fruit 
- Mean Fixation Distance to Object 
- Mean Fixation Duration 
- Mean Latency 
- Mean Saccade Length 
- Median Blink Duration 
- Median Fixation Distance to Fruit 
- Median Fixation Distance to Object 
- Median Fixation Duration 
- Median Latency 
- Median Saccade Length 
- Number of Blinks per Minute 
- Number of Fixations per Minute 
- Number of Saccades per Minute 
- Percent of Fixations Outside the Bounding Boxes 
- Percent Un-Fixated Fruits 
- SD Fixation Intersection Coefficient (FIC) 
- SD Latency 
---

## 🧪 Models Used

- Random Forest Feature Importance Ranking and SHAP for feature selection
- Pearson Correlations for feature correlations
- Linear Regression for RAN prediction
- K-Nearest Neighbors, Logistic Regression, Support Vector Machine, Random Forest for classification
---

## 👁️ Data Structure

Due to privacy and ethical constraints, participant data is not included.
See [`data/README.md`](data/README.md) for expected data folder layout after running the full pipeline.
---

## 🧰 Requirements

```bash
# Using conda
conda env create -f environment.yml
conda activate your_env_name
```
---

## 📄 Thesis Document

📥 [Download thesis.pdf](./thesis.pdf)
---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
---

## 📬 Contact

For questions or collaboration opportunities, feel free to reach out:

**Olivia Lecomte**  
✉️ olivialecomte33@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/olivia-lecomte33)