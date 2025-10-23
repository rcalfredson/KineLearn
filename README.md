# KineLearn
Pose-based behavior classification from DeepLabCut keypoints.

KineLearn is designed to bridge **pose estimation** and **behavior classification**.  
It builds on DeepLabCut keypoint data to extract interpretable kinematic features (e.g., angles, distances, relative coordinates) and align them with user-defined behavioral annotations.  

Currently, KineLearn focuses on **feature extraction and normalization**, serving as the foundation for upcoming modules that will handle **model training**, **prediction**, and **visualization**.

➡️ See [Using KineLearn to Generate Features](#-using-kinelearn-to-generate-features) for setup and usage.


## Installation

### 1. Prerequisites
- **Python:** 3.11 (recommended)
- **CUDA-enabled GPU (optional):**  
  If you plan to train models on GPU, make sure your system has:
  - NVIDIA driver ≥ 535
  - CUDA 12.x toolkit
  - cuDNN 9.x  
  These are installed automatically when using `tensorflow[and-cuda]` if the drivers are present.

> 💡 If you don’t have an NVIDIA GPU, TensorFlow will fall back to CPU automatically.

---

### 2. Create a virtual environment
You can use **Conda**, **venv**, or any other environment manager.

**Using Conda (recommended):**
```bash
conda create -n kinelearn python=3.11
conda activate kinelearn
```

**Using venv:**
```bash
python3.11 -m venv kinelearn
source kinelearn/bin/activate
```

### 3. Install dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

The main dependencies include:
- TensorFlow (with GPU support if available)
- Keras
- Scikit-learn
- OpenCV
- Matplotlib / Seaborn for visualization
- Pandas / PyArrow for data handling

---

### 4. Verify the installation
Check that TensorFlow detects your GPU (optional):

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output (if GPU is available):
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---
## 🧩 Using KineLearn to Generate Features

KineLearn extracts pose-based behavioral features from DeepLabCut (DLC) keypoints and prepares them for model training.  
Before running the feature extraction step, ensure the following setup is complete.

---

### 1. Prerequisites

You should already have:

- A **DeepLabCut project and trained model** that you’ve used to generate pose predictions (`.csv` or `.h5`) for your videos.
- Behavior annotations for those videos created in **BORIS**.

> ⚠️ **Note:** Currently, KineLearn supports one animal per video scene.

---

### 2. Export behavior annotations

After annotating behaviors in BORIS:
1. Go to **Export → Tabular events**.
2. Save each file in the same directory as its corresponding video and DLC output.
3. Use the following filename convention:

```
{stem_of_video_filename}_ground_truth.tsv
```

Example:

```
video_20250704_ground_truth.tsv
```

---

### 3. Create a KineLearn configuration file

Use the provided example in `configs/` as a template.  
This YAML file defines:
- `dlc_config`: Path to your DeepLabCut project configuration file.
- `behaviors`: List of behavior names that appear in your BORIS annotations.
- `features`:  
  - `ref_pt`: The keypoint used as the origin for relative coordinates.  
  - `body_length_pts`: A pair of keypoints defining body length (used to normalize relative coordinates).  
  - `distances`: Pairs of keypoints for distance features.  
  - `angles`: Triplets of keypoints for angle features.

Example:

```yaml
dlc_config: /path/to/your/dlc/config.yaml

behaviors:
- grooming
- wing_extension
- abdomen_bend

features:
ref_pt: thorax
body_length_pts:
 - thorax
 - abdomen_tip
distances:
 - [head, thorax]
 - [thorax, abdomen_tip]
angles:
 - [head, thorax, abdomen_tip]
```

---
### 4. Create a video list file
Create a YAML file containing the list of videos you want to process.
Each line should be a full path to a video file:
```yaml
- /data/videos/fly_001.mp4
- /data/videos/fly_002.mp4
- /data/videos/fly_003.mp4
```
Save this file somewhere convenient, e.g. `video_lists/train_videos.yaml`.

---

### 5. Run feature extraction

Use the `kinelearn-calc` command-line tool to extract and scale features.

```bash
kinelearn-calc \
  -v video_lists/train_videos.yaml \
  --kl-config configs/drosophila.yaml \
  --create-scalers
```

* The `--create-scalers` flag fits new normalization scalers (`StandardScaler`) on the data in your training/validation/test set.
* The fitted scalers are automatically saved under `scalers/` and their paths added to your KineLearn config file.
* When you later process new videos (for inference or testing), **omit** `--create-scalers` so the existing scalers will be loaded automatically.

Each processed video will produce:
* `frame_features_<video_stem>.parquet`— per-frame features
* `frame_labels_<video_stem>.parquet`— per-frame labels (from BORIS)

Both files are saved in your output directory (default: `features/`).

---

### ✅ Example output structure
```
features/
├── extracted_features_video_20250704.csv
├── frame_features_video_20250704.parquet
├── frame_labels_video_20250704.parquet
scalers/
├── scaler_drosophila_coordinates.pkl
├── scaler_drosophila_velocity.pkl
├── ...

```

---

### 🧠 Tips
* Ensure all videos in a single run share the same frame rate (the script checks this automatically).
* For DeepLabCut `.h5` files, KineLearn will convert them to `.csv` automatically.
* Missing or incomplete feature data are mean-imputed during export.
* Behavior labels not found in BORIS exports default to zeros (no behavior active).

---
## 🧠 Training a Behavior Classifier
_Coming soon_

---
## 📊 Evaluating Predictions
_Coming soon_

---

## 🎨 Visualizing Behavioral Dynamics
_Coming soon_