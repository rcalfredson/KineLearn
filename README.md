# KineLearn
Pose-based behavior classification from DeepLabCut keypoints.

KineLearn is designed to bridge **pose estimation** and **behavior classification**.  
It builds on DeepLabCut keypoint data to extract interpretable kinematic features (e.g., angles, distances, relative coordinates) and align them with user-defined behavioral annotations.  

Currently, KineLearn focuses on **feature extraction and normalization**, serving as the foundation for upcoming modules that will handle **model training**, **prediction**, and **visualization**.

## Table of Contents
- [Installation](#installation)
- [🧩 Using KineLearn to Generate Features](#-using-kinelearn-to-generate-features)
- [🧭 Splitting Data into Train and Test Sets](#-splitting-data-into-train-and-test-sets)
- [🧠 Training a Behavior Classifier](#-training-a-behavior-classifier)
- [📊 Evaluating Predictions](#-evaluating-predictions)
- [🎨 Visualizing Behavioral Dynamics](#-visualizing-behavioral-dynamics)

---

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

### 4. Verify the environment
Check that TensorFlow detects your GPU (optional):

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output (if GPU is available):
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
---
### 5. Installing KineLearn itself

You can install KineLearn either as a **user package** or in **editable (developer) mode**.

**A. Standard installation (recommended for users):**
```bash
pip install .
```
This installs KineLearn normally, adding the CLI commands  
`kinelearn-calc`, `kinelearn-split`, and (when available) `kinelearn-train` to your PATH.

**B. Developer installation (for code modification):**
```bash
pip install -e .
```
The `-e` (editable) flag links the package directly to your working directory,  
so any changes you make to the source code are applied immediately—  
no need to reinstall unless you modify `pyproject.toml` or add new dependencies.

---

### 6. Verify command-line tools

After installation, check that the KineLearn CLI commands are recognized:

```bash
kinelearn-calc --help
kinelearn-split --help
```

If these commands run successfully, your environment is correctly configured.

---

💡 *Why the difference?*  
- Use `pip install .` if you’re just **using** KineLearn (e.g., for feature extraction or model training).  
- Use `pip install -e .` if you’re **developing** KineLearn and want edits to take effect instantly.  

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
## 🧭 Splitting Data into Train and Test Sets

Once you have generated the feature and label files for your videos,
the next step is to divide them into **training** and **testing** sets.

KineLearn provides a command-line tool for this: `kinelearn-split`.

---

### 1. Create a split

```bash
kinelearn-split \
  -v video_lists/all_videos.yaml \
  --seed 42
```

This command:
* Reads the list of video paths from the YAML file (`-v`).
* Randomly splits the videos into train and test sets (80/20 by default).
* Saves the split information as a YAML file under `data_splits/`.

Example auto-generated filename:

```
data_splits/all_videos_split_20251023_153805.yaml
```

---

### 2. Customize the split

You can control the split behavior with optional flags:

| **Flag** | **Description** | **Default** |
| :--- | :--- | :--- |
| `--seed` | Sets a random seed for reproducible splits. | None |
| `--test-fraction` | Fraction of videos to reserve for testing. | 0.2 |
| `--out` | Custom path for the split output YAML file. | Automatically generated under `data_splits/` |

Example:

```
kinelearn-split \
  -v video_lists/july_sessions.yaml \
  --test-fraction 0.25 \
  --seed 123 \
  --out data_splits/july_sessions_split.yaml
```

---

### 3. Split file format
The generated YAML file lists which video stems belong to each set:

```yaml
seed: 42
test_fraction: 0.2
train:
  - fly_001
  - fly_002
  - fly_003
test:
  - fly_004
```

These stems correspond directly to the feature and label filenames, e.g., `frame_features_fly_001.parquet`, `frame_labels_fly_001.parquet`.

---

### ✅ Example output structure

```
data_splits/
├── all_videos_split_20251023_153805.yaml
features/
├── frame_features_fly_001.parquet
├── frame_labels_fly_001.parquet
scalers/
├── scaler_drosophila_velocity.pkl
├── scaler_drosophila_angles.pkl

```

---

### 🧠 Tips

* The split uses only **filename stems** (not full paths), so it's flexible across systems.
* The split file can be reused for consistent training/testing across multiple runs.
* You can generate different splits (e.g., by session, condition, or date) and store them in `data_splits/`.

---
## 🧠 Training a Behavior Classifier
_Coming soon_

---
## 📊 Evaluating Predictions
_Coming soon_

---

## 🎨 Visualizing Behavioral Dynamics
_Coming soon_