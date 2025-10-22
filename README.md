# KineLearn
Pose-based behavior classification from DeepLabCut keypoints.

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
