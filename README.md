# Bone Fracture Classification — MURA X-ray Dataset

## ML Engineering Assessment — Computer Vision

A modular, production-ready bone fracture classification system built with **TensorFlow/Keras** and **FastAPI**. Compares a **Baseline CNN** against a **fine-tuned MobileNetV3-Small** on the [MURA (Musculoskeletal Radiographs)](https://stanfordmlgroup.github.io/competitions/mura/) dataset.

---

## Project Structure

```
├── Models/
│   ├── CNN_final.keras          # Optimized CNN weights
│   └── MobileNetV3_final.keras  # MobileNetV3 weights (production model)
├── Notebooks/
│   ├── 01-training-and-comparison.ipynb   # Data processing, training, evaluation
│   └── 02-testing-and-error-analysis.ipynb # Error analysis & random inference
├── src/
│   ├── config.py            # All configurable parameters
│   ├── model.py             # Architecture definitions & weight loading
│   ├── inference.py         # Preprocessing, prediction, CLI entry point
│   ├── monitoring.py        # Prediction logging and drift detection
│   └── api.py               # FastAPI application (multi-model + static UI)
├── static/
│   └── index.html           # Custom native HTML/JS clinical dashboard
├── logs/
│   └── predictions.log      # Created at runtime (not committed)
├── test_api.py              # Automated API test suite
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Dataset & Preprocessing

### MURA-v1.1

The **MURA (Musculoskeletal Radiographs)** dataset is one of the largest public radiograph datasets, containing **40,561** images across 7 upper-extremity types. *Note: For this prototype, to accommodate hardware constraints and rapid iteration, a representative subset of the dataset was utilized for training and evaluation.*

### Data Pipeline

The raw MURA dataset is stored in a deeply nested folder hierarchy with separate CSV files for train/validation paths. We flatten and reorganize it into a clean, model-ready structure:

```
MURA-v1.1/                          mura_flat/
├── train_image_paths.csv    →      ├── train/
├── valid_image_paths.csv    →      │   ├── fractured/
├── train/                          │   └── not_fractured/
│   └── XR_WRIST/                   ├── val/
│       └── patient00001/           │   ├── fractured/
│           └── study1_positive/    │   └── not_fractured/
│               └── image1.png      └── test/
└── valid/                              ├── fractured/
    └── ...                             └── not_fractured/
```

**Key steps:**
1. **Flattening:** Extracted all images from the nested patient/study structure into flat `fractured/` and `not_fractured/` directories, using the `positive`/`negative` folder naming convention to assign labels.
2. **Patient-Level Splitting:** The validation set was further split into `val` and `test` splits **at the patient level** (not image level) to prevent data leakage.
3. **Leakage Verification:** Explicit patient ID extraction and cross-set intersection checks confirm **zero patient overlap** between train, val, and test.
4. **Image Resizing & Cleaning:** All images were resized from their native high-resolution formats down to **224×224 pixels** (standard ImageNet input size) to drastically reduce memory overhead during training while preserving fracture visibility. Images were also converted to clean JPEGs via PIL to handle any corrupted decodings.

### Class Imbalance

| Class | Count | Weight |
|-------|-------|--------|
| `fractured` | ~1,567 | **1.28** (upweighted) |
| `not_fractured` | ~2,436 | **0.82** (downweighted) |

The dataset has a **40/60 class imbalance**. We handle this via **weighted loss** (`class_weight` parameter in `model.fit()`), which penalizes misclassifications on the minority class (`fractured`) more heavily. This is critical in a medical context — a missed fracture (false negative) is far more dangerous than a false alarm.

### Data Augmentation (Training Only)

```python
RandomHorizontalFlip()    # X-rays can appear mirrored
RandomRotation(0.15)      # Simulates slight positioning variation
RandomZoom(0.1)           # Handles distance variation
```

Conservative augmentation preserves diagnostic features while adding variety.

---

## Models & Training Strategy

### Model Comparison

| Model | Architecture | Params | Strategy | LR |
|-------|-------------|--------|----------|----|
| **Baseline CNN** | 4-block CNN (16→32→64→128) | ~200K | Train from scratch | 1e-3 |
| **MobileNetV3-Small** | ImageNet pretrained backbone | ~1.5M | Two-phase fine-tuning | 1e-3 → 1e-4 |

### Baseline CNN (Trained from Scratch)

An optimized custom CNN inspired by modern best practices:

```
Input(224×224×3)
├── Rescaling(1/255)
├── Conv2D(16, use_bias=False) → BatchNorm → ReLU → MaxPool
├── Conv2D(32, use_bias=False) → BatchNorm → ReLU → MaxPool
├── Conv2D(64, use_bias=False) → BatchNorm → ReLU → MaxPool
├── Conv2D(128, use_bias=False) → BatchNorm → ReLU → MaxPool
├── GlobalAveragePooling2D
├── Dropout(0.4) → Dense(128, L2=0.005) → Dropout(0.3)
└── Dense(2, softmax, L2=0.005)
```

**Key design choices:**
- **BN-before-ReLU:** Normalizes raw convolution outputs before activation (original BatchNorm paper pattern).
- **`use_bias=False`:** Since BatchNorm subtracts the mean, the Conv2D bias is mathematically redundant — removing it saves parameters.
- **L2 Regularization (0.005):** Prevents the dense layers from memorizing training data.
- **Dual Dropout (0.4 + 0.3):** Additional overfitting defense for the classifier head.

### MobileNetV3-Small (Two-Phase Fine-Tuning)

Our production model uses a two-phase training strategy that combines Transfer Learning and Fine-Tuning:

**Phase 1 — Transfer Learning (Head Only):**
- Freeze entire MobileNetV3 base (ImageNet weights locked).
- Train only the new classification head for **5 epochs** at **LR=1e-3**.
- Purpose: Align the random head weights with the pre-trained feature representations.

**Phase 2 — Fine-Tuning (Last 15 Layers):**
- Unfreeze the last **15 layers** of the base model.
- Train with a much smaller **LR=1e-4** for **15 epochs**.
- Purpose: Adapt the deepest feature detectors to bone X-ray patterns without destroying earlier edge/shape detectors.

**Why two phases?** Training a random head on an unfrozen network causes "Gradient Shock" — massive error gradients from the untrained head backpropagate and destroy the valuable pre-trained weights. Phase 1 eliminates this risk.

### Callbacks

Both models use:
- **EarlyStopping:** Stops training when validation loss stops improving.
- **ReduceLROnPlateau:** Halves the learning rate when loss plateaus.
- **ModelCheckpoint:** Saves the best model (lowest val_loss) during training.

---

## Results

### Default Threshold (0.50)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
|-------|----------|--------------------|----------------|------------------|
| Baseline CNN | **61%** | 0.61 | 0.61 | 0.60 |
| **MobileNetV3** | **74%** | 0.75 | 0.74 | 0.73 |

**Per-class breakdown:**

| Model | Class | Precision | Recall | F1-Score | Support |
|-------|-------|-----------|--------|----------|---------|
| Baseline CNN | fractured | 0.63 | 0.48 | 0.54 | 730 |
| Baseline CNN | not_fractured | 0.60 | 0.73 | 0.66 | 770 |
| **MobileNetV3** | **fractured** | **0.80** | **0.62** | **0.70** | **730** |
| **MobileNetV3** | **not_fractured** | **0.70** | **0.86** | **0.77** | **770** |

> MobileNetV3 outperforms the Baseline CNN by **+13% accuracy** and **+13% macro F1**, demonstrating the power of two-phase fine-tuning with ImageNet pre-trained features on medical imaging data.

### Optimal Threshold Analysis

Since this is a medical application, we tuned the classification threshold to maximize F1 for fracture detection:

| Model | Default F1 | Optimal Threshold | Optimized F1 | Fractured Recall |
|-------|-----------|-------------------|--------------|------------------|
| Baseline CNN | 0.54 | **0.35** | 0.68 | 93% |
| **MobileNetV3** | 0.70 | **0.28** | **0.73** | **85%** |

**Key insight:** By lowering the MobileNetV3 threshold from 0.50 to 0.28, we boost fracture recall from 62% to **85%** — meaning the model catches 85% of all fractures. This is critical in clinical screening where missing a fracture is far more dangerous than a false alarm.

### Evaluation Includes

- **Per-model training curves** (accuracy & loss, train vs val)
- **Confusion matrices** with side-by-side comparison
- **ROC curves** with AUC scores
- **Classification reports** (precision, recall, F1 per class)
- **Metrics bar chart** comparing both models
- **Optimal threshold analysis** — finds the best classification threshold to maximize F1
- **Random inference visualization** — side-by-side model predictions on test images

---

## How to Run

### 1. Training (on Kaggle)

1. Upload the [MURA dataset](https://stanfordmlgroup.github.io/competitions/mura/) as a Kaggle Dataset
2. Upload the `Notebooks/` folder to a Kaggle Notebook with **GPU accelerator**
3. Run `01-training-and-comparison.ipynb` → models are saved as `.keras` files
4. Download `CNN_final.keras` and `MobileNetV3_final.keras` to `Models/`
5. Optionally run `02-testing-and-error-analysis.ipynb` for visual error analysis

### 2. Inference (CLI)

```bash
pip install -r requirements.txt

# Using MobileNetV3 (default, recommended)
python -m src.inference --image path/to/xray.png --model mobilenetv3

# Using Baseline CNN
python -m src.inference --image path/to/xray.png --model cnn
```

### 3. API Server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (lists loaded models) |
| `GET` | `/model-info?model=mobilenetv3` | Model details & parameter count |
| `POST` | `/predict?model=mobilenetv3` | Upload X-ray → fracture prediction |
| `GET` | `/stats` | Prediction statistics & drift analysis |
| `GET` | `/api/samples` | Dynamically lists all images in the `samples/` directory |
| `GET` | `/` or `/ui` | Serves the custom static HTML dashboard |

**Example (MobileNetV3):**

```bash
curl -X POST "http://localhost:8000/predict?model=mobilenetv3" \
     -F "file=@xray_sample.png"
```

**Example (CNN):**

```bash
curl -X POST "http://localhost:8000/predict?model=cnn" \
     -F "file=@xray_sample.png"
```

**Response:**
```json
{
    "prediction": "fractured",
    "confidence": 0.9234,
    "probabilities": {"fractured": 0.9234, "not_fractured": 0.0766},
    "model": "mobilenetv3"
}
```

### 4. Docker

```bash
docker build -t fracture-classifier .
docker run -p 8000:8000 fracture-classifier
```

### 5. Tests

```bash
pytest test_api.py -v
```

### 6. Deployment on Render (PaaS)

This project is architected specifically for lightweight cloud deployment on platforms like Render:
1. Connect your GitHub repository to Render as a **Web Service**.
2. Render will automatically detect the `Dockerfile`.
3. The `COPY samples/ samples/` and `COPY static/ static/` commands in the Dockerfile ensure that both the UI and the test images are securely baked into the cloud container.
4. Because we replaced Gradio with a native HTML/JS frontend, the RAM usage easily fits within Render's 512MB free tier limits!

**A Note on "Cold Starts"**: Because this prototype runs on Render's free tier, the server spins down after periods of inactivity. If you access the URL from a "dead start", it may take up to 60-90 seconds for the application to boot up and load the ML models into memory before the UI responds. This is normal behavior for free PaaS tiers.

---

## Engineering Decisions

| Decision | Rationale |
|----------|-----------|
| TensorFlow/Keras over PyTorch | Native `.keras` model saving, Kaggle-friendly, cleaner data pipeline API |
| MobileNetV3-Small over ResNet50 | Compute-aware: trains on free Kaggle GPU, deploys on CPU, 6× fewer params |
| Two-phase fine-tuning | Prevents catastrophic forgetting; industry standard for medical imaging |
| BN-before-ReLU in CNN | Original BatchNorm paper pattern; more stable gradient flow |
| Weighted loss over oversampling | No data duplication, works cleanly with augmentation pipeline |
| MURA flattening pipeline | Converts deeply nested clinical hierarchy into standard image classification format |
| Patient-level splitting | Prevents data leakage — no patient's images appear in multiple splits |
| FastAPI with multi-model support | Select model at inference time via query param; auto-generated OpenAPI docs |
| Custom HTML/JS Frontend | Replaced Gradio with a native static frontend. Eliminates huge dependencies, prevents URL routing bugs, and slashes memory usage for Render deployment. |
| JSON-line monitoring | No external deps (no MLflow/Prometheus), easy to parse, drift detection built-in |
| Optimal threshold tuning | Medical context demands high recall; threshold optimization maximizes F1 |

---

## Development Challenges & Solutions

Building a production-ready application requires overcoming several engineering hurdles. Here are the key errors encountered and how they were resolved:

| Error / Challenge | Root Cause | Solution Implemented |
|-------------------|------------|----------------------|
| **Gradio Image Loading 404s** | Gradio 4.x has complex internal routing logic that often breaks relative file paths and `file=` URLs when serving static galleries. | **Complete Architecture Pivot:** Removed the Gradio dependency entirely. Built a custom Vanilla HTML/JS frontend served natively via FastAPI `StaticFiles`, ensuring 100% reliable image loading. |
| **Render OOM (Out of Memory) Crashes** | Cloud platforms like Render offer 512MB RAM on free tiers. Heavy libraries like Gradio and full TensorFlow often exhaust this memory during startup. | **Dependency Pruning:** Removing Gradio drastically reduced RAM usage. The new custom frontend runs strictly on the client browser, leaving server RAM exclusively for FastAPI and Model Inference. |
| **Render "Model Not Loaded" Error** | Standard `.gitignore` templates block `*.keras` files to prevent exceeding GitHub's 100MB limit, preventing the cloud from seeing the models. | **Selective Tracking:** Verified that `MobileNetV3` is highly efficient (only 7.3MB) and explicitly removed the `.keras` block from `.gitignore`, allowing seamless deployment. |
| **Sample Images Missing on Cloud** | Hardcoded UI image lists break if the user uploads new sample images to the cloud container but forgets to update the frontend code. | **Dynamic API Endpoint:** Created a `/api/samples` REST endpoint that dynamically scans the container's hard drive at runtime and auto-populates the UI gallery. |
| **TensorFlow Gradient Shock** | Fine-tuning a pre-trained network with an uninitialized classification head destroys the valuable pre-trained weights via massive backpropagated errors. | **Two-Phase Fine-Tuning:** Phase 1 freezes the base and trains only the head. Phase 2 unfreezes the deepest layers using a 10× smaller learning rate. |
| **Keras Deserialization Crash (`quantization_config`)** | Models trained in Kaggle (Keras 3) crash when loaded in older deployment environments (Keras 2) due to an unrecognized `quantization_config` parameter. | **Binary Patch & Version Pinning:** Wrote a Python script to directly edit the `.keras` ZIP binary and strip the incompatible JSON configuration, while also strictly pinning `tensorflow==2.16.1` to force Keras 3 compatibility. |

---

## Future Work

- **Larger Backbone:** Swap MobileNetV3 for EfficientNet-B3 or ConvNeXt-Tiny for higher capacity
- **Ensemble Methods:** Average predictions from CNN + MobileNetV3 for improved robustness
- **Advanced Augmentation:** MixUp, CutMix, GridDistortion for better generalization
- **Grad-CAM Visualization:** Overlay heatmaps showing which bone regions drive predictions
- **Body Region Classification:** Multi-task learning to predict fracture AND body region simultaneously
- **DICOM Support:** Accept raw medical DICOM files directly instead of requiring JPEG conversion
- **Model Quantization:** Convert to TFLite for edge deployment on mobile/embedded devices
- **A/B Testing:** Route traffic between CNN and MobileNetV3 in production to measure real-world performance
- **Continuous Learning:** Retrain on new hospital data with drift-triggered retraining pipeline
