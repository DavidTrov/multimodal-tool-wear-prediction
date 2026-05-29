# MATWI Thesis – Project Context & Current State

## 🎯 Goal
Develop a **multimodal model (image + sensor)** for tool wear prediction using the MATWI dataset.

Final constraint:
- Model must eventually fit **~256KB**
- Therefore: initial work focuses on **reproducible baselines**, then later **compression / lightweight models**

---

# 📦 Dataset Overview

## Dataset: MATWI
- Multimodal dataset: **images + sensor data**
- Each **Set = one tool run until failure**
- Total ~1663 labeled samples

---

## Folder Structure

data/
raw/
Set1/
images/
sensordata/
Set2/
…
labels.csv
sets.csv

---

## `labels.csv` (CRITICAL FILE)

This is the **ground truth pairing file**.

Each row contains:
- `ImageFile` → path to image
- `SensorFile` → path to sensor CSV
- `Set` → tool run
- `wear` → wear value (µm) → regression target
- `type` → wear type (classification)
- timestamps (not needed for pairing)

⚠️ Important:
- Pairing is already done here
- Do NOT match by filename manually
- Some rows have:
  - image but no sensor
  - sensor but no image

---

## `sets.csv`

Contains per-set metadata:
- cutting parameters (Vc, n, fz, etc.)
- **crop region**:

(left, top, right, bottom)

This crop must be applied before training.

---

## Sensor CSV Structure

Each `.csv` file:
- 6 columns, no header
- comma-separated

Order:

1. Accelerometer
2. Acoustic
3. Force X
4. Force Y
5. Force Z
6. Timestamp

Each file:
- ~78k rows
- time-series data (NOT single measurement)

---

# 🧠 Observations from Exploration

## 1. Sensor Data Behavior

- Most files:
- similar distributions
- stable signals

- Some files:
- significantly different scale/distribution

Possible reasons:
- different cutting parameters
- different material (important)
- sensor anomalies

---

## 2. Outliers (Spikes)

Example:
- acoustic ≈ 0.41 normally
- spikes to ~7.88 briefly

Interpretation:
- could be:
- real signal (vibration, chatter, wear)
- sensor glitch

Decision:
- DO NOT remove yet
- keep raw data for baseline
- later experiment with clipping

---

# 📊 Feature Extraction (tsfresh)

Used for:
- **sensor-only baseline**
- NOT for final fusion model

Workflow:
- each sensor CSV = one sample
- extract statistical features (mean, std, FFT, etc.)

Used:
- `MinimalFCParameters` (for speed/stability)

---

# 📈 Evaluation Metric

From paper:

- **Primary:** MAE (Mean Absolute Error)
- Also report:
- std of MAE
- optionally RMSE

Why MAE:
- interpretable (µm)
- robust to outliers

---

# 🔀 Dataset Split (from paper)

## Training

Sets: 1, 2, 5, 7, 8, 10, 11

## Validation

Sets: 3, 6, 12

## Testing

Sets: 4, 9, 13

### Important notes:

- Sets 12–17 use **different material**
- Used to test generalization
- Set 3 also used to test different wear type

⚠️ Must follow this split exactly for reproduction

---

# 🖼️ Image Pipeline

## Required preprocessing

For each sample:

1. Load image from `ImageFile`
2. Get crop from `sets.csv` for that Set
3. Apply crop
4. Resize → `224x224`
5. Normalize (ImageNet)

---

# 🧱 Model Plan (Current Stage)

## Step 1 – Image-only baseline

Goal:
- reproduce paper baseline
- ensure pipeline is correct

### Backbone

Paper:
- modified ResNet50

Chosen:
- **ResNet18**

Reason:
- smaller
- faster
- sufficient for baseline
- easier to debug

---

## ResNet Sizes

| Model     | Params |
|----------|--------|
| ResNet18 | ~11.7M |
| ResNet34 | ~21.8M |
| ResNet50 | ~25.6M |

⚠️ All are too large for final 256KB constraint  
→ this stage is only for baseline validation

---

## Model Setup

- Backbone: `resnet18(pretrained=True)`
- Output: `1` (regression)
- Loss: `MSELoss`
- Eval metric: `MAE`

---

## Training Config (initial)

- LR: `1e-4`
- Optimizer: Adam
- Batch size: 16–32
- Epochs: ~30
- Input size: 224

---

# 🧪 Current Scripts

## Sensor Analysis

Script:
- iterates over all sensor CSVs in a set
- computes:
  - min, max
  - mean
  - std
  - percentiles
  - outlier count

Outputs:
- `set_stats_long.csv`
- `set_stats_wide.csv`

Also prints:
- ranking per channel (acc, acoustic, fx, fy, fz)

---

# 🚀 Next Tasks

## Immediate

1. Implement **PyTorch Dataset**
   - read `labels.csv`
   - apply split
   - apply per-set crop

2. Implement **ResNet18 model**

3. Train image-only model

4. Compare performance with paper

---

## Critical Checkpoint

If performance is:

- ✅ Close → proceed to fusion  
- ❌ Worse → debug pipeline (likely crop / split issue)

---

## After That

1. Sensor-only baseline (tsfresh + ML model)
2. Compare:
   - image vs sensor
3. Build fusion model
4. Optimize for size (later stage)

---

# ⚠️ Key Rules Going Forward

- Do NOT change data cleaning during baseline reproduction
- Do NOT mix sets randomly → always split by Set
- Always apply correct crop per Set
- Always evaluate with MAE

---

# 🧠 Summary

You are currently in:

> **Phase 1: Baseline reproduction (image-only)**

Do NOT jump to fusion yet.

Everything depends on:
- correct split
- correct crop
- correct evaluation

---

# 🔥 If continuing this work (for another LLM)

Start from:

1. Implement dataset class using `labels.csv` + `sets.csv`
2. Train ResNet18 image-only model
3. Validate against paper MAE

Then proceed to multimodal fusion.
