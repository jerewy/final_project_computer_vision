# Motorcycle Helmet Detection with YOLO11

This project implements a motorcycle helmet detection system using the YOLO11 object detection model. The system classifies two classes:

- Helmet
- No Helmet

The model is trained on the v2 Helmet Detection Dataset (3,735 images) and evaluated across multiple YOLO11 variants.

---

## 1. Dataset

The dataset was resplit into an 80/10/10 structure:

```
dataset_resplit_80_10_10/
+- train/
+- val/
+- test/
```

Dataset classes:

```
nc: 2
names: ['Helmet', 'No Helmet']
```

---

## 2. Experiments

Three training experiments were performed:

### Exp1 - YOLO11n (No Augmentation)
- No online augmentation
- Used as baseline

### Exp2 - YOLO11n (Light Augmentation)
- Small rotation, translation, scale, and shear
- Horizontal flip enabled
- Mosaic / mixup disabled

### Exp3 - YOLO11s (Light Augmentation)
- Same augmentations as Exp2
- Larger model for better accuracy

All models were trained with:

- epochs = 80
- imgsz = 640
- batch = 16
- seed = 42
- GPU: RTX 4060 (if available)

---

## 3. Results (Test Set)

| Model                     | mAP50  | mAP50-95 | Precision | Recall | FPS   |
| ------------------------- | ------ | -------- | --------- | ------ | ----- |
| Exp1: YOLO11n (No Aug)    | 0.7737 | 0.4736   | 0.8160    | 0.6920 | 133.8 |
| Exp2: YOLO11n (Light Aug) | 0.9027 | 0.5601   | 0.9169    | 0.8255 | 188.8 |
| Exp3: YOLO11s (Light Aug) | 0.9096 | 0.5774   | 0.9225    | 0.8358 | 127.1 |

**Key Findings**

- Light augmentation significantly improves performance.
- YOLO11s (Exp3) achieves the highest accuracy.
- YOLO11n (Exp2) is the fastest at ~189 FPS.
- All models exceed real-time requirements (>30 FPS).

---

## 4. Notebook

All training, evaluation, and plotting are inside:

```
notebooks/helmet_yolo11_experiments.ipynb
```

This notebook includes:

- Dataset verification
- Training for Exp1-Exp3
- Test evaluation metrics
- mAP / Precision / Recall / FPS plots
- Training curves
- Confusion matrix
- Sample inference results

---

## 5. Installation & Reproduction

### Clone the repository

```bash
git clone <your-repo-url>.git
cd final_project_computer_vision
```

### Install dependencies

```bash
# Step 1 - Install PyTorch (choose version based on your GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 2 - Install everything else
pip install -r requirements.txt
```

### Run the notebook

Open:

```
notebooks/helmet_yolo11_experiments.ipynb
```

Run the cells to:

- Train models (optional)
- Reproduce evaluations
- View plots and confusion matrix
- Test inference on sample images

---

## 6. Repository Structure

```
.
+- notebooks/
|  +- helmet_yolo11_experiments.ipynb
+- dataset_resplit_80_10_10/
+- runs/
|  +- helmet/
+- models/
+- helmet.yaml
+- requirements.txt
+- README.md
```

---

## 7. Streamlit App (Helmet Detector)

A ready-to-run Streamlit app is in `app/streamlit_app.py` with three input modes: image upload, live webcam, and video playback (upload a clip or give a local/RTSP/HLS path). The UI centers results and renders around 800px wide with responsive columns, while resizing frames to ~960px for inference to reduce lag. Prefer uploads/local files for reliability; external CCTV/RTSP links can be flaky.

### Run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app expects the weights at `models/helmet_detector_yolo11s_v2.pt`. Update `MODEL_PATH` in the script if you move the file.

### Deploy options

- Hugging Face Spaces (Streamlit runtime):
  1. Create a new Space and choose Streamlit.
  2. Add this repo content (include the model file or add a download step).
  3. Set the entrypoint command to `streamlit run app/streamlit_app.py`.
  4. RTSP/HLS/CCTV sources must be reachable from HF; private networks will not work.

- Streamlit Community Cloud:
  1. Connect this repo.
  2. Set the app file to `app/streamlit_app.py`.
  3. Ensure the model file is committed or fetched at startup.

For both deployments, confirm outbound network access if you need to pull model weights or read CCTV/RTSP/HLS sources.
