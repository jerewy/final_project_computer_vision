import time
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "helmet_detector_yolo11s_v2.pt"
DEFAULT_HLS = ""  # leave blank; prefer uploads/local files
DISPLAY_WIDTH = 800  # target display width for images/frames
INFER_MAX_WIDTH = 960  # resize frames before inference to keep speed reasonable


@st.cache_resource(show_spinner="Loading YOLO11 helmet detector...")
def load_model() -> YOLO:
    return YOLO(str(MODEL_PATH))


def _count_detections(result) -> Dict[str, int]:
    names = result.names
    cls = result.boxes.cls.cpu().numpy() if result.boxes else np.array([])
    counts: Dict[str, int] = {name: 0 for name in names.values()}
    if cls.size == 0:
        return counts
    unique, freq = np.unique(cls.astype(int), return_counts=True)
    for idx, count in zip(unique, freq):
        counts[names[int(idx)]] = int(count)
    return counts


def run_inference(image_bgr: np.ndarray, conf: float) -> Tuple[np.ndarray, Dict[str, int]]:
    model = load_model()
    result = model.predict(image_bgr, conf=conf, verbose=False)[0]
    annotated_bgr = result.plot()  # OpenCV-style BGR array
    return annotated_bgr, _count_detections(result)


def resize_for_inference(image_bgr: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if w <= max_width:
        return image_bgr
    scale = max_width / float(w)
    new_size = (max_width, int(h * scale))
    return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)


def inject_center_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg1: #0d1b2a;
            --bg2: #1b263b;
            --card: #102641;
            --card-border: rgba(255,255,255,0.07);
            --accent: #7bd389;
            --accent-2: #4cc9f0;
            --text: #f5f7fb;
            --muted: #9fb1c7;
        }
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 20% 20%, rgba(124, 211, 137, 0.12), transparent 25%),
                        radial-gradient(circle at 80% 0%, rgba(76, 201, 240, 0.18), transparent 32%),
                        linear-gradient(135deg, var(--bg1), var(--bg2));
            color: var(--text);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: #0f1e33;
            color: var(--text);
        }
        .hero {
            padding: 18px 18px 10px 18px;
            border-radius: 14px;
            border: 1px solid var(--card-border);
            background: linear-gradient(135deg, rgba(124, 211, 137, 0.16), rgba(76, 201, 240, 0.14));
            color: var(--text);
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            margin-bottom: 12px;
        }
        .hero h1 { margin: 0; }
        .hero p { margin: 6px 0 0 0; color: var(--muted); }
        .card {
            background: var(--card);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        }
        .stTabs [data-baseweb="tab"] {
            color: var(--muted);
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--accent-2);
        }
        .stTabs [aria-selected="true"] {
            color: var(--text) !important;
            border-color: var(--accent-2) !important;
        }
        .stSlider [role="slider"] {
            accent-color: var(--accent);
        }
        .stButton>button {
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            color: #0d1b2a;
            border: none;
            font-weight: 700;
        }
        .stImage { text-align: center; }
        .stImage img {
            margin-left: auto;
            margin-right: auto;
            width: min(800px, 100%);
            height: auto;
            border-radius: 8px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.08);
        }
        .st-emotion-cache-r8fbmg.e1mq0gaz2 {
            text-align: center;
            color: var(--muted);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def centered_slot():
    """Return an st.empty() hosted in a center column."""
    _, col_c, _ = st.columns([1, 2, 1])
    return col_c.empty()


def show_centered_frame(frame_bgr: np.ndarray, caption: str) -> None:
    """Render a frame centered with fixed display width."""
    slot = centered_slot()
    slot.image(frame_bgr[:, :, ::-1], channels="RGB", caption=caption, width=DISPLAY_WIDTH)


def render_image_mode(conf: float) -> None:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Upload a helmet/no-helmet photo to run detection.")
        return

    image = Image.open(uploaded).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_bgr = resize_for_inference(image_bgr, INFER_MAX_WIDTH)
    annotated_bgr, counts = run_inference(image_bgr, conf)

    orig_rgb = image_bgr[:, :, ::-1]
    annotated_rgb = annotated_bgr[:, :, ::-1]
    col_orig, col_det = st.columns(2)
    col_orig.image(orig_rgb, caption="Original", width=DISPLAY_WIDTH // 2)
    col_det.image(annotated_rgb, caption="Detections", width=DISPLAY_WIDTH // 2)
    st.success(f"Helmet: {counts.get('Helmet', 0)} | No Helmet: {counts.get('No Helmet', 0)}")


def render_webcam_live(conf: float) -> None:
    st.write("Live webcam detection (device 0). Centered view ~800px wide; frames resized to ~960px for inference to reduce lag.")
    fps_limit = st.slider("Max FPS to display", 1, 15, 6, 1, key="webcam_fps")
    max_frames = st.slider("Max frames per run", 60, 900, 300, 30, key="webcam_max_frames")

    start = st.button("Start webcam")
    if not start:
        st.info("Click Start to open the webcam and begin live detection.")
        return

    frame_slot = centered_slot()
    stats_slot = st.empty()
    status = st.empty()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        st.error("Could not open webcam (device 0). Check permissions or camera availability.")
        return

    status.info("Webcam running... click Stop or wait for the frame limit.")
    stop = st.button("Stop webcam", key="stop_webcam")

    frames_shown = 0
    delay = 1.0 / fps_limit
    while cap.isOpened():
        if stop:
            status.warning("Webcam stopped by user.")
            break

        ok, frame = cap.read()
        if not ok:
            status.error("Cannot read webcam frame.")
            break

        frame = resize_for_inference(frame, INFER_MAX_WIDTH)
        annotated_bgr, counts = run_inference(frame, conf)
        frame_slot.image(annotated_bgr[:, :, ::-1], channels="RGB", caption="Webcam detections", width=DISPLAY_WIDTH)
        stats_slot.write(f"Helmet: {counts.get('Helmet', 0)} | No Helmet: {counts.get('No Helmet', 0)}")

        frames_shown += 1
        if frames_shown >= max_frames:
            status.warning("Reached frame limit. Restart to continue.")
            break

        time.sleep(delay)

    cap.release()


def render_stream_mode(conf: float) -> None:
    st.write("Use a local video file path or upload a clip (preferred). Network RTSP/HTTP/HLS streams are optional and may be unreliable.")
    source = st.text_input("Optional: Stream URL or local video path", value=DEFAULT_HLS, placeholder="rtsp://... or /path/to/video.mp4")
    uploaded_video = st.file_uploader("Or upload a video file (mp4/mov/avi/mkv)", type=["mp4", "mov", "avi", "mkv"])
    fps_limit = st.slider("Max FPS to display", min_value=1, max_value=15, value=5, key="stream_fps")
    max_frames = st.slider("Max frames per run (keeps sessions responsive)", 30, 900, 300, step=30, key="stream_max_frames")

    start = st.button("Start stream")
    if not start:
        st.info("Upload a clip, provide a file path, or enter a URL, then click Start.")
        return

    temp_file = None
    input_source = source
    if uploaded_video:
        suffix = f".{uploaded_video.name.split('.')[-1]}" if '.' in uploaded_video.name else ".mp4"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(uploaded_video.read())
        temp_file.flush()
        input_source = temp_file.name

    if not input_source:
        st.info("Please provide a URL/path or upload a video file.")
        if temp_file:
            Path(temp_file.name).unlink(missing_ok=True)
        return

    frame_slot = centered_slot()
    stats_slot = st.empty()
    status = st.empty()

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        st.error("Could not open the stream or video file. Check the URL/path or upload a different clip.")
        if temp_file:
            Path(temp_file.name).unlink(missing_ok=True)
        return

    status.info("Streaming... click Stop or wait for the frame limit.")
    stop = st.button("Stop stream", key="stop_stream")

    frames_shown = 0
    delay = 1.0 / fps_limit
    while cap.isOpened():
        if stop:
            status.warning("Stream stopped by user.")
            break

        ok, frame = cap.read()
        if not ok:
            status.error("Stream ended or cannot read frames.")
            break

        frame = resize_for_inference(frame, INFER_MAX_WIDTH)
        annotated_bgr, counts = run_inference(frame, conf)
        frame_slot.image(annotated_bgr[:, :, ::-1], channels="RGB", caption="Stream detections", width=DISPLAY_WIDTH)
        stats_slot.write(f"Helmet: {counts.get('Helmet', 0)} | No Helmet: {counts.get('No Helmet', 0)}")

        frames_shown += 1
        if frames_shown >= max_frames:
            status.warning("Reached frame limit for this session. Restart to continue.")
            break

        time.sleep(delay)

    cap.release()
    if temp_file:
        Path(temp_file.name).unlink(missing_ok=True)


def main() -> None:
    st.set_page_config(page_title="Helmet Detection", page_icon="Helmet", layout="wide")
    inject_center_style()
    st.markdown(
        """
        <div class="hero">
            <h1>Helmet Detection (YOLO11)</h1>
            <p>Upload an image, run live webcam detection, or stream RTSP/HLS/video feeds to check for helmets.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}. Place the trained .pt file there and retry.")
        return

    conf = st.slider("Confidence threshold", 0.1, 0.8, 0.3, 0.05)

    tabs = st.tabs(["Image Upload", "Webcam Live", "RTSP/Video Stream"])
    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_image_mode(conf)
        st.markdown("</div>", unsafe_allow_html=True)
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_webcam_live(conf)
        st.markdown("</div>", unsafe_allow_html=True)
    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_stream_mode(conf)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
