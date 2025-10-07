import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from tensorflow.keras.models import load_model
from src.utils import preprocess_frame

st.set_page_config(page_title="Early Theft Detect (minimal)", layout="wide")
st.title("Early Theft Attempt Detection — Minimal Demo")

# ------- Config / Sidebar -------
st.sidebar.header("Settings")
MODEL_FILENAME = st.sidebar.text_input("Model filename (in src/model/)", "lrcn_160S_90_90Q.h5")
MODEL_PATH = os.path.join("src", "model", MODEL_FILENAME)
FRAME_WIDTH = st.sidebar.number_input("Webcam width", value=320)
FRAME_HEIGHT = st.sidebar.number_input("Webcam height", value=180)
CONF_THRESH = st.sidebar.slider("Suspicion threshold", 0.0, 1.0, 0.5)
INFERENCE_INTERVAL = st.sidebar.number_input("Seconds between predictions", value=0.6, format="%.2f")
SCALE_FACTOR = st.sidebar.number_input("Confidence scale factor", value=1.0, format="%.1f")
DEBUG_MODE = st.sidebar.checkbox("Debug Mode", value=True)

load_model_btn = st.sidebar.button("Load model")

# ------- Model loader (store in session_state) -------
if "detector" not in st.session_state:
    st.session_state["detector"] = None

def test_model_with_dummy_data():
    """Test model with random data to see output range"""
    if st.session_state.get("detector"):
        det = st.session_state["detector"]
        model = det["model"]
        frame_len = det["frame_len"]
        target_size = det["target_size"]
        channels = det["channels"]
        
        # Create dummy input matching model expected shape
        dummy_input = np.random.random((1, frame_len, target_size[1], target_size[0], channels))
        
        st.sidebar.subheader("Model Test Results")
        prediction = model.predict(dummy_input, verbose=0)
        st.sidebar.write(f"Raw prediction: {prediction}")
        st.sidebar.write(f"Prediction shape: {prediction.shape}")
        st.sidebar.write(f"Prediction range: [{np.min(prediction):.6f}, {np.max(prediction):.6f}]")
        st.sidebar.write(f"Prediction mean: {np.mean(prediction):.6f}")

def debug_preprocess_frame(frame, target_size, channels):
    """Debug version of preprocess frame"""
    processed = preprocess_frame(frame, target_size, channels)
    if DEBUG_MODE:
        st.sidebar.write(f"Input frame shape: {frame.shape}")
        st.sidebar.write(f"Processed frame shape: {processed.shape}")
        st.sidebar.write(f"Processed range: [{np.min(processed):.3f}, {np.max(processed):.3f}]")
    return processed

if load_model_btn:
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Put your .h5 in src/model/")
    else:
        try:
            st.info("Loading model (this can take a few seconds)...")
            model = load_model(MODEL_PATH)
            
            # DIAGNOSTICS: Print model info
            if DEBUG_MODE:
                st.sidebar.subheader("Model Architecture Info")
                # Get model summary as string
                summary_list = []
                model.summary(print_fn=lambda x: summary_list.append(x))
                st.sidebar.text_area("Model Summary", "\n".join(summary_list), height=200)
                
                # Check the last layer
                last_layer = model.layers[-1]
                st.sidebar.write(f"Last layer: {last_layer.name}")
                st.sidebar.write(f"Last layer type: {type(last_layer).__name__}")
                if hasattr(last_layer, 'activation'):
                    st.sidebar.write(f"Last layer activation: {last_layer.activation.__name__}")
            
            # infer input shape if possible
            inp = model.input_shape  # e.g., (None, 160, 90, 90, 3)
            # default settings
            frame_len = 160
            target_size = (90, 90)
            channels = 3
            if isinstance(inp, tuple) or isinstance(inp, list):
                # try to handle (batch, time, h, w, c)
                if len(inp) == 5:
                    _, t, h, w, c = inp
                    frame_len = t or frame_len
                    # cv2.resize takes (w,h)
                    target_size = (w, h)
                    channels = c or channels
                elif len(inp) == 4:
                    # maybe (batch, h, w, c)
                    _, h, w, c = inp
                    frame_len = 1
                    target_size = (w, h)
                    channels = c or channels
            
            st.sidebar.write("Inferred model input shape:", f"{frame_len} frames, {target_size[1]}x{target_size[0]}, {channels} channels")
            
            st.session_state["detector"] = {
                "model": model,
                "frame_len": int(frame_len),
                "target_size": (int(target_size[0]), int(target_size[1])),
                "channels": int(channels),
            }
            st.success("Model loaded 👍")
            
            # Auto-run dummy test
            test_model_with_dummy_data()
            
        except Exception as e:
            st.error(f"Failed to load model: {e}")

# Add test button
if st.sidebar.button("Test Model with Dummy Data") and st.session_state.get("detector"):
    test_model_with_dummy_data()

# ------- UI & webcam interaction -------
col1, col2 = st.columns([1, 1])
with col1:
    st.header("Webcam")
    start_cam = st.button("Start webcam")
    stop_cam = st.button("Stop webcam")
    webcam_view = st.empty()

with col2:
    st.header("Prediction")
    status = st.empty()
    confidence = st.empty()
    raw_confidence = st.empty()
    clip_link = st.empty()

if start_cam:
    if st.session_state.get("detector") is None:
        st.warning("Please load the model first in the sidebar.")
    else:
        det = st.session_state["detector"]
        model = det["model"]
        frame_len = det["frame_len"]
        target_size = det["target_size"]  # (w,h)
        channels = det["channels"]

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(FRAME_WIDTH))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(FRAME_HEIGHT))

        buf = []
        tmpfile = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(tmpfile.name, fourcc, 20.0, (int(FRAME_WIDTH), int(FRAME_HEIGHT)))

        st.session_state["_cam_running"] = True
        last_pred_time = 0.0
        last_pred = None

        try:
            while st.session_state.get("_cam_running", True):
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera read failed.")
                    break

                writer.write(frame)
                # show frame
                webcam_view.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                # preprocess for model - use debug version
                pre = debug_preprocess_frame(frame, target_size, channels)
                buf.append(pre)
                if len(buf) > frame_len:
                    buf.pop(0)

                now = time.time()
                if len(buf) == frame_len and (now - last_pred_time) >= float(INFERENCE_INTERVAL):
                    X = np.array(buf)
                    X = np.expand_dims(X, axis=0)  # (1, t, h, w, c)
                    try:
                        preds = model.predict(X, verbose=0)
                        
                        # DEBUG: Show raw predictions
                        if DEBUG_MODE:
                            st.sidebar.write(f"Raw prediction shape: {preds.shape}")
                            st.sidebar.write(f"Raw prediction values: {preds}")
                        
                        # Flexible prediction interpretation
                        if preds.ndim == 2:
                            if preds.shape[1] == 2:  # Binary classification [normal, suspicious]
                                prob_susp = float(preds[0, 1])  # Second class probability
                            elif preds.shape[1] == 1:  # Single output (sigmoid)
                                prob_susp = float(preds[0, 0])
                            else:  # Multi-class
                                prob_susp = float(np.max(preds[0]))  # Maximum probability
                        else:
                            # Single value output or other format
                            prob_susp = float(preds[0])
                        
                        # Apply scaling if needed
                        prob_susp_scaled = min(1.0, prob_susp * SCALE_FACTOR)
                        
                        # Display results
                        raw_confidence.markdown(f"Raw Confidence: `{prob_susp:.6f}`")
                        confidence.markdown(f"Confidence (suspicious): `{prob_susp_scaled:.3f}`")
                        
                        label = "🚨 Suspicious" if prob_susp_scaled >= CONF_THRESH else "✅ Normal"
                        status.markdown(f"**{label}**")
                        
                        last_pred_time = now
                        last_pred = (label, prob_susp_scaled)
                        
                    except Exception as e:
                        status.markdown(f"Prediction error: {e}")

                # stop if user pressed Stop
                if stop_cam:
                    st.session_state["_cam_running"] = False
                    break

                # small sleep
                time.sleep(0.01)

        finally:
            cap.release()
            writer.release()
            webcam_view.empty()
            st.info("Webcam stopped.")
            if last_pred and last_pred[0].startswith("🚨"):
                clip_link.markdown(f"🚨 **Suspicious activity detected!** Saved clip: `{tmpfile.name}`")
            else:
                clip_link.markdown(f"Saved test clip: `{tmpfile.name}` (no suspicious activity)")