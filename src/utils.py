# src/utils.py
import cv2
import numpy as np

def preprocess_frame(frame, target_size, channels):
    """
    Resize frame to target_size (w,h), convert channels, and normalize to [0,1].
    target_size is (w, h) as used by cv2.resize.
    channels: expected number of channels by the model (1 or 3).
    """
    # cv2.resize expects (width,height) as (w,h) but we commonly pass (w,h) below
    resized = cv2.resize(frame, target_size)

    if channels == 1:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        resized = np.expand_dims(resized, axis=-1)  # h,w,1
    else:
        # convert from BGR (OpenCV) to RGB expected by many models
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    resized = resized.astype("float32") / 255.0
    return resized
