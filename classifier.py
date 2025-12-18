# classifier.py
import os
import cv2
import numpy as np
try:
    # tensorflow.keras import (works after `pip install tensorflow`)
    from tensorflow.keras.models import load_model
except Exception as e:
    raise RuntimeError("tensorflow.keras not available. Install tensorflow 2.12+") from e

class TMClassifier:
    """
    Small wrapper around a Keras model exported from Teachable Machine.
    Expects:
     - model_path: .h5 keras model
     - labels_path: labels.txt with one label per line (order must match model)
    """
    def __init__(self, model_path, labels_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        self.model = load_model(model_path, compile=False)
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = [l.strip() for l in f.readlines() if l.strip()]

        # Basic model info
        print(f"[TMClassifier] Loaded model: {model_path}, labels: {self.labels}")

    def predict(self, image):
        """
        image: BGR OpenCV image (numpy array) of the person crop.
        returns: label string (as found in labels.txt), not lowercased here.
        """
        try:
            # Teachable machine typical input size is 224x224 (but check your exported model)
            target_w = 224
            target_h = 224

            img = image.copy()
            # If grayscale -> convert to BGR
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Resize, convert BGR->RGB, normalize to [0,1]
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)  # batch dim

            preds = self.model.predict(img, verbose=0)
            # preds can be (1, n). get index
            idx = int(np.argmax(preds, axis=1)[0])
            label = self.labels[idx] if idx < len(self.labels) else "unknown"
            return label
        except Exception as e:
            # In case of any failure, fallback to "Not_Working" label if exists or "unknown"
            print(f"[TMClassifier] prediction error: {e}")
            fallback = "Not_Working" if "Not_Working" in self.labels else (self.labels[0] if self.labels else "unknown")
            return fallback
