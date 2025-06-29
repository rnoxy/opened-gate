# This is very simple Flask app that serves a model loaded from data/06_models/model.ckpt
# The model is a very simple Neural Network that takes the image input and outputs the prediction
import datetime
import logging
import os
import urllib.request

import numpy as np
import onnxruntime
from PIL import Image
from flask import Flask, jsonify

# ----------------------------------------
# Configuration and Logging
# ----------------------------------------
MODEL_PATH = "model.onnx"
IMAGE_SAVE_DIR = "images"
CAMERA_URL = os.getenv("OPENEDGATE_CAMERA_URL")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------
# Ensure required directories exist
# ----------------------------------------
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)


# ----------------------------------------
# Load the ONNX model
# ----------------------------------------
try:
    ort_session = onnxruntime.InferenceSession(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.exception("Failed to load the model")
    raise RuntimeError(f"Failed to load the model: {e}")


# ----------------------------------------
# Create the Flask app
# ----------------------------------------
app = Flask(__name__)


# ----------------------------------------
# Health check endpoint
# ----------------------------------------
@app.route("/health", methods=["GET"])
def health():
    try:
        ort_session.get_inputs()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "details": str(e)}), 500


# ----------------------------------------
# Prediction endpoint
# ----------------------------------------
@app.route("/", methods=["GET"])
def predict():
    if not CAMERA_URL:
        logger.error("OPENEDGATE_CAMERA_URL environment variable not set")
        return jsonify({"error": "Camera URL not set"}), 500

    try:
        # Download latest image
        urllib.request.urlretrieve(CAMERA_URL, "latest.jpg")
    except Exception as e:
        logger.exception("Failed to download image")
        return jsonify({"error": f"Failed to download image: {e}"}), 500

    try:
        # Load and preprocess image
        # Load the image
        image_raw = Image.open("latest.jpg")

        # Convert to numpy array
        image = image_raw.resize((224, 224))
        image = np.array(image)

        # Normalize the image
        # transforms.Normalize(
        #     #             mean=[0.485, 0.456, 0.406],  # RGB
        #     #             std=[0.229, 0.224, 0.225],  # RGB
        #     #         ),
        image = image / 255.0
        image = image - np.array([0.485, 0.456, 0.406])
        image = image / np.array([0.229, 0.224, 0.225])
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        # Make the prediction
        ort_inputs = {ort_session.get_inputs()[0].name: image}
        ort_outs = ort_session.run(None, ort_inputs)
        predictions = ort_outs[0]
        prediction = predictions[0]

        prediction_class = np.argmax(prediction)

        # calculate probability
        softmax = np.exp(prediction) / np.sum(np.exp(prediction))
        prob = softmax[prediction_class]

        # Save uncertain predictions
        if prob <= 0.8:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{prediction_class}_{timestamp}_{prob:.2f}.jpg"
            filepath = os.path.join(IMAGE_SAVE_DIR, filename)
            image_raw.save(filepath)
            logger.info(f"Saved low-confidence image to {filepath}")

        return jsonify(
            {
                "prediction": prediction_class,
                "probability": prob,
                "all_probabilities": softmax.tolist(),
            }
        )
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {e}"}), 500


# ----------------------------------------
# App entry point
# ----------------------------------------
if __name__ == "__main__":
    app.run(
        host=os.getenv("OPENEDGATE_HOST", "0.0.0.0"),
        port=int(os.getenv("OPENEDGATE_PORT", 5000)),
    )
