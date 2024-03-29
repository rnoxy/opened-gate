# This is very simple Flask app that serves a model loaded from data/06_models/model.ckpt
# The model is a very simple Neural Network that takes the image input and outputs the prediction
import datetime
import os

import numpy as np
import onnxruntime
from PIL import Image
from flask import Flask, jsonify

# Load the model (using lightning)
MODEL_PATH = "model.onnx"
ort_session = onnxruntime.InferenceSession(MODEL_PATH)

# Create the Flask app
app = Flask(__name__)


# This is the endpoint that return the prediction for the latest camera image
@app.route("/")
def predict():
    import urllib.request

    # Download the latest image
    urllib.request.urlretrieve(
        os.getenv("OPENEDGATE_CAMERA_URL"),
        "latest.jpg",
    )

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

    # save the images with prob <= 0.8
    if prob <= 0.8:
        print(f"Saving image with prob {prob}")
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filepath = f"images/{prediction_class}_{timestamp}_{prob:.2f}.jpg"
        image_raw.save(filepath)

    # Return the prediction
    return jsonify(
        {
            "prediction": prediction_class.tolist(),
            "probability": softmax[prediction_class].tolist(),
            "all_probabilities": softmax.tolist(),
        }
    )


if __name__ == "__main__":
    app.run(
        host=os.getenv("OPENEDGATE_HOST", "0.0.0.0"),
        port=int(os.getenv("OPENEDGATE_PORT", 5000)),
    )
