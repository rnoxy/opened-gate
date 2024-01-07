# This is very simple Flask app that serves a model loaded from data/06_models/model.ckpt
# The model is a very simple Neural Network that takes the image input and outputs the prediction
import datetime

import numpy as np
import onnxruntime
from PIL import Image
from flask import Flask, jsonify
from torchvision import transforms

# Load the model (using lightning)
MODEL_PATH = "model.onnx"
ort_session = onnxruntime.InferenceSession(MODEL_PATH)

# Create the Flask app
app = Flask(__name__)


@app.route("/")
def hello():
    return "Run the /predict endpoint to get the prediction"


# This is the endpoint that return the prediction for the latest camera image
@app.route("/predict")
def predict():
    # Download the image from http://192.168.3.10:5000/api/front/latest.jpg
    # use urllib.request.urlretrieve to download the image
    import urllib.request

    urllib.request.urlretrieve(
        "http://192.168.3.10:5000/api/front/latest.jpg", "latest.jpg"
    )

    # Load the image
    image_raw = Image.open("latest.jpg")

    # Preprocess the image
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # RGB
                std=[0.229, 0.224, 0.225],  # RGB
            ),
        ]
    )
    image = transform(image_raw)
    image = image.unsqueeze(0)

    # Make the prediction
    ort_inputs = {ort_session.get_inputs()[0].name: image.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    predictions = ort_outs[0]
    prediction = predictions[0]

    prediction_class = np.argmax(prediction)

    # calculate probability
    softmax = np.exp(prediction) / np.sum(np.exp(prediction))
    prob = softmax[prediction_class]

    # save the images with prob <= 0.8 and all images with prediction == 1
    if prob <= 0.8 or prediction_class == 1:
        print(f"Saving image with prob {prob}")
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filepath = f"images/{prediction_class}_{timestamp}_{prob:.2f}.jpg"
        image_raw.save(filepath)

    # Return the prediction
    return jsonify(
        {
            "prediction": prediction_class.tolist(),
            "probability": softmax[prediction_class].tolist(),
        }
    )


if __name__ == "__main__":
    app.run()
