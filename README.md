# opened-gate
Simple CV project to check if the gate (visible from camera) is opened
<table><tr><td width="50%"><img alt="image" src="https://github.com/rnoxy/opened-gate/assets/12031664/6b2b7f7b-8b4a-4389-b683-dc8d83ac843c"></td><td width="50%"><img alt="image" src="https://github.com/rnoxy/opened-gate/assets/12031664/cbbb3d34-5d82-426d-af85-a8a3d4bff7a0"></td></tr></table>

## Train the model
The following Python script will train the model with [lightning](https://lightning.ai/) framework
and save it to `data/06_models/model.ckpt`.
```shell
python3 src/opened_gate/train.py
```

## Deploy the model
Run the following script to export the model to ONNX format and save it to `app/model.onnx`.
```shell
python3 src/opened_gate/deploy.py
```

## Build the app
Run the following script to build the app.
```shell
docker build -t opened-gate .
```

## Run the app
Run the following script to run the app.
```shell
docker run --restart unless-stopped -e OPENEDGATE_CAMERA_URL=<URL> -p <PORT>:5000 -v ./data/06_models/model.onnx:/app/model.onnx:ro opened-gate:latest
```
where `<URL>` is the URL to fetch the camera image from 
and `<PORT>` is the port on which the app will be available (e.g. `5000`).
