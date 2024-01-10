# opened-gate
Simple CV project to check if the gate (visible from camera) is opened

## Train the model
The following Python script will train the model with [lightning](https://lightning.ai/) framework
and save it to `data/06_models/model.ckpt`.
and save it to `data/06_models/model.ckpt`
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
docker run -e OPENEDGATE_CAMERA_URL=<URL> -p <PORT>:5000 opened-gate:latest
```
where `<URL>` is the URL to fetch the camera image from 
and `<PORT>` is the port on which the app will be available (e.g. `5000`).
