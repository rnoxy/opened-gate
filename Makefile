
# Download images from ssh server
download:
	@echo "Downloading images from ssh server"
	sh scripts/download_images_from_ssh_server.sh

train:
	@echo "Training model"
	python3 src/opened_gate/train.py

deploy:
	@echo "Creating ONNX model"
	python3 src/opened_gate/deploy.py
	sh scripts/upload_model_to_monitoring_server.sh

docker-run:
    @echo "Running docker container"
    docker run --detach --restart unless-stopped -e OPENEDGATE_CAMERA_URL=http://nvr.arrakis.internal:5000/api/front/latest.jpg -p 5009:5000 -v ./data/06_models/model.onnx:/app/model.onnx:ro --name opened-gate opened-gate:latest

all: download train deploy
	@echo "All done"
