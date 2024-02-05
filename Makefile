
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
