# Script to upload the model.onnx file to the monitoring server.

DEST_LOCATION=opened-gate/data/06_models/model.onnx
SSH_SERVER=monitoring
SOURCE_LOCATION=data/06_models/model.onnx

# Upload the model.onnx file to the monitoring server
rsync -avz -e ssh $SOURCE_LOCATION $SSH_SERVER:$DEST_LOCATION
