# Script to download the camera images from the SSH server.
# The script is using SSH to connect to the server and download the images.
# The script is using the `rsync` command to download the images.
# Warning: The script is using the `rsync` command with the `--remove-source-files` option.
#          This option will remove the source files after the download.
#          This option is used to avoid downloading the same images again.

#
# The script is using the following environment variables:
#   - DEST_LOCATION: The destination location where the images will be downloaded.
#   - SSH_SERVER: The SSH server to connect to.
#   - SSH_USER: The SSH user to use to connect to the server.
#   - SOURCE_LOCATION: The source location where the images are located.
#   - OPENED_IMAGES_PATTERN: The pattern to use to download the opened images.
#   - CLOSED_IMAGES_PATTERN: The pattern to use to download the closed images.

# The opened and closed images will be downloaded the following locations:
#  - DEST_LOCATION/opened
#  - DEST_LOCATION/closed

# The script is using the following commands:
#   - rsync: To download the images from the server.
#   - ssh: To connect to the server.

DEST_LOCATION=data/01_raw/camera-images
SSH_SERVER=ha
SSH_USER=hassio
SOURCE_LOCATION=/config/camera-images
OPENED_IMAGES_PATTERN="opened_*.jpg"
CLOSED_IMAGES_PATTERN="closed_*.jpg"

# Download the opened images.
rsync -avz --remove-source-files -e ssh ${SSH_USER}@${SSH_SERVER}:${SOURCE_LOCATION}/${OPENED_IMAGES_PATTERN} ${DEST_LOCATION}/opened

# Download the closed images.
rsync -avz --remove-source-files -e ssh ${SSH_USER}@${SSH_SERVER}:${SOURCE_LOCATION}/${CLOSED_IMAGES_PATTERN} ${DEST_LOCATION}/closed
