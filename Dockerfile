# Dockerfile to run a simple Flask application on Python 3.10 runtime

# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Copy the current directory contents into the container at /app
COPY app /app

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install flask onnxruntime numpy Pillow

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]
