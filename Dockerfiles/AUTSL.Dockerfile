# Use a base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the Python script to the working directory
COPY run_autsl_training.py .

# Download tar file from the source
RUN wget https://www-i6.informatik.rwth-aachen.de/~koller/1autsl/mmpose-full.tar.gz

# Copy the tar file to the working directory
COPY mmpose-full.tar.gz .

# Extract the contents of the tar file to the data/autsl directory
RUN tar -xvf mmpose-full.tar.gz -C data/autsl

# Install any dependencies required by the Python script
RUN pip install -r requirements.txt

# Run the Python script
CMD ["python", "lib/run_autsl_training.py"]
