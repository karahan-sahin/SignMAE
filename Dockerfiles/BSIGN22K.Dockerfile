# Use a base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the Python script to the working directory
COPY run_bsign22k_training.py .

# Download tar file from the source
RUN wget https://www-i6.informatik.rwth-aachen.de/~koller/1bsign22k/data.tar.gz

# Copy the tar file to the working directory
COPY data.tar.gz .

# Extract the contents of the tar file to the data/bsign22k directory
RUN tar -xvf data.tar.gz -C data/bsign22k

# Install any dependencies required by the Python script
RUN pip install -r requirements.txt

# Run the Python script
CMD ["python", "lib/run_bsign22k_training.py"]
