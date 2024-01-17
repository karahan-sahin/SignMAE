# SignMAE - Transformer-based Masked AutoEncoders for Sign Language Recognition/Translation

## Introduction

This project presents a state-of-the-art sign language recognition system employing a transformer-based video encoder architecture. Our system uniquely encodes pose topology by embedding different adjacency matrix definitions, including cosine, Euclidean, and Mahalanobis-based distance adjacencies, as well as absolute axis differences. This approach allows for a more nuanced and accurate interpretation of sign language gestures.

## System Overview

The core of our system is the transformer-based video encoder which processes sequential frames of sign language gestures. By embedding various adjacency matrices, the system captures the complex spatial relationships between different parts of the body involved in sign language communication.

## Key Features

Transformer-Based Video Encoder: Leverages the power of transformers for efficient video processing.
Multiple Adjacency Matrix Embeddings: 
- Incorporates cosine, Euclidean, and Mahalanobis distances, plus absolute axis differences for robust pose recognition.
- High Accuracy: Designed to achieve state-of-the-art accuracy in sign language recognition.

## Requirements

- Python 3.x
- PyTorch 1.x
- Additional dependencies listed in requirements.txt

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/karahan-sahin/SignMAE.git
cd SignMAE
python3 -m venv signmae_env
source signmae_env/bin/activate
pip install -r requirements.txt
```


## Usage

### Training

```
lib
|
|--...
|--run_asl_translation_training.py
|--run_bsign22k_training.py
|--run_autsl_training.py

```

To use the system, first give desired parameters for the training then run:

```bash
python3 lib/<name-the-training-script>

```

Replace <name-the-training-script> with the appropriate training data is available

## Contributing

Contributions to this project are welcome. Please follow these steps:

* Fork the repository.
* Create a new branch for your feature (git checkout -b feature/YourFeature).
* Commit your changes (git commit -am 'Add some YourFeature').
* Push to the branch (git push origin feature/YourFeature).
* Open a new Pull Request.


## License

MIT License


## TODO:
- [ ] Add Mahalanobis distance
- [ ] Add push to Huggingface
- [ ] Add evaluation scripts

## DONE:
- [x] Add documentation for running
- [x] Refactor code
- [x] Add AUTSL Training
- [x] Add ASL FINGERSPELLING Training
- [x] Fix Translation Accumulation Problem