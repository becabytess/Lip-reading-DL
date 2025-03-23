# LipNet Implementation

A custom implementation of the research paper "LipNet: End-to-End Sentence-level Lipreading" with PyTorch, constructed without using high-level APIs.

## Overview

This repository contains my reproduction of the LipNet architecture as described in the original research paper [LipNet: End-to-End Sentence-level Lipreading](https://arxiv.org/abs/1611.01599) by Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, and Nando de Freitas. The model features an impressive 20 million trainable parameters, making it a substantial deep learning system capable of processing complex visual speech patterns. My implementation combines 3D convolutional neural networks and a custom LSTM implementation to recognize spoken words from video frames of lip movements.

![Modifed LipNet Architecture](arc.png)

## Dataset

The model is trained on the GRID Audio-Visual Speech Corpus, a large multitalker audiovisual dataset designed for speech perception studies. The dataset consists of high-quality audio and video recordings of 1000 sentences spoken by 34 different speakers.

### Downloading the Dataset

The dataset is available on Zenodo and can be downloaded from:
[GRID Audio-Visual Speech Corpus](https://zenodo.org/records/3625687)

You'll need to download:
- `alignments.zip` - Contains word-level time alignments for each speaker
- Video files for each speaker (`s1.zip`, `s2.zip`, etc.)

After downloading, extract the files and organize them as follows:
```
datasets/
├── videos/
│   ├── s1/
│   ├── s2/
│   └── ...
└── alignments/
    ├── s1/
    ├── s2/
    └── ...
```

Note: The dataset is quite large (approximately 16GB), so ensure you have sufficient disk space before downloading.

## Research Implementation

This project faithfully reproduces the architecture described in the paper while implementing the LSTM components from scratch. Key aspects of the original research that have been maintained:

- End-to-end trainable neural network
- Spatiotemporal convolutions for visual feature extraction
- Bidirectional recurrence for sequential modeling
- CTC loss for sequence prediction without explicit alignment

I made the deliberate choice to implement LSTM cells instead of the GRU cells used in the original paper, as I wanted to explore the full capabilities of LSTM's memory mechanisms in the context of lipreading. This modification maintains the bidirectional sequence processing while potentially offering enhanced memory retention for longer sequences.

## Model Architecture

The architecture consists of:

- **3D Convolutional Layers**: Three Conv3D layers with batch normalization, ReLU activation, and max pooling to extract spatial-temporal features from video frames, as specified in the paper.
- **Custom Bidirectional LSTM**: Implemented from scratch without using PyTorch's built-in RNN modules.
  - Forward and backward passes through the sequence
  - Manually implemented LSTM gates (forget, input, candidate, output)
  - Cell state and hidden state management
- **Classifier**: Linear layer to map to vocabulary output with softmax activation

## Training Process

- Uses CTC (Connectionist Temporal Classification) loss for sequence prediction, as in the original paper
- Adam optimizer with learning rate scheduling
- Gradient accumulation for effective batch processing
- Memory management techniques for large model training

## Requirements

- PyTorch
- OpenCV
- NumPy

## Directory Structure

```
├── datasets/
│   ├── videos/       # Video files organized by speaker
│   └── alignments/   # Text alignment files
├── main.py           # Main training script
├── explore/
│   └── explore.ipynb # Jupyter notebook with detailed training process
└── README.md         # Project documentation
```

## Usage

1. Prepare your dataset in the required format
2. Configure paths in `main.py`
3. Run training:
```bash
python main.py
```

## Implementation Details

This implementation avoids high-level PyTorch APIs for RNN components, instead building them from scratch using basic tensor operations. Key features:

- Custom LSTM cell implementation following the paper's specifications
- Bidirectional sequence processing as described in the research
- Memory-efficient training with gradient accumulation
- Robust error handling to prevent NaN values during training

### Data Preprocessing

The implementation includes specialized data handling:
- Video frames are extracted and converted to RGB format
- Frames are normalized to [0,1] range
- Alignments are processed to extract text and convert to numerical indices 
- Custom collate function ensures all training samples have the same number of frames (75)
- A specialized vocabulary mapping for the 28 characters (26 letters, space, and the CTC blank token)

## Results

The model checkpoints are saved after each epoch, allowing for evaluation of training progress and eventual inference on new lip-reading videos. The performance metrics aim to match those reported in the original research.

## Acknowledgements

This implementation is based on the original LipNet paper: ["LipNet: End-to-End Sentence-level Lipreading"](https://arxiv.org/abs/1611.01599) by Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, and Nando de Freitas. 
