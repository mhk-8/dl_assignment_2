
# DA6401: Introduction to Deep Learning - Assignment 2
**Name:** Majji Hari Krishna
**Roll Number:** CS25M028

## Important Links
* **Weights & Biases (W&B) Public Report:** [Report Link](https://wandb.ai/majjiharikrishna2002-iitm-ac-in/da6401-assignment2/reports/DA6401-Assignment-2-Report--VmlldzoxNjQ0MjkwNw?accessToken=jl922e9j3dfltiyiodopqrnprgno0kk12p2v7wntpgo14tflvg204fwz3kvfos6w)
* **GitHub Repository:**[Public Repository](https://github.com/mhk-8/dl_assignment_2)

## Project Overview
This repository contains the implementation of a cohesive, multi-stage Visual Perception Pipeline built using PyTorch.
Moving beyond isolated tasks, this unified system is capable of simultaneously classifying, localizing, and segmenting subjects using the Oxford-IIIT Pet Dataset. 

## Architecture & Features
The pipeline integrates three primary computer vision tasks into a single multi-task architecture:

* **Classification (Task 1):** A VGG11 backbone implemented entirely from scratch using standard `torch.nn` modules. 
The architecture is modernized with Batch Normalization layers and a custom-built Dropout layer to prevent overfitting. 

* **Object Localization (Task 2):** Utilizing the VGG11 encoder, a custom regression head predicts four continuous bounding box coordinates ($X_{center}$, $Y_{center}$, $width$, $height$). 
The network is optimized using a custom Intersection over Union (IoU) module.

* **Semantic Segmentation (Task 3):** A U-Net style expansive path geometrically mirrors the VGG11 encoder.
This symmetric decoder progressively rebuilds spatial resolution using Transposed Convolutions and fuses features via concatenation with the encoder's spatial maps.

* **Unified Multi-Task Learning (Task 4):** A single `forward(self, x)` pass efficiently branches from the shared backbone, simultaneously yielding 37-class breed logits, bounding box coordinate regression, and a dense pixel-wise segmentation map.

## Repository Structure
As per the submission guidelines, the local checkpoints have been removed, and the final repository follows this structure:

```text
.
├── checkpoints
│   └── checkpoints.md
├── data
│   └── pets_dataset.py
├── inference.py
├── losses
│   ├── __init__.py
│   └── iou_loss.py
├── models
│   ├── classification.py
│   ├── __init__.py
│   ├── layers.py
│   ├── localization.py
│   ├── multitask.py
│   ├── segmentation.py
│   └── vgg11.py
├── README.md
├── requirements.txt
└── train.py
```

## Setup & Model Initialization 
To respect storage limits and GitHub constraints, the pre-trained weights for this project are securely hosted on Google Drive. 

Upon initializing the `MultiTaskPerceptionModel`, the required `.pth` files are automatically downloaded into the `checkpoints/` directory using `gdown`. 

**Dependencies:**
Install the required libraries before running the pipeline:
```bash
pip install -r requirements.txt
```

**Running Inference:**
```bash
python inference.py
```

