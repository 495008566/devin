# Cross-Domain 3D Model Retrieval

This project implements a cross-domain 3D model retrieval system based on the Domain Disentangled Generative Adversarial Network (DD-GAN) framework. The system enables retrieving 3D models using sketch queries through domain disentanglement and composition.

## Features

- **Unsupervised Clustering Algorithm** for feature extraction and data preprocessing
- **Deep Metric Learning Algorithm** using triplet loss and contrastive loss
- **DD-GAN Framework** with:
  - Domain disentanglement module
  - Domain composition module
  - Transformer module
- **ResNet-50** as the CNN backbone for ShapeCNN and SketchCNN feature extraction

## Project Structure

```
z105/
├── configs/              # Configuration files
├── data/                 # Data loading and preprocessing
├── docs/                 # Documentation
├── models/               # Model implementations
│   ├── clustering.py     # Unsupervised clustering algorithms
│   ├── dd_gan.py         # Main DD-GAN implementation
│   ├── domain_composition.py    # Domain composition module
│   ├── domain_disentanglement.py # Domain disentanglement module
│   ├── metric_learning.py # Triplet and contrastive loss
│   ├── resnet.py         # ResNet-50 backbone
│   └── transformer.py    # Transformer module
├── utils/                # Utility functions
│   ├── evaluation.py     # Evaluation metrics
│   └── visualization.py  # Visualization tools
├── visualizations/       # Generated visualizations
├── results/              # Inference results
├── train.py              # Training script
├── inference.py          # Inference script
├── demo.py               # Demo script for generating sample results
└── requirements.txt      # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

To train the model:

```bash
python train.py --sketch_data path/to/sketch_dataset --model_data path/to/model_dataset
```

Training parameters:
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of epochs to train (default: 100)
- `--lr`: Learning rate (default: 0.0002)
- `--lambda_triplet`: Weight for triplet loss (default: 1.0)
- `--margin`: Margin for triplet loss (default: 0.3)
- `--checkpoint_dir`: Directory to save checkpoints (default: 'checkpoints')
- `--vis_dir`: Directory to save visualizations (default: 'visualizations')

## Inference

To perform inference with a trained model:

```bash
python inference.py --model_path path/to/model_checkpoint --query_sketch path/to/sketch_image
```

Inference parameters:
- `--model_path`: Path to trained model checkpoint (required)
- `--query_sketch`: Path to query sketch image (optional)
- `--top_k`: Number of top retrievals to show (default: 5)
- `--output_dir`: Directory to save results (default: 'results')

## Demo

To generate sample visualizations without training:

```bash
python demo.py
```

This will create:
1. Training curves visualization
2. Feature space visualization
3. Retrieval results visualization

## Visualizations

The project includes comprehensive visualization tools:

1. **Training Metrics Visualization**
   - Loss curves
   - Mean Average Precision (mAP) curves

2. **Feature Space Visualization**
   - t-SNE visualization of sketch features
   - t-SNE visualization of model features
   - Combined feature space visualization

3. **Retrieval Results Visualization**
   - Query sketches with corresponding retrieved 3D models
   - Color-coded borders indicating correct/incorrect retrievals

## Datasets

The system is designed to work with the following datasets:
- TU-Berlin Sketch Dataset
- ModelNet40 3D Model Dataset
- SHREC Benchmark Datasets

## References

This implementation is based on the following papers and open-source implementations:
- "Domain Disentangled Generative Adversarial Network for Zero-Shot Sketch-Based 3D Shape Retrieval"
- [Luffy212/DD-GAN](https://github.com/Luffy212/DD-GAN)
- [ddongcui/Awsome-Sketch-based-3dshape-retrieval](https://github.com/ddongcui/Awsome-Sketch-based-3dshape-retrieval)
