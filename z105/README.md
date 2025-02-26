# Cross-Domain 3D Model Retrieval

This project implements a simplified version of a cross-domain 3D model retrieval system based on the Domain Disentangled Generative Adversarial Network (DD-GAN) framework. The system enables retrieving 3D models using sketch queries through domain disentanglement and composition.

## Key Components

1. **Unsupervised Clustering Algorithm**
   - Feature extraction and data preprocessing
   - K-means and DBSCAN implementations

2. **Deep Metric Learning Algorithm**
   - Triplet loss implementation
   - Contrastive loss implementation

3. **DD-GAN Framework**
   - Domain disentanglement module
   - Domain composition module
   - Transformer module

4. **ResNet-50 Backbone**
   - SketchCNN for sketch feature extraction
   - ShapeCNN for 3D model feature extraction

## Project Structure

```
z105/
├── configs/
│   └── default_config.py
├── data/
│   ├── __init__.py
│   └── data_loader.py
├── models/
│   ├── __init__.py
│   ├── clustering.py
│   ├── dd_gan.py
│   ├── domain_composition.py
│   ├── domain_disentanglement.py
│   ├── metric_learning.py
│   ├── resnet.py
│   └── transformer.py
├── utils/
│   ├── __init__.py
│   └── evaluation.py
├── main.py
└── README.md
```

## Usage

To train the model:

```bash
python main.py --data_dir ./data --sketch_dataset tu_berlin --shape_dataset modelnet40
```

## Datasets

The system is designed to work with the following datasets:
- TU-Berlin Sketch Dataset
- ModelNet40 3D Model Dataset
- SHREC Benchmark Datasets

## References

This implementation is based on the Domain Disentangled Generative Adversarial Network for Zero-Shot Sketch-Based 3D Shape Retrieval paper and related open-source implementations.
