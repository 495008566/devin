# Cross-Domain 3D Model Retrieval Usage Guide

This guide explains how to use the cross-domain 3D model retrieval system for training and inference.

## Prerequisites

- Python 3.8+
- PyTorch 1.7+
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/495008566/devin.git
cd devin/z105
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

### Sketch Dataset

The system is designed to work with the TU-Berlin Sketch Dataset:

1. Download the TU-Berlin Sketch Dataset from [http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/)
2. Extract the dataset to `data/tu_berlin`

### 3D Model Dataset

The system is designed to work with the ModelNet40 Dataset:

1. Download the ModelNet40 Dataset from [https://modelnet.cs.princeton.edu/](https://modelnet.cs.princeton.edu/)
2. Extract the dataset to `data/modelnet40`

## Training

To train the model with default parameters:

```bash
python main.py
```

To customize training parameters:

```bash
python main.py --feature_dim 512 --content_dim 256 --style_dim 128 --batch_size 32 --epochs 100 --lr 0.0002
```

## Evaluation

To evaluate the model on the SHREC benchmark:

```bash
python main.py --mode evaluate --checkpoint path/to/checkpoint.pth
```

## Inference

To retrieve 3D models using a sketch query:

```bash
python main.py --mode retrieve --sketch_path path/to/sketch.png --top_k 10
```

## Configuration

The system can be configured using the configuration files in the `configs` directory:

- `default_config.py`: Default configuration for the DD-GAN model

To use a custom configuration:

```bash
python main.py --config configs/custom_config.py
```

## Examples

### Training Example

```bash
python main.py --data_dir ./data --sketch_dataset tu_berlin --shape_dataset modelnet40 --batch_size 32 --epochs 100
```

### Retrieval Example

```bash
python main.py --mode retrieve --sketch_path examples/airplane.png --top_k 10 --checkpoint checkpoints/best_model.pth
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model complexity
   ```bash
   python main.py --batch_size 16 --feature_dim 256
   ```

2. **Dataset Not Found**: Ensure datasets are downloaded and extracted to the correct directories
   ```bash
   ls data/tu_berlin
   ls data/modelnet40
   ```

3. **Training Instability**: Adjust learning rate or loss weights
   ```bash
   python main.py --lr 0.0001 --lambda_triplet 0.5 --lambda_recon 5.0
   ```

## References

For more information, refer to the architecture documentation in `docs/architecture.md`.
