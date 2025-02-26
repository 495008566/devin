# Cross-Domain 3D Model Retrieval Architecture

This document explains the architecture and approach of our cross-domain 3D model retrieval system based on the Domain Disentangled Generative Adversarial Network (DD-GAN) framework.

## System Overview

The system enables retrieving 3D models using sketch queries through domain disentanglement and composition. It follows a zero-shot learning approach, allowing retrieval of 3D models from categories not seen during training.

## Key Components

### 1. Unsupervised Clustering Algorithm

The unsupervised clustering algorithm is used for feature extraction and data preprocessing. It helps organize the feature space and identify patterns in the data without requiring labeled examples.

**Implementation Details:**
- Located in `models/clustering.py`
- Supports K-means and DBSCAN clustering methods
- Used for:
  - Organizing feature space
  - Identifying patterns in unlabeled data
  - Preprocessing data for training

```python
# Example usage
clustering = UnsupervisedClustering(n_clusters=10, method='kmeans')
clustering.fit(features)
cluster_assignments = clustering.predict(new_features)
```

### 2. Deep Metric Learning Algorithm

The deep metric learning algorithm learns a distance metric between sketches and 3D models. It uses triplet loss or contrastive loss to ensure that similar items are close in the embedding space and dissimilar items are far apart.

**Implementation Details:**
- Located in `models/metric_learning.py`
- Implements both triplet loss and contrastive loss
- Used for:
  - Learning a joint embedding space for sketches and 3D models
  - Ensuring cross-domain similarity preservation
  - Enabling effective retrieval

```python
# Example usage
triplet_loss = TripletLoss(margin=1.0)
loss = triplet_loss(anchor, positive, negative)

contrastive_loss = ContrastiveLoss(margin=1.0)
loss = contrastive_loss(x1, x2, y)
```

### 3. DD-GAN Framework

The DD-GAN framework is the core of the system, enabling zero-shot sketch-based 3D model retrieval through domain disentanglement and composition.

#### 3.1 Domain Disentanglement Module

The domain disentanglement module separates content and style information from both sketches and 3D models. Content information is domain-invariant, while style information is domain-specific.

**Implementation Details:**
- Located in `models/domain_disentanglement.py`
- Consists of content encoder and style encoder
- Used for:
  - Extracting domain-invariant content features
  - Extracting domain-specific style features
  - Enabling cross-domain retrieval

```python
# Example usage
disentanglement = DomainDisentanglement(input_dim=512, content_dim=256, style_dim=128)
content, style = disentanglement(features)
```

#### 3.2 Domain Composition Module

The domain composition module combines content and style information to generate features in the target domain. It enables translating features from one domain to another.

**Implementation Details:**
- Located in `models/domain_composition.py`
- Combines content and style features
- Used for:
  - Translating sketch features to 3D model features
  - Translating 3D model features to sketch features
  - Enabling cycle consistency

```python
# Example usage
composition = DomainComposition(content_dim=256, style_dim=128, output_dim=512)
composed_features = composition(content, style)
```

#### 3.3 Transformer Module

The transformer module enhances feature representation through self-attention mechanisms. It captures long-range dependencies and improves the quality of the embeddings.

**Implementation Details:**
- Located in `models/transformer.py`
- Implements self-attention and transformer blocks
- Used for:
  - Enhancing feature representation
  - Capturing long-range dependencies
  - Improving embedding quality

```python
# Example usage
transformer = TransformerModule(dim=256, depth=2, heads=8)
enhanced_features = transformer(features)
```

### 4. ResNet-50 Backbone

ResNet-50 is used as the CNN backbone for both SketchCNN and ShapeCNN feature extraction. It provides a strong foundation for extracting meaningful features from both sketches and 3D model renderings.

**Implementation Details:**
- Located in `models/resnet.py`
- Uses pretrained ResNet-50 with custom modifications
- Used for:
  - Extracting features from sketches (SketchCNN)
  - Extracting features from 3D model renderings (ShapeCNN)
  - Providing a strong foundation for the entire system

```python
# Example usage
sketch_cnn = SketchCNN(feature_dim=512)
sketch_features = sketch_cnn(sketches)

shape_cnn = ShapeCNN(feature_dim=512, num_views=12)
shape_features = shape_cnn(shapes)
```

## Data Flow

1. **Feature Extraction**: Sketches and 3D models are processed through SketchCNN and ShapeCNN respectively to extract features.
2. **Domain Disentanglement**: Features are disentangled into content and style components.
3. **Feature Enhancement**: Content features are enhanced using the transformer module.
4. **Cross-Domain Translation**: Content and style features are combined to translate features across domains.
5. **Retrieval**: During inference, sketch query features are extracted and used to retrieve similar 3D models.

## Datasets

The system is designed to work with the following datasets:

### TU-Berlin Sketch Dataset
- 20,000 sketches of 250 object categories
- Hand-drawn sketches with diverse categories
- Used for training sketch feature extraction

### ModelNet40 3D Model Dataset
- 12,311 3D CAD models from 40 categories
- Clean, manually verified 3D models
- Used for training 3D model feature extraction

### SHREC Benchmark Datasets
- Specifically designed for sketch-based 3D model retrieval
- Provides evaluation protocols and metrics
- Used for evaluation and benchmarking

## References

This implementation is based on the following papers and open-source implementations:

1. "Domain Disentangled Generative Adversarial Network for Zero-Shot Sketch-Based 3D Shape Retrieval" by Rui Xu, Zongyan Han, Le Hui, Jianjun Qian, and Jin Xie.
2. [Luffy212/DD-GAN](https://github.com/Luffy212/DD-GAN): Original implementation of DD-GAN for zero-shot sketch-based 3D shape retrieval.
3. [ddongcui/Awsome-Sketch-based-3dshape-retrieval](https://github.com/ddongcui/Awsome-Sketch-based-3dshape-retrieval): Collection of resources for sketch-based 3D shape retrieval.
