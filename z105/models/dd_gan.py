import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import SketchCNN, ShapeCNN
from models.domain_disentanglement import DomainDisentanglement
from models.domain_composition import DomainComposition
from models.transformer import TransformerModule

class DD_GAN(nn.Module):
    """
    Domain Disentangled Generative Adversarial Network for Zero-Shot Sketch-Based 3D Shape Retrieval.
    Simplified implementation based on the paper.
    """
    def __init__(self, feature_dim=512, content_dim=256, style_dim=128):
        super(DD_GAN, self).__init__()
        
        # Feature extractors
        self.sketch_cnn = SketchCNN(feature_dim=feature_dim)
        self.shape_cnn = ShapeCNN(feature_dim=feature_dim)
        
        # Domain disentanglement
        self.sketch_disentanglement = DomainDisentanglement(feature_dim, content_dim, style_dim)
        self.shape_disentanglement = DomainDisentanglement(feature_dim, content_dim, style_dim)
        
        # Domain composition
        self.sketch_to_shape_composition = DomainComposition(content_dim, style_dim, feature_dim)
        self.shape_to_sketch_composition = DomainComposition(content_dim, style_dim, feature_dim)
        
        # Transformer module
        self.transformer = TransformerModule(content_dim)
        
    def forward(self, sketches, shapes=None, mode='train'):
        """
        Forward pass through the DD-GAN model.
        
        Args:
            sketches: Batch of sketch images
            shapes: Batch of 3D shape renderings (multi-view)
            mode: 'train' for training, 'sketch_query' for retrieval using sketch query
            
        Returns:
            Dictionary containing various outputs depending on the mode
        """
        outputs = {}
        
        # Extract sketch features
        sketch_features = self.sketch_cnn(sketches)
        outputs['sketch_features'] = sketch_features
        
        # Disentangle sketch features
        sketch_content, sketch_style = self.sketch_disentanglement(sketch_features)
        outputs['sketch_content'] = sketch_content
        outputs['sketch_style'] = sketch_style
        
        # Apply transformer to sketch content
        sketch_content_transformed = self.transformer(sketch_content)
        outputs['sketch_content_transformed'] = sketch_content_transformed
        
        if mode == 'train' and shapes is not None:
            # Extract shape features
            shape_features = self.shape_cnn(shapes)
            outputs['shape_features'] = shape_features
            
            # Disentangle shape features
            shape_content, shape_style = self.shape_disentanglement(shape_features)
            outputs['shape_content'] = shape_content
            outputs['shape_style'] = shape_style
            
            # Apply transformer to shape content
            shape_content_transformed = self.transformer(shape_content)
            outputs['shape_content_transformed'] = shape_content_transformed
            
            # Cross-domain composition
            sketch_to_shape = self.sketch_to_shape_composition(sketch_content_transformed, shape_style)
            shape_to_sketch = self.shape_to_sketch_composition(shape_content_transformed, sketch_style)
            outputs['sketch_to_shape'] = sketch_to_shape
            outputs['shape_to_sketch'] = shape_to_sketch
        
        elif mode == 'sketch_query':
            # For retrieval, we only need the transformed sketch content
            outputs['query_embedding'] = sketch_content_transformed
            
        return outputs
