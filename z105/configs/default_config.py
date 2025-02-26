"""
Default configuration for the DD-GAN model.
"""

# Model parameters
MODEL_CONFIG = {
    'feature_dim': 512,
    'content_dim': 256,
    'style_dim': 128,
    'transformer_depth': 2,
    'transformer_heads': 8,
    'transformer_dropout': 0.1
}

# Training parameters
TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'lr': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,
    'lambda_triplet': 1.0,
    'lambda_recon': 10.0,
    'lambda_cycle': 10.0
}

# Data parameters
DATA_CONFIG = {
    'sketch_dataset': 'tu_berlin',
    'shape_dataset': 'modelnet40',
    'num_views': 12
}
