import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-50 feature extractor for both sketch and 3D model views.
    """
    def __init__(self, pretrained=True, feature_dim=2048):
        super(ResNetFeatureExtractor, self).__init__()
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add a projection layer if needed
        if feature_dim != 2048:
            self.projection = nn.Linear(2048, feature_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x):
        # Extract features
        features = self.features(x)
        # Flatten
        features = torch.flatten(features, 1)
        # Project if needed
        features = self.projection(features)
        return features

class SketchCNN(ResNetFeatureExtractor):
    """
    CNN for sketch feature extraction based on ResNet-50.
    """
    def __init__(self, pretrained=True, feature_dim=512):
        super(SketchCNN, self).__init__(pretrained=pretrained, feature_dim=feature_dim)
        
    def forward(self, x):
        # Convert grayscale to RGB if needed
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return super(SketchCNN, self).forward(x)

class ShapeCNN(ResNetFeatureExtractor):
    """
    CNN for 3D shape feature extraction based on ResNet-50.
    Multi-view approach is used where multiple renderings of a 3D model are processed.
    """
    def __init__(self, pretrained=True, feature_dim=512, num_views=12):
        super(ShapeCNN, self).__init__(pretrained=pretrained, feature_dim=feature_dim)
        self.num_views = num_views
        
    def forward(self, x):
        # x shape: [batch_size, num_views, channels, height, width]
        batch_size = x.size(0)
        
        # Reshape to process all views
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        
        # Extract features for all views
        features = super(ShapeCNN, self).forward(x)
        
        # Reshape back to [batch_size, num_views, feature_dim]
        features = features.view(batch_size, self.num_views, -1)
        
        # Max pooling across views
        features, _ = torch.max(features, dim=1)
        
        return features
