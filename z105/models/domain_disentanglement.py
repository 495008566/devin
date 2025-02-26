import torch
import torch.nn as nn

class ContentEncoder(nn.Module):
    """
    Content encoder for domain disentanglement.
    Extracts domain-invariant content features.
    """
    def __init__(self, input_dim=512, content_dim=256):
        super(ContentEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, content_dim),
            nn.BatchNorm1d(content_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class StyleEncoder(nn.Module):
    """
    Style encoder for domain disentanglement.
    Extracts domain-specific style features.
    """
    def __init__(self, input_dim=512, style_dim=128):
        super(StyleEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, style_dim),
            nn.BatchNorm1d(style_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class DomainDisentanglement(nn.Module):
    """
    Domain disentanglement module that separates content and style.
    """
    def __init__(self, input_dim=512, content_dim=256, style_dim=128):
        super(DomainDisentanglement, self).__init__()
        
        self.content_encoder = ContentEncoder(input_dim, content_dim)
        self.style_encoder = StyleEncoder(input_dim, style_dim)
    
    def forward(self, x):
        content = self.content_encoder(x)
        style = self.style_encoder(x)
        return content, style
