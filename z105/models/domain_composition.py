import torch
import torch.nn as nn

class DomainComposition(nn.Module):
    """
    Domain composition module that combines content and style features.
    """
    def __init__(self, content_dim=256, style_dim=128, output_dim=512):
        super(DomainComposition, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(content_dim + style_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim)
        )
    
    def forward(self, content, style):
        # Concatenate content and style features
        combined = torch.cat([content, style], dim=1)
        # Generate composed features
        return self.decoder(combined)
