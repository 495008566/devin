import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Triplet loss for deep metric learning.
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
            
        Returns:
            Triplet loss value
        """
        # Compute distances
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Compute triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for deep metric learning.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, x1, x2, y):
        """
        Compute contrastive loss.
        
        Args:
            x1: First embeddings [batch_size, embedding_dim]
            x2: Second embeddings [batch_size, embedding_dim]
            y: Binary labels (1 for similar pairs, 0 for dissimilar pairs) [batch_size]
            
        Returns:
            Contrastive loss value
        """
        # Compute Euclidean distance
        dist = torch.sum((x1 - x2) ** 2, dim=1)
        
        # Compute contrastive loss
        loss = y * dist + (1 - y) * F.relu(self.margin - dist)
        
        return loss.mean()
