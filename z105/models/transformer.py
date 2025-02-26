import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Self-attention mechanism for the transformer module.
    """
    def __init__(self, dim, heads=8):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear projections
        queries = self.query(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        keys = self.key(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        values = self.value(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        
        # Attention
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(energy, dim=-1)
        out = torch.matmul(attention, values)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        return self.out(out)

class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward network.
    """
    def __init__(self, dim, heads=8, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = SelfAttention(dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.dropout(x)
        
        # Feed-forward with residual connection
        fed_forward = self.feed_forward(x)
        x = self.norm2(fed_forward + x)
        x = self.dropout(x)
        
        return x

class TransformerModule(nn.Module):
    """
    Transformer module for enhancing feature representation.
    """
    def __init__(self, dim=256, depth=2, heads=8, dropout=0.1):
        super(TransformerModule, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dropout) for _ in range(depth)
        ])
        
    def forward(self, x):
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x)
            
        # Remove sequence dimension if it was added
        if x.size(1) == 1:
            x = x.squeeze(1)
            
        return x
