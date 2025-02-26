import torch
import numpy as np
from sklearn.metrics import average_precision_score

def compute_map(query_embeddings, gallery_embeddings, query_labels, gallery_labels):
    """
    Compute mean Average Precision (mAP) for retrieval.
    
    Args:
        query_embeddings: Query embeddings [num_queries, embedding_dim]
        gallery_embeddings: Gallery embeddings [num_gallery, embedding_dim]
        query_labels: Query labels [num_queries]
        gallery_labels: Gallery labels [num_gallery]
        
    Returns:
        mAP score
    """
    # Compute pairwise distances
    distances = torch.cdist(query_embeddings, gallery_embeddings)
    
    # Convert to numpy
    distances = distances.cpu().numpy()
    query_labels = query_labels.cpu().numpy()
    gallery_labels = gallery_labels.cpu().numpy()
    
    # Compute AP for each query
    aps = []
    for i in range(len(query_labels)):
        # Get distances for this query
        query_distances = distances[i]
        
        # Create binary relevance
        relevance = (gallery_labels == query_labels[i]).astype(np.float32)
        
        # Sort by distance
        sorted_indices = np.argsort(query_distances)
        sorted_relevance = relevance[sorted_indices]
        
        # Compute AP
        ap = average_precision_score(sorted_relevance, -query_distances)
        aps.append(ap)
    
    # Compute mAP
    return np.mean(aps)

def compute_precision_at_k(query_embeddings, gallery_embeddings, query_labels, gallery_labels, k=10):
    """
    Compute Precision@K for retrieval.
    
    Args:
        query_embeddings: Query embeddings [num_queries, embedding_dim]
        gallery_embeddings: Gallery embeddings [num_gallery, embedding_dim]
        query_labels: Query labels [num_queries]
        gallery_labels: Gallery labels [num_gallery]
        k: Number of top results to consider
        
    Returns:
        Precision@K score
    """
    # Compute pairwise distances
    distances = torch.cdist(query_embeddings, gallery_embeddings)
    
    # Convert to numpy
    distances = distances.cpu().numpy()
    query_labels = query_labels.cpu().numpy()
    gallery_labels = gallery_labels.cpu().numpy()
    
    # Compute Precision@K for each query
    precisions = []
    for i in range(len(query_labels)):
        # Get distances for this query
        query_distances = distances[i]
        
        # Sort by distance
        sorted_indices = np.argsort(query_distances)[:k]
        
        # Compute precision
        precision = np.mean((gallery_labels[sorted_indices] == query_labels[i]).astype(np.float32))
        precisions.append(precision)
    
    # Compute average Precision@K
    return np.mean(precisions)
