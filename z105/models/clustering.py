import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

class UnsupervisedClustering:
    """
    Unsupervised clustering for feature extraction and data preprocessing.
    """
    def __init__(self, n_clusters=10, method='kmeans'):
        self.n_clusters = n_clusters
        self.method = method
        
        if method == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
            
    def fit(self, features):
        """
        Fit the clustering model to the features.
        
        Args:
            features: Feature vectors [n_samples, feature_dim]
        """
        # Convert to numpy if it's a torch tensor
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
            
        self.model.fit(features)
        return self
        
    def predict(self, features):
        """
        Predict cluster assignments for the features.
        
        Args:
            features: Feature vectors [n_samples, feature_dim]
            
        Returns:
            Cluster assignments [n_samples]
        """
        # Convert to numpy if it's a torch tensor
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
            
        return self.model.predict(features)
        
    def get_cluster_centers(self):
        """
        Get the cluster centers.
        
        Returns:
            Cluster centers [n_clusters, feature_dim]
        """
        if self.method == 'kmeans':
            return self.model.cluster_centers_
        else:
            raise NotImplementedError(f"Cluster centers not available for {self.method}")
