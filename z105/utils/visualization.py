import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_training_curves(epochs, train_losses, eval_metrics, eval_freq, vis_dir):
    """
    Plot and save training curves.
    
    Args:
        epochs: Number of epochs
        train_losses: List of training losses
        eval_metrics: List of evaluation metrics (mAP)
        eval_freq: Frequency of evaluation
        vis_dir: Directory to save visualizations
    """
    # Create x-axis values
    epochs_range = list(range(1, epochs + 1))
    eval_epochs = list(range(eval_freq, epochs + 1, eval_freq))
    
    # Plot training losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Total Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot evaluation metrics
    plt.subplot(1, 2, 2)
    plt.plot(eval_epochs, eval_metrics, 'm-', label='mAP')
    plt.title('Mean Average Precision')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'training_curves.png'))
    plt.close()

def visualize_feature_space(features, labels, title, save_path):
    """
    Visualize feature space using t-SNE.
    
    Args:
        features: Feature vectors
        labels: Class labels
        title: Plot title
        save_path: Path to save visualization
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = labels == label
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=f'Class {label}', alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def visualize_retrieval_results(query_images, retrieved_images, query_labels, retrieved_labels, save_path):
    """
    Visualize retrieval results.
    
    Args:
        query_images: List of query images
        retrieved_images: List of lists of retrieved images
        query_labels: List of query labels
        retrieved_labels: List of lists of retrieved labels
        save_path: Path to save visualization
    """
    n_queries = len(query_images)
    n_results = len(retrieved_images[0])
    
    plt.figure(figsize=(15, n_queries * 3))
    
    for i in range(n_queries):
        # Plot query image
        plt.subplot(n_queries, n_results + 1, i * (n_results + 1) + 1)
        plt.imshow(np.transpose(query_images[i], (1, 2, 0)))
        plt.title(f'Query\nClass: {query_labels[i]}')
        plt.axis('off')
        
        # Plot retrieved images
        for j in range(n_results):
            plt.subplot(n_queries, n_results + 1, i * (n_results + 1) + j + 2)
            plt.imshow(np.transpose(retrieved_images[i][j], (1, 2, 0)))
            
            # Color the border based on whether the retrieval is correct
            if retrieved_labels[i][j] == query_labels[i]:
                plt.gca().spines['bottom'].set_color('green')
                plt.gca().spines['top'].set_color('green')
                plt.gca().spines['left'].set_color('green')
                plt.gca().spines['right'].set_color('green')
                plt.gca().spines['bottom'].set_linewidth(5)
                plt.gca().spines['top'].set_linewidth(5)
                plt.gca().spines['left'].set_linewidth(5)
                plt.gca().spines['right'].set_linewidth(5)
            else:
                plt.gca().spines['bottom'].set_color('red')
                plt.gca().spines['top'].set_color('red')
                plt.gca().spines['left'].set_color('red')
                plt.gca().spines['right'].set_color('red')
                plt.gca().spines['bottom'].set_linewidth(5)
                plt.gca().spines['top'].set_linewidth(5)
                plt.gca().spines['left'].set_linewidth(5)
                plt.gca().spines['right'].set_linewidth(5)
            
            plt.title(f'Rank {j+1}\nClass: {retrieved_labels[i][j]}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
