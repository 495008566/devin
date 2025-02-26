import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision import transforms

from data_loader import get_mnist_data_loaders
from model import load_model

def visualize_predictions(model, test_loader, num_samples=10, 
                         save_path='../results/sample_predictions.png', device='cpu'):
    """
    Visualize model predictions on random samples from the test set.
    
    Args:
        model: PyTorch model to use for predictions
        test_loader: DataLoader for test data
        num_samples (int): Number of samples to visualize
        save_path (str): Path to save the visualization
        device (str): Device to run inference on ('cpu' or 'cuda')
    """
    # Ensure results directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Get random batch
    images, labels = next(iter(test_loader))
    
    # Select random samples
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    images = images[indices]
    labels = labels[indices]
    
    # Move to device
    images = images.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    # Move back to CPU for visualization
    images = images.cpu()
    predictions = predictions.cpu().numpy()
    labels = labels.numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # Plot each sample
    for i, ax in enumerate(axes):
        if i < len(images):
            # Convert image from tensor to numpy and reshape
            img = images[i].squeeze().numpy()
            
            # Display image
            ax.imshow(img, cmap='gray')
            
            # Set title with true and predicted labels
            title = f"True: {labels[i]}\nPred: {predictions[i]}"
            color = 'green' if labels[i] == predictions[i] else 'red'
            ax.set_title(title, color=color)
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Sample predictions visualization saved to {save_path}")

def visualize_feature_maps(model, test_loader, layer_name='conv1', 
                          save_path='../results/feature_maps.png', device='cpu'):
    """
    Visualize feature maps from a convolutional layer for a sample image.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        layer_name (str): Name of the convolutional layer to visualize
        save_path (str): Path to save the visualization
        device (str): Device to run inference on ('cpu' or 'cuda')
    """
    # Ensure results directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Get a sample image
    images, labels = next(iter(test_loader))
    image = images[0:1].to(device)  # Add batch dimension and move to device
    label = labels[0].item()
    
    # Create a hook to capture feature maps
    feature_maps = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output.detach().cpu())
    
    # Register the hook
    if layer_name == 'conv1':
        hook = model.conv1.register_forward_hook(hook_fn)
    elif layer_name == 'conv2':
        hook = model.conv2.register_forward_hook(hook_fn)
    else:
        raise ValueError(f"Layer {layer_name} not recognized")
    
    # Forward pass
    with torch.no_grad():
        output = model(image)
        _, prediction = torch.max(output, 1)
    
    # Remove the hook
    hook.remove()
    
    # Get the feature maps
    feature_map = feature_maps[0][0]  # First batch
    
    # Determine grid size
    num_features = feature_map.size(0)
    grid_size = int(np.ceil(np.sqrt(num_features)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    # Plot the original image
    plt.figure(figsize=(5, 5))
    plt.imshow(image[0, 0].cpu().numpy(), cmap='gray')
    plt.title(f"Original Image (Digit: {label}, Predicted: {prediction.item()})")
    plt.axis('off')
    plt.savefig(save_path.replace('.png', '_original.png'))
    plt.close()
    
    # Plot each feature map
    for i in range(grid_size * grid_size):
        ax = axes[i]
        if i < num_features:
            # Display feature map
            ax.imshow(feature_map[i].numpy(), cmap='viridis')
            ax.set_title(f"Filter {i+1}")
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Adjust layout and save
    plt.suptitle(f"Feature Maps from {layer_name} Layer", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Feature maps visualization saved to {save_path}")

def visualize_misclassified(model, test_loader, num_samples=10, 
                           save_path='../results/misclassified.png', device='cpu'):
    """
    Visualize misclassified samples from the test set.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        num_samples (int): Number of misclassified samples to visualize
        save_path (str): Path to save the visualization
        device (str): Device to run inference on ('cpu' or 'cuda')
    """
    # Ensure results directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Collect misclassified samples
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            # Move to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            # Find misclassified samples
            mask = predictions != labels
            if torch.any(mask):
                misclassified_idx = torch.where(mask)[0]
                misclassified_images.extend(images[misclassified_idx].cpu())
                misclassified_labels.extend(labels[misclassified_idx].cpu().numpy())
                misclassified_preds.extend(predictions[misclassified_idx].cpu().numpy())
            
            # Break if we have enough samples
            if len(misclassified_images) >= num_samples:
                break
    
    # Limit to requested number of samples
    misclassified_images = misclassified_images[:num_samples]
    misclassified_labels = misclassified_labels[:num_samples]
    misclassified_preds = misclassified_preds[:num_samples]
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # Plot each misclassified sample
    for i, ax in enumerate(axes):
        if i < len(misclassified_images):
            # Convert image from tensor to numpy and reshape
            img = misclassified_images[i].squeeze().numpy()
            
            # Display image
            ax.imshow(img, cmap='gray')
            
            # Set title with true and predicted labels
            title = f"True: {misclassified_labels[i]}\nPred: {misclassified_preds[i]}"
            ax.set_title(title, color='red')
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Adjust layout and save
    plt.suptitle("Misclassified Samples", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Misclassified samples visualization saved to {save_path}")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Hyperparameters
    batch_size = 64
    
    # Get data loaders (only need test loader)
    _, _, test_loader = get_mnist_data_loaders(batch_size=batch_size)
    print(f"Test set size: {len(test_loader.dataset)}")
    
    # Load trained model
    model_path = '../models/mnist_cnn.pth'
    try:
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        print("Please run train.py first to train the model.")
        return
    
    # Visualize predictions
    visualize_predictions(model, test_loader, device=device)
    
    # Visualize feature maps
    visualize_feature_maps(model, test_loader, layer_name='conv1', device=device)
    visualize_feature_maps(model, test_loader, layer_name='conv2', 
                          save_path='../results/feature_maps_conv2.png', device=device)
    
    # Visualize misclassified samples
    visualize_misclassified(model, test_loader, device=device)
    
    print("Visualization completed successfully!")

if __name__ == "__main__":
    main()
