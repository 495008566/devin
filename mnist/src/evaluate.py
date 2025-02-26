import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from data_loader import get_mnist_data_loaders, get_class_names
from model import MnistCNN, load_model

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        device (str): Device to evaluate on ('cpu' or 'cuda')
        
    Returns:
        tuple: (accuracy, predictions, true_labels)
    """
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Initialize variables
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # No gradient calculation for evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Calculate statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    
    return accuracy, np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(true_labels, predictions, save_path='../results/confusion_matrix.png'):
    """
    Plot the confusion matrix for the model predictions.
    
    Args:
        true_labels (numpy.ndarray): True labels
        predictions (numpy.ndarray): Model predictions
        save_path (str): Path to save the plot
    """
    # Ensure results directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get class names
    class_names = get_class_names()
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save and close
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")

def save_classification_report(true_labels, predictions, save_path='../results/classification_report.txt'):
    """
    Save the classification report to a file.
    
    Args:
        true_labels (numpy.ndarray): True labels
        predictions (numpy.ndarray): Model predictions
        save_path (str): Path to save the report
    """
    # Ensure results directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get class names
    class_names = get_class_names()
    
    # Generate classification report
    report = classification_report(true_labels, predictions, 
                                  target_names=class_names)
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"Classification report saved to {save_path}")

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
        print("Training a new model...")
        model = MnistCNN()
        print("Please run train.py first to train the model.")
        return
    
    # Evaluate model
    accuracy, predictions, true_labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device
    )
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions)
    
    # Save classification report
    save_classification_report(true_labels, predictions)
    
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
