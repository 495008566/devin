import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistCNN(nn.Module):
    """
    Convolutional Neural Network for MNIST digit recognition.
    
    Architecture:
    - 2 convolutional layers with ReLU activation and max pooling
    - 2 fully connected layers
    - Dropout for regularization
    """
    def __init__(self):
        super(MnistCNN, self).__init__()
        # First convolutional layer
        # Input: 1x28x28, Output: 32x28x28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Max pooling: 32x28x28 -> 32x14x14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        # Input: 32x14x14, Output: 64x14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling: 64x14x14 -> 64x7x7
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # First conv block
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Second conv block
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

def save_model(model, path):
    """
    Save the trained model to disk.
    
    Args:
        model: PyTorch model to save
        path (str): Path to save the model
    """
    torch.save(model.state_dict(), path)

def load_model(path):
    """
    Load a trained model from disk.
    
    Args:
        path (str): Path to the saved model
        
    Returns:
        MnistCNN: Loaded model
    """
    model = MnistCNN()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
