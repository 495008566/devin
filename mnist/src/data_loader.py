import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_mnist_data_loaders(batch_size=64, val_split=0.1, data_dir='../data'):
    """
    Load the MNIST dataset and create train, validation, and test data loaders.
    
    Args:
        batch_size (int): Batch size for the data loaders
        val_split (float): Fraction of training data to use for validation
        data_dir (str): Directory to store the dataset
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load the training data
    train_dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Download and load the test data
    test_dataset = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Split training data into train and validation sets
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader

def get_class_names():
    """
    Get the class names for MNIST dataset.
    
    Returns:
        list: List of class names
    """
    return [str(i) for i in range(10)]
