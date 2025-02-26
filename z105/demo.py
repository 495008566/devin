import os
import numpy as np
import matplotlib.pyplot as plt
import random

def create_mock_data():
    """Generate mock data and visualizations for demonstration"""
    # Create directories
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Generate mock training curves
    epochs = 100
    train_losses = [random.uniform(0.5, 5.0) * np.exp(-0.03 * i) + random.uniform(0.1, 0.3) for i in range(epochs)]
    eval_freq = 5
    eval_epochs = list(range(eval_freq, epochs + 1, eval_freq))
    eval_metrics = [0.2 + 0.6 * (1 - np.exp(-0.1 * i)) + random.uniform(-0.05, 0.05) for i in eval_epochs]
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(1, epochs + 1)), train_losses, 'b-', label='Total Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(eval_epochs, eval_metrics, 'm-', label='mAP')
    plt.title('Mean Average Precision')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/training_curves.png')
    plt.close()
    
    # Generate mock feature space visualization
    n_samples = 500
    n_classes = 10
    
    # Generate random features
    features = np.random.randn(n_samples, 2)
    labels = np.random.randint(0, n_classes, n_samples)
    
    # Add some class separation
    for i in range(n_classes):
        class_center = np.random.randn(2) * 5
        idx = labels == i
        features[idx] = features[idx] + class_center
    
    # Plot feature space
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        idx = labels == i
        plt.scatter(features[idx, 0], features[idx, 1], label=f'Class {i}', alpha=0.7)
    plt.title('Feature Space Visualization')
    plt.legend()
    plt.savefig('visualizations/feature_space.png')
    plt.close()
    
    # Generate mock retrieval results
    create_retrieval_visualization()
    
    print("Mock data and visualizations created successfully!")
    return {
        'training_curves': 'visualizations/training_curves.png',
        'feature_space': 'visualizations/feature_space.png',
        'retrieval_results': 'results/retrieval_results.png'
    }

def create_retrieval_visualization():
    """Create visualization of retrieval results"""
    n_queries = 3
    n_results = 5
    n_classes = 10
    
    # Create mock images (colored squares with random patterns)
    def create_mock_image(is_sketch=False, size=64):
        if is_sketch:
            # Create a sketch-like image (black and white)
            img = np.ones((size, size, 3)) * 255
            # Add random lines
            for _ in range(20):
                x1, y1 = np.random.randint(0, size, 2)
                x2, y2 = np.random.randint(0, size, 2)
                rr, cc = np.array([np.linspace(x1, x2, 100).astype(int), 
                                  np.linspace(y1, y2, 100).astype(int)])
                # Keep points within bounds
                valid_idx = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
                rr, cc = rr[valid_idx], cc[valid_idx]
                img[rr, cc] = 0
            return img
        else:
            # Create a colored 3D model-like image
            base_color = np.random.rand(3)
            img = np.ones((size, size, 3)) * 0.8
            # Add a colored shape
            center_x, center_y = np.random.randint(20, size-20, 2)
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if dist < size/4:
                        img[i, j] = base_color
                    elif dist < size/3:
                        # Add shading
                        t = (dist - size/4) / (size/12)
                        img[i, j] = base_color * (1 - t) + np.ones(3) * 0.8 * t
            return img
    
    # Create query and retrieved images
    query_images = [create_mock_image(is_sketch=True) for _ in range(n_queries)]
    query_labels = np.random.randint(0, n_classes, n_queries)
    
    retrieved_images = []
    retrieved_labels = []
    
    for i in range(n_queries):
        # Make some retrievals correct and some incorrect
        q_label = query_labels[i]
        r_images = []
        r_labels = []
        
        # First result is always correct
        r_images.append(create_mock_image(is_sketch=False))
        r_labels.append(q_label)
        
        # Rest are random with 50% chance of being correct
        for _ in range(n_results - 1):
            r_images.append(create_mock_image(is_sketch=False))
            if random.random() < 0.5:
                r_labels.append(q_label)  # Correct match
            else:
                # Incorrect match
                wrong_label = q_label
                while wrong_label == q_label:
                    wrong_label = np.random.randint(0, n_classes)
                r_labels.append(wrong_label)
        
        retrieved_images.append(r_images)
        retrieved_labels.append(r_labels)
    
    # Visualize retrieval results
    plt.figure(figsize=(15, n_queries * 3))
    
    for i in range(n_queries):
        # Plot query image
        plt.subplot(n_queries, n_results + 1, i * (n_results + 1) + 1)
        plt.imshow(query_images[i])
        plt.title(f'Query\nClass: {query_labels[i]}')
        plt.axis('off')
        
        # Plot retrieved images
        for j in range(n_results):
            plt.subplot(n_queries, n_results + 1, i * (n_results + 1) + j + 2)
            plt.imshow(retrieved_images[i][j])
            
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
    plt.savefig('results/retrieval_results.png')
    plt.close()

if __name__ == '__main__':
    results = create_mock_data()
    
    # Display paths to generated visualizations
    print("\nGenerated Visualizations:")
    for name, path in results.items():
        print(f"- {name}: {path}")
