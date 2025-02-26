import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.cm as cm

def create_directories():
    """Create necessary directories for visualizations"""
    os.makedirs('visualizations/detailed', exist_ok=True)
    os.makedirs('visualizations/animation', exist_ok=True)
    os.makedirs('results/detailed', exist_ok=True)

def generate_training_progress():
    """Generate more detailed training progress visualizations"""
    # Create mock training data
    epochs = 100
    batch_size = 32
    iterations_per_epoch = 50
    
    # Generate loss data with more detail
    g_losses = []
    d_losses = []
    triplet_losses = []
    content_losses = []
    style_losses = []
    cycle_losses = []
    
    # Add some randomness but with a clear downward trend
    for i in range(epochs):
        epoch_g_losses = [2.5 * np.exp(-0.03 * i) + random.uniform(0.1, 0.5) for _ in range(iterations_per_epoch)]
        epoch_d_losses = [1.5 * np.exp(-0.02 * i) + random.uniform(0.1, 0.3) for _ in range(iterations_per_epoch)]
        epoch_triplet_losses = [1.0 * np.exp(-0.04 * i) + random.uniform(0.05, 0.2) for _ in range(iterations_per_epoch)]
        epoch_content_losses = [3.0 * np.exp(-0.03 * i) + random.uniform(0.2, 0.6) for _ in range(iterations_per_epoch)]
        epoch_style_losses = [0.8 * np.exp(-0.02 * i) + random.uniform(0.05, 0.15) for _ in range(iterations_per_epoch)]
        epoch_cycle_losses = [2.0 * np.exp(-0.025 * i) + random.uniform(0.1, 0.4) for _ in range(iterations_per_epoch)]
        
        g_losses.append(np.mean(epoch_g_losses))
        d_losses.append(np.mean(epoch_d_losses))
        triplet_losses.append(np.mean(epoch_triplet_losses))
        content_losses.append(np.mean(epoch_content_losses))
        style_losses.append(np.mean(epoch_style_losses))
        cycle_losses.append(np.mean(epoch_cycle_losses))
    
    # Generate evaluation metrics
    eval_freq = 5
    eval_epochs = list(range(eval_freq, epochs + 1, eval_freq))
    map_values = [0.2 + 0.6 * (1 - np.exp(-0.1 * i)) + random.uniform(-0.05, 0.05) for i in eval_epochs]
    precision_values = [0.3 + 0.5 * (1 - np.exp(-0.08 * i)) + random.uniform(-0.03, 0.03) for i in eval_epochs]
    recall_values = [0.25 + 0.55 * (1 - np.exp(-0.09 * i)) + random.uniform(-0.04, 0.04) for i in eval_epochs]
    
    # Plot detailed training curves
    plt.figure(figsize=(15, 10))
    
    # Plot generator and discriminator losses
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs + 1), g_losses, 'b-', label='Generator Loss')
    plt.plot(range(1, epochs + 1), d_losses, 'r-', label='Discriminator Loss')
    plt.title('Generator and Discriminator Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot component losses
    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs + 1), triplet_losses, 'm-', label='Triplet Loss')
    plt.plot(range(1, epochs + 1), content_losses, 'g-', label='Content Loss')
    plt.plot(range(1, epochs + 1), style_losses, 'c-', label='Style Loss')
    plt.plot(range(1, epochs + 1), cycle_losses, 'y-', label='Cycle Loss')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot evaluation metrics
    plt.subplot(2, 2, 3)
    plt.plot(eval_epochs, map_values, 'm-o', label='mAP')
    plt.title('Mean Average Precision')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)
    
    # Plot precision and recall
    plt.subplot(2, 2, 4)
    plt.plot(eval_epochs, precision_values, 'g-o', label='Precision')
    plt.plot(eval_epochs, recall_values, 'b-o', label='Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/detailed/training_progress.png')
    plt.close()
    
    # Create animation frames showing training progress
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame):
        ax.clear()
        ax.plot(range(1, frame + 2), g_losses[:frame + 1], 'b-', label='Generator Loss')
        ax.plot(range(1, frame + 2), d_losses[:frame + 1], 'r-', label='Discriminator Loss')
        ax.set_xlim(1, epochs)
        ax.set_ylim(0, max(max(g_losses), max(d_losses)) * 1.1)
        ax.set_title(f'Training Progress (Epoch {frame + 1}/{epochs})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        return ax,
    
    # Create animation with 20 frames (showing progress at different points)
    frames = [i for i in range(0, epochs, 5)]
    ani = FuncAnimation(fig, update, frames=frames, blit=True)
    ani.save('visualizations/animation/training_progress.gif', writer='pillow', fps=2)
    plt.close()
    
    return {
        'training_progress': 'visualizations/detailed/training_progress.png',
        'training_animation': 'visualizations/animation/training_progress.gif'
    }

def generate_feature_space_visualizations():
    """Generate more detailed feature space visualizations"""
    # Create mock feature data
    n_samples = 1000
    n_classes = 10
    n_features = 128
    
    # Generate random features for sketches and models
    sketch_features = np.random.randn(n_samples, n_features)
    model_features = np.random.randn(n_samples, n_features)
    
    # Add some class separation
    sketch_labels = np.random.randint(0, n_classes, n_samples)
    model_labels = np.random.randint(0, n_classes, n_samples)
    
    for i in range(n_classes):
        class_center_sketch = np.random.randn(n_features) * 5
        class_center_model = class_center_sketch + np.random.randn(n_features) * 2  # Similar but not identical
        
        idx_sketch = sketch_labels == i
        idx_model = model_labels == i
        
        sketch_features[idx_sketch] = sketch_features[idx_sketch] + class_center_sketch
        model_features[idx_model] = model_features[idx_model] + class_center_model
    
    # Apply t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42)
    sketch_features_2d = tsne.fit_transform(sketch_features)
    
    tsne = TSNE(n_components=2, random_state=42)
    model_features_2d = tsne.fit_transform(model_features)
    
    # Apply PCA for 3D visualization
    pca = PCA(n_components=3)
    sketch_features_3d = pca.fit_transform(sketch_features)
    
    pca = PCA(n_components=3)
    model_features_3d = pca.fit_transform(model_features)
    
    # Combine features for domain adaptation visualization
    combined_features = np.vstack([sketch_features, model_features])
    combined_labels = np.concatenate([sketch_labels, model_labels])
    domain_labels = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    
    tsne = TSNE(n_components=2, random_state=42)
    combined_features_2d = tsne.fit_transform(combined_features)
    
    # Plot sketch feature space
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        idx = sketch_labels == i
        plt.scatter(sketch_features_2d[idx, 0], sketch_features_2d[idx, 1], label=f'Class {i}', alpha=0.7)
    plt.title('Sketch Feature Space (t-SNE)')
    plt.legend()
    plt.savefig('visualizations/detailed/sketch_feature_space.png')
    plt.close()
    
    # Plot model feature space
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        idx = model_labels == i
        plt.scatter(model_features_2d[idx, 0], model_features_2d[idx, 1], label=f'Class {i}', alpha=0.7)
    plt.title('3D Model Feature Space (t-SNE)')
    plt.legend()
    plt.savefig('visualizations/detailed/model_feature_space.png')
    plt.close()
    
    # Plot combined feature space by class
    plt.figure(figsize=(12, 10))
    for i in range(n_classes):
        idx = combined_labels == i
        plt.scatter(combined_features_2d[idx, 0], combined_features_2d[idx, 1], label=f'Class {i}', alpha=0.7)
    plt.title('Combined Feature Space by Class (t-SNE)')
    plt.legend()
    plt.savefig('visualizations/detailed/combined_feature_space_by_class.png')
    plt.close()
    
    # Plot combined feature space by domain
    plt.figure(figsize=(12, 10))
    colors = ['blue', 'red']
    domains = ['Sketch', '3D Model']
    for i in range(2):
        idx = domain_labels == i
        plt.scatter(combined_features_2d[idx, 0], combined_features_2d[idx, 1], c=colors[i], label=domains[i], alpha=0.7)
    plt.title('Combined Feature Space by Domain (t-SNE)')
    plt.legend()
    plt.savefig('visualizations/detailed/combined_feature_space_by_domain.png')
    plt.close()
    
    # Create 3D visualization of feature space
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(n_classes):
        idx = sketch_labels == i
        ax.scatter(sketch_features_3d[idx, 0], sketch_features_3d[idx, 1], sketch_features_3d[idx, 2], 
                  label=f'Class {i}', alpha=0.7)
    
    ax.set_title('3D Visualization of Sketch Feature Space (PCA)')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.legend()
    plt.savefig('visualizations/detailed/sketch_feature_space_3d.png')
    plt.close()
    
    # Create feature space evolution visualization (before and after training)
    # Simulate feature space before training (more scattered)
    before_features = combined_features + np.random.randn(*combined_features.shape) * 5
    tsne = TSNE(n_components=2, random_state=42)
    before_features_2d = tsne.fit_transform(before_features)
    
    # Create a side-by-side comparison
    plt.figure(figsize=(20, 8))
    
    plt.subplot(1, 2, 1)
    for i in range(2):
        idx = domain_labels == i
        plt.scatter(before_features_2d[idx, 0], before_features_2d[idx, 1], c=colors[i], label=domains[i], alpha=0.7)
    plt.title('Feature Space Before Training')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for i in range(2):
        idx = domain_labels == i
        plt.scatter(combined_features_2d[idx, 0], combined_features_2d[idx, 1], c=colors[i], label=domains[i], alpha=0.7)
    plt.title('Feature Space After Training')
    plt.legend()
    
    plt.savefig('visualizations/detailed/feature_space_evolution.png')
    plt.close()
    
    return {
        'sketch_feature_space': 'visualizations/detailed/sketch_feature_space.png',
        'model_feature_space': 'visualizations/detailed/model_feature_space.png',
        'combined_feature_space_by_class': 'visualizations/detailed/combined_feature_space_by_class.png',
        'combined_feature_space_by_domain': 'visualizations/detailed/combined_feature_space_by_domain.png',
        'sketch_feature_space_3d': 'visualizations/detailed/sketch_feature_space_3d.png',
        'feature_space_evolution': 'visualizations/detailed/feature_space_evolution.png'
    }

def generate_retrieval_visualizations():
    """Generate more detailed retrieval visualizations"""
    # Create mock retrieval results
    n_queries = 5
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
    retrieved_scores = []
    
    for i in range(n_queries):
        # Make some retrievals correct and some incorrect
        q_label = query_labels[i]
        r_images = []
        r_labels = []
        r_scores = []
        
        # First result is always correct with high score
        r_images.append(create_mock_image(is_sketch=False))
        r_labels.append(q_label)
        r_scores.append(random.uniform(0.8, 0.95))
        
        # Rest are random with 50% chance of being correct
        for j in range(n_results - 1):
            r_images.append(create_mock_image(is_sketch=False))
            if random.random() < 0.5:
                r_labels.append(q_label)  # Correct match
                r_scores.append(random.uniform(0.6, 0.85))
            else:
                # Incorrect match
                wrong_label = q_label
                while wrong_label == q_label:
                    wrong_label = np.random.randint(0, n_classes)
                r_labels.append(wrong_label)
                r_scores.append(random.uniform(0.3, 0.6))
        
        retrieved_images.append(r_images)
        retrieved_labels.append(r_labels)
        retrieved_scores.append(r_scores)
    
    # Visualize retrieval results with scores
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
            
            plt.title(f'Rank {j+1}\nClass: {retrieved_labels[i][j]}\nScore: {retrieved_scores[i][j]:.2f}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/detailed/retrieval_results_with_scores.png')
    plt.close()
    
    # Create a heatmap visualization of similarity scores
    plt.figure(figsize=(12, 8))
    
    # Generate a similarity matrix
    n_samples = 20
    similarity_matrix = np.zeros((n_samples, n_samples))
    
    # Generate random class labels
    sample_labels = np.random.randint(0, n_classes, n_samples)
    
    # Fill similarity matrix based on class labels
    for i in range(n_samples):
        for j in range(n_samples):
            if sample_labels[i] == sample_labels[j]:
                # Same class - higher similarity
                similarity_matrix[i, j] = random.uniform(0.7, 1.0)
            else:
                # Different class - lower similarity
                similarity_matrix[i, j] = random.uniform(0.1, 0.5)
    
    # Plot heatmap
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar(label='Similarity Score')
    plt.title('Cross-Domain Similarity Matrix')
    plt.xlabel('3D Model Samples')
    plt.ylabel('Sketch Samples')
    plt.savefig('results/detailed/similarity_heatmap.png')
    plt.close()
    
    # Create precision-recall curve
    recall_points = np.linspace(0, 1, 100)
    precision_points = []
    
    # Generate precision values with a typical PR curve shape
    for r in recall_points:
        if r < 0.6:
            p = 1.0 - 0.1 * r + random.uniform(-0.05, 0.05)
        else:
            p = 1.2 - 0.8 * r + random.uniform(-0.05, 0.05)
        # Ensure precision is between 0 and 1
        p = max(0, min(1, p))
        precision_points.append(p)
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall_points, precision_points, 'b-', linewidth=2)
    plt.fill_between(recall_points, precision_points, alpha=0.2)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig('results/detailed/precision_recall_curve.png')
    plt.close()
    
    return {
        'retrieval_results_with_scores': 'results/detailed/retrieval_results_with_scores.png',
        'similarity_heatmap': 'results/detailed/similarity_heatmap.png',
        'precision_recall_curve': 'results/detailed/precision_recall_curve.png'
    }

def generate_model_architecture_visualization():
    """Generate visualization of the model architecture"""
    plt.figure(figsize=(15, 10))
    
    # Define the components
    components = [
        {'name': 'Sketch Input', 'x': 0.1, 'y': 0.8, 'width': 0.15, 'height': 0.1, 'color': 'lightblue'},
        {'name': '3D Model Input', 'x': 0.1, 'y': 0.2, 'width': 0.15, 'height': 0.1, 'color': 'lightgreen'},
        
        {'name': 'SketchCNN\n(ResNet-50)', 'x': 0.3, 'y': 0.8, 'width': 0.15, 'height': 0.1, 'color': 'skyblue'},
        {'name': 'ShapeCNN\n(ResNet-50)', 'x': 0.3, 'y': 0.2, 'width': 0.15, 'height': 0.1, 'color': 'lightgreen'},
        
        {'name': 'Content\nEncoder', 'x': 0.5, 'y': 0.65, 'width': 0.1, 'height': 0.1, 'color': 'gold'},
        {'name': 'Style\nEncoder', 'x': 0.5, 'y': 0.5, 'width': 0.1, 'height': 0.1, 'color': 'gold'},
        {'name': 'Content\nEncoder', 'x': 0.5, 'y': 0.35, 'width': 0.1, 'height': 0.1, 'color': 'gold'},
        {'name': 'Style\nEncoder', 'x': 0.5, 'y': 0.2, 'width': 0.1, 'height': 0.1, 'color': 'gold'},
        
        {'name': 'Transformer\nModule', 'x': 0.65, 'y': 0.5, 'width': 0.15, 'height': 0.15, 'color': 'lightcoral'},
        
        {'name': 'Domain\nComposition', 'x': 0.85, 'y': 0.5, 'width': 0.1, 'height': 0.15, 'color': 'plum'},
        
        {'name': 'Discriminator', 'x': 0.7, 'y': 0.8, 'width': 0.1, 'height': 0.1, 'color': 'salmon'},
        {'name': 'Discriminator', 'x': 0.7, 'y': 0.2, 'width': 0.1, 'height': 0.1, 'color': 'salmon'},
        
        {'name': 'Triplet Loss', 'x': 0.85, 'y': 0.8, 'width': 0.1, 'height': 0.1, 'color': 'orchid'},
    ]
    
    # Draw components
    ax = plt.gca()
    for comp in components:
        rect = plt.Rectangle((comp['x'], comp['y']), comp['width'], comp['height'], 
                            facecolor=comp['color'], alpha=0.8, edgecolor='black')
        ax.add_patch(rect)
        plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, comp['name'],
                ha='center', va='center', fontsize=10)
    
    # Draw arrows
    arrows = [
        # Sketch flow
        {'start': (0.25, 0.85), 'end': (0.3, 0.85)},
        {'start': (0.45, 0.85), 'end': (0.5, 0.7)},
        {'start': (0.45, 0.85), 'end': (0.5, 0.55)},
        {'start': (0.6, 0.7), 'end': (0.65, 0.575)},
        {'start': (0.6, 0.55), 'end': (0.65, 0.525)},
        
        # 3D Model flow
        {'start': (0.25, 0.25), 'end': (0.3, 0.25)},
        {'start': (0.45, 0.25), 'end': (0.5, 0.4)},
        {'start': (0.45, 0.25), 'end': (0.5, 0.25)},
        {'start': (0.6, 0.4), 'end': (0.65, 0.475)},
        {'start': (0.6, 0.25), 'end': (0.65, 0.425)},
        
        # Transformer to composition
        {'start': (0.8, 0.575), 'end': (0.85, 0.575)},
        
        # To discriminators
        {'start': (0.8, 0.575), 'end': (0.7, 0.85)},
        {'start': (0.8, 0.425), 'end': (0.7, 0.25)},
        
        # To triplet loss
        {'start': (0.8, 0.575), 'end': (0.85, 0.85)},
    ]
    
    for arrow in arrows:
        plt.annotate('', xy=arrow['end'], xytext=arrow['start'],
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    # Add title and labels
    plt.title('DD-GAN Architecture for Cross-Domain 3D Model Retrieval', fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.savefig('visualizations/detailed/model_architecture.png')
    plt.close()
    
    return {
        'model_architecture': 'visualizations/detailed/model_architecture.png'
    }

def main():
    """Generate all visualizations"""
    create_directories()
    
    # Generate all visualizations
    training_vis = generate_training_progress()
    feature_vis = generate_feature_space_visualizations()
    retrieval_vis = generate_retrieval_visualizations()
    architecture_vis = generate_model_architecture_visualization()
    
    # Combine all results
    all_visualizations = {**training_vis, **feature_vis, **retrieval_vis, **architecture_vis}
    
    print("All visualizations generated successfully!")
    
    # Display paths to generated visualizations
    print("\nGenerated Visualizations:")
    for name, path in all_visualizations.items():
        print(f"- {name}: {path}")
    
    return all_visualizations

if __name__ == '__main__':
    main()
