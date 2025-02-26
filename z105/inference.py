import os
import torch
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.dd_gan import DDGAN
from data.data_loader import SketchDataset, ModelDataset
from utils.visualization import visualize_retrieval_results

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for cross-domain 3D model retrieval')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--sketch_data', type=str, default='data/TU_Berlin', help='Path to sketch dataset')
    parser.add_argument('--model_data', type=str, default='data/ModelNet40', help='Path to 3D model dataset')
    parser.add_argument('--query_sketch', type=str, default=None, help='Path to query sketch image')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top retrievals to show')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    
    # Load model
    model = DDGAN(device=device).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create datasets
    sketch_dataset = SketchDataset(args.sketch_data)
    model_dataset = ModelDataset(args.model_data)
    
    # Extract all model features
    model_loader = DataLoader(model_dataset, batch_size=64, shuffle=False, num_workers=4)
    model_features = []
    model_labels = []
    model_images = []
    
    with torch.no_grad():
        for models, labels in model_loader:
            models = models.to(device)
            features = model.extract_model_features(models)
            model_features.append(features.cpu().numpy())
            model_labels.append(labels.numpy())
            model_images.append(models.cpu().numpy())
    
    model_features = np.concatenate(model_features, axis=0)
    model_labels = np.concatenate(model_labels, axis=0)
    model_images = np.concatenate(model_images, axis=0)
    
    # Process query sketch
    if args.query_sketch is not None:
        # Load and preprocess single query sketch
        query_image = Image.open(args.query_sketch).convert('RGB')
        # Apply same preprocessing as in dataset
        query_tensor = sketch_dataset.transform(query_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            query_feature = model.extract_sketch_features(query_tensor).cpu().numpy()
        
        # Compute distances
        distances = np.linalg.norm(model_features - query_feature, axis=1)
        
        # Get top-k retrievals
        top_indices = np.argsort(distances)[:args.top_k]
        
        # Display results
        plt.figure(figsize=(15, 5))
        plt.subplot(1, args.top_k + 1, 1)
        plt.imshow(query_image)
        plt.title('Query Sketch')
        plt.axis('off')
        
        for i, idx in enumerate(top_indices):
            plt.subplot(1, args.top_k + 1, i + 2)
            plt.imshow(np.transpose(model_images[idx], (1, 2, 0)))
            plt.title(f'Rank {i+1}\nClass: {model_labels[idx]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'single_query_results.png'))
        plt.close()
    else:
        # Use random samples from sketch dataset as queries
        num_queries = 5
        sketch_loader = DataLoader(sketch_dataset, batch_size=num_queries, shuffle=True, num_workers=4)
        
        sketches, sketch_labels = next(iter(sketch_loader))
        sketches = sketches.to(device)
        
        with torch.no_grad():
            sketch_features = model.extract_sketch_features(sketches).cpu().numpy()
        
        # Compute distances for each query
        retrieved_images = []
        retrieved_labels = []
        
        for i in range(num_queries):
            distances = np.linalg.norm(model_features - sketch_features[i], axis=1)
            top_indices = np.argsort(distances)[:args.top_k]
            
            retrieved_images.append([model_images[idx] for idx in top_indices])
            retrieved_labels.append([model_labels[idx] for idx in top_indices])
        
        # Visualize retrieval results
        visualize_retrieval_results(
            sketches.cpu().numpy(), retrieved_images, 
            sketch_labels.numpy(), retrieved_labels,
            os.path.join(args.output_dir, 'batch_query_results.png')
        )

if __name__ == '__main__':
    main()
