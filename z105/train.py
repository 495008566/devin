import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.dd_gan import DDGAN
from models.metric_learning import TripletLoss, ContrastiveLoss
from data.data_loader import SketchDataset, ModelDataset
from utils.evaluation import compute_map
from utils.visualization import plot_training_curves, visualize_feature_space, visualize_retrieval_results

def parse_args():
    parser = argparse.ArgumentParser(description='Train DD-GAN for cross-domain 3D model retrieval')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    parser.add_argument('--lambda_content', type=float, default=10.0, help='Weight for content loss')
    parser.add_argument('--lambda_style', type=float, default=1.0, help='Weight for style loss')
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Weight for cycle consistency loss')
    parser.add_argument('--lambda_triplet', type=float, default=1.0, help='Weight for triplet loss')
    parser.add_argument('--margin', type=float, default=0.3, help='Margin for triplet loss')
    parser.add_argument('--sketch_data', type=str, default='data/TU_Berlin', help='Path to sketch dataset')
    parser.add_argument('--model_data', type=str, default='data/ModelNet40', help='Path to 3D model dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--vis_dir', type=str, default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--eval_freq', type=int, default=5, help='Frequency of evaluation during training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    
    # Create datasets and dataloaders
    sketch_dataset = SketchDataset(args.sketch_data)
    model_dataset = ModelDataset(args.model_data)
    
    sketch_loader = DataLoader(sketch_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    model_loader = DataLoader(model_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Create model
    model = DDGAN(device=device).to(device)
    
    # Create loss functions
    triplet_loss = TripletLoss(margin=args.margin)
    contrastive_loss = ContrastiveLoss(margin=args.margin)
    
    # Create optimizers
    optimizer_G = optim.Adam(model.generator_parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D = optim.Adam(model.discriminator_parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint at epoch {start_epoch}")
    
    # Training loop
    train_losses = []
    eval_metrics = []
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = []
        
        # Use the shorter of the two dataloaders to determine iterations
        n_iterations = min(len(sketch_loader), len(model_loader))
        
        # Create iterators for the dataloaders
        sketch_iter = iter(sketch_loader)
        model_iter = iter(model_loader)
        
        # Progress bar
        pbar = tqdm(range(n_iterations), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for _ in pbar:
            try:
                sketch_data, sketch_labels = next(sketch_iter)
                model_data, model_labels = next(model_iter)
            except StopIteration:
                # Restart iterators if one of them is exhausted
                sketch_iter = iter(sketch_loader)
                model_iter = iter(model_loader)
                sketch_data, sketch_labels = next(sketch_iter)
                model_data, model_labels = next(model_iter)
            
            # Move data to device
            sketch_data = sketch_data.to(device)
            sketch_labels = sketch_labels.to(device)
            model_data = model_data.to(device)
            model_labels = model_labels.to(device)
            
            # Train discriminator
            optimizer_D.zero_grad()
            d_loss = model.compute_discriminator_loss(sketch_data, model_data)
            d_loss.backward()
            optimizer_D.step()
            
            # Train generator
            optimizer_G.zero_grad()
            g_loss, content_loss, style_loss, cycle_loss = model.compute_generator_loss(
                sketch_data, model_data, 
                lambda_content=args.lambda_content,
                lambda_style=args.lambda_style,
                lambda_cycle=args.lambda_cycle
            )
            
            # Compute triplet loss
            sketch_features = model.extract_sketch_features(sketch_data)
            model_features = model.extract_model_features(model_data)
            
            # Create positive and negative pairs for triplet loss
            t_loss = triplet_loss(sketch_features, model_features, sketch_labels, model_labels)
            
            # Total generator loss
            total_g_loss = g_loss + args.lambda_triplet * t_loss
            total_g_loss.backward()
            optimizer_G.step()
            
            # Update progress bar
            pbar.set_postfix({
                'D_loss': d_loss.item(),
                'G_loss': g_loss.item(),
                'Triplet_loss': t_loss.item()
            })
            
            # Record losses
            epoch_losses.append({
                'D_loss': d_loss.item(),
                'G_loss': g_loss.item(),
                'Content_loss': content_loss.item(),
                'Style_loss': style_loss.item(),
                'Cycle_loss': cycle_loss.item(),
                'Triplet_loss': t_loss.item(),
                'Total_loss': total_g_loss.item()
            })
        
        # Calculate average losses for the epoch
        avg_losses = {k: np.mean([loss[k] for loss in epoch_losses]) for k in epoch_losses[0].keys()}
        train_losses.append(avg_losses['Total_loss'])
        
        print(f"Epoch {epoch+1}/{args.epochs} - " + " ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()]))
        
        # Evaluate model
        if (epoch + 1) % args.eval_freq == 0:
            mAP = evaluate(model, sketch_dataset, model_dataset, device, args.vis_dir, epoch)
            eval_metrics.append(mAP)
            print(f"Epoch {epoch+1}/{args.epochs} - mAP: {mAP:.4f}")
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'mAP': mAP
            }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            # Plot training curves
            plot_training_curves(epoch+1, train_losses, eval_metrics, args.eval_freq, args.vis_dir)
    
    # Final evaluation
    mAP = evaluate(model, sketch_dataset, model_dataset, device, args.vis_dir, args.epochs)
    eval_metrics.append(mAP)
    print(f"Final mAP: {mAP:.4f}")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'mAP': mAP
    }, os.path.join(args.checkpoint_dir, 'final_model.pth'))
    
    # Plot final training curves
    plot_training_curves(args.epochs, train_losses, eval_metrics, args.eval_freq, args.vis_dir)

def evaluate(model, sketch_dataset, model_dataset, device, vis_dir, epoch):
    """
    Evaluate the model and compute mAP.
    """
    model.eval()
    
    # Create dataloaders for evaluation
    sketch_loader = DataLoader(sketch_dataset, batch_size=64, shuffle=False, num_workers=4)
    model_loader = DataLoader(model_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Extract features
    sketch_features = []
    sketch_labels = []
    model_features = []
    model_labels = []
    
    with torch.no_grad():
        # Extract sketch features
        for sketches, labels in sketch_loader:
            sketches = sketches.to(device)
            features = model.extract_sketch_features(sketches)
            sketch_features.append(features.cpu().numpy())
            sketch_labels.append(labels.numpy())
        
        # Extract model features
        for models, labels in model_loader:
            models = models.to(device)
            features = model.extract_model_features(models)
            model_features.append(features.cpu().numpy())
            model_labels.append(labels.numpy())
    
    # Concatenate features and labels
    sketch_features = np.concatenate(sketch_features, axis=0)
    sketch_labels = np.concatenate(sketch_labels, axis=0)
    model_features = np.concatenate(model_features, axis=0)
    model_labels = np.concatenate(model_labels, axis=0)
    
    # Compute mAP
    mAP = compute_map(sketch_features, model_features, sketch_labels, model_labels)
    
    # Visualize feature space
    if epoch % 10 == 0 or epoch == 0:  # Visualize every 10 epochs to save time
        # Subsample for visualization (t-SNE can be slow for large datasets)
        max_samples = 1000
        if len(sketch_features) > max_samples:
            idx = np.random.choice(len(sketch_features), max_samples, replace=False)
            s_features = sketch_features[idx]
            s_labels = sketch_labels[idx]
        else:
            s_features = sketch_features
            s_labels = sketch_labels
            
        if len(model_features) > max_samples:
            idx = np.random.choice(len(model_features), max_samples, replace=False)
            m_features = model_features[idx]
            m_labels = model_labels[idx]
        else:
            m_features = model_features
            m_labels = model_labels
        
        # Visualize sketch features
        visualize_feature_space(
            s_features, s_labels, 
            f'Sketch Feature Space (Epoch {epoch})',
            os.path.join(vis_dir, f'sketch_features_epoch_{epoch}.png')
        )
        
        # Visualize model features
        visualize_feature_space(
            m_features, m_labels, 
            f'Model Feature Space (Epoch {epoch})',
            os.path.join(vis_dir, f'model_features_epoch_{epoch}.png')
        )
        
        # Visualize combined features
        combined_features = np.concatenate([s_features, m_features], axis=0)
        combined_labels = np.concatenate([s_labels, m_labels], axis=0)
        domain_labels = np.concatenate([np.zeros(len(s_features)), np.ones(len(m_features))])
        
        visualize_feature_space(
            combined_features, combined_labels, 
            f'Combined Feature Space by Class (Epoch {epoch})',
            os.path.join(vis_dir, f'combined_features_by_class_epoch_{epoch}.png')
        )
        
        visualize_feature_space(
            combined_features, domain_labels, 
            f'Combined Feature Space by Domain (Epoch {epoch})',
            os.path.join(vis_dir, f'combined_features_by_domain_epoch_{epoch}.png')
        )
    
    return mAP

if __name__ == '__main__':
    main()
