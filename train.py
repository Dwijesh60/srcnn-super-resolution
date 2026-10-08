import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import SRCNNDataset
from models.srcnn import SRCNN
from baselines.interpolation import calculate_psnr

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    image_dir = os.path.join('data', 'DIV2K', 'DIV2K_valid_HR')
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Dataset not found at {image_dir}. Run data/download_div2k.py first.")
        
    train_dataset = SRCNNDataset(image_dir, split='train')
    val_dataset = SRCNNDataset(image_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = SRCNN(use_bn=args.use_bn, use_tanh=args.use_tanh).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_psnr = 0.0
    
    os.makedirs('checkpoints', exist_ok=True)
    model_path = f"checkpoints/srcnn_lr{args.lr}_wd{args.weight_decay}_bn{args.use_bn}_tanh{args.use_tanh}.pth"
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for lr_up, hr, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            lr_up = lr_up.to(device)
            hr = hr.to(device)
            
            optimizer.zero_grad()
            output = model(lr_up)
            
            # If not using tanh, we might want to clamp the output to [0, 1] during eval, 
            # but for training we just compute MSE
            loss = criterion(output, hr)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * lr_up.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for lr_up, hr, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                lr_up = lr_up.to(device)
                hr = hr.to(device)
                
                output = model(lr_up)
                output = torch.clamp(output, 0.0, 1.0)
                
                val_psnr += calculate_psnr(output, hr) * lr_up.size(0)
                
        val_psnr /= len(val_loader.dataset)
        
        print(f"Epoch {epoch} | Train Loss (MSE): {train_loss:.6f} | Val PSNR: {val_psnr:.2f} dB")
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), model_path)
            print(f"-> Saved new best model to {model_path} with PSNR {best_psnr:.2f} dB")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use_bn', action='store_true', help='Use BatchNorm')
    parser.add_argument('--use_tanh', action='store_true', help='Use Tanh')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 penalty)')
    
    args = parser.parse_args()
    train(args)
