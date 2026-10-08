import os
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import SRCNNDataset
from models.srcnn import SRCNN
from baselines.interpolation import calculate_psnr, nearest_neighbor, bilinear, bicubic

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    image_dir = os.path.join('data', 'DIV2K', 'DIV2K_valid_HR')
    test_dataset = SRCNNDataset(image_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = SRCNN(use_bn=args.use_bn, use_tanh=args.use_tanh).to(device)
    if args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print("Warning: No model checkpoint provided or found. Evaluating untrained model.")
        
    model.eval()
    
    psnr_srcnn = 0.0
    psnr_nn = 0.0
    psnr_bilinear = 0.0
    psnr_bicubic = 0.0
    
    os.makedirs('results', exist_ok=True)
    
    with torch.no_grad():
        for i, (lr_up, hr, lr_112) in enumerate(test_loader):
            lr_up = lr_up.to(device)
            hr = hr.to(device)
            
            # SRCNN Prediction
            output = model(lr_up)
            output = torch.clamp(output, 0.0, 1.0)
            
            # Baselines
            nn_out = nearest_neighbor(lr_112).to(device)
            bilinear_out = bilinear(lr_112).to(device)
            bicubic_out = bicubic(lr_112).to(device)
            
            psnr_srcnn += calculate_psnr(output, hr)
            psnr_nn += calculate_psnr(nn_out, hr)
            psnr_bilinear += calculate_psnr(bilinear_out, hr)
            psnr_bicubic += calculate_psnr(bicubic_out, hr)
            
            if i < args.num_visualize:
                # To make the visual difference obvious, we extract a zoomed-in 64x64 patch
                # from the center of the image
                start_x, start_y = 80, 80
                patch_size = 64
                
                def crop_patch(tensor):
                    return tensor[0, :, start_y:start_y+patch_size, start_x:start_x+patch_size].cpu().permute(1, 2, 0)
                
                fig, axes = plt.subplots(1, 5, figsize=(20, 5))
                axes[0].imshow(crop_patch(lr_up))
                axes[0].set_title('Bicubic (Blurry)')
                axes[1].imshow(crop_patch(nn_out))
                axes[1].set_title('Nearest Neighbor')
                axes[2].imshow(crop_patch(bilinear_out))
                axes[2].set_title('Bilinear')
                axes[3].imshow(crop_patch(output))
                axes[3].set_title('SRCNN (Ours)')
                axes[4].imshow(crop_patch(hr))
                axes[4].set_title('Ground Truth (HR)')
                
                for ax in axes:
                    ax.axis('off')
                    
                plt.suptitle(f"Zoomed-in Patch Comparison (Image {i+1})", fontsize=16)
                plt.tight_layout()
                plt.savefig(f'results/test_sample_{i+1}_zoomed.png')
                plt.close()
                
    n = len(test_loader.dataset)
    print("\n--- Evaluation Results (Test Set) ---")
    print(f"Nearest Neighbor PSNR : {psnr_nn/n:.2f} dB")
    print(f"Bilinear PSNR         : {psnr_bilinear/n:.2f} dB")
    print(f"Bicubic PSNR          : {psnr_bicubic/n:.2f} dB")
    print(f"SRCNN PSNR            : {psnr_srcnn/n:.2f} dB")
    print("-------------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/srcnn_lr0.001_wd0.0_bnFalse_tanhFalse.pth')
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--use_tanh', action='store_true')
    parser.add_argument('--num_visualize', type=int, default=3, help='Number of test images to save visualizations for')
    
    args = parser.parse_args()
    evaluate(args)
