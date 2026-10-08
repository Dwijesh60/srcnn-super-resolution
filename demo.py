import torch
import gradio as gr
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.srcnn import SRCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_checkpoint(use_bn, use_tanh):
    """Find the best matching checkpoint for the given model configuration."""
    # Try exact match first
    exact_path = f"checkpoints/srcnn_lr0.001_wd0.0_bn{use_bn}_tanh{use_tanh}.pth"
    if os.path.exists(exact_path):
        return exact_path
    
    # Search for any checkpoint matching the bn/tanh config
    pattern = f"checkpoints/srcnn_*_bn{use_bn}_tanh{use_tanh}.pth"
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    
    # Fall back to any available checkpoint
    all_checkpoints = glob.glob("checkpoints/srcnn_*.pth")
    if all_checkpoints:
        return all_checkpoints[0]
    
    return None

def load_model(use_bn, use_tanh, model_path):
    model = SRCNN(use_bn=use_bn, use_tanh=use_tanh).to(device)
    
    if model_path and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # Checkpoint was trained with different bn/tanh setting — load compatible weights
            model.load_state_dict(state_dict, strict=False)
            print(f"Warning: Checkpoint {model_path} was trained with different settings. "
                  f"Loaded compatible weights only (some layers use random initialization).")
    else:
        print(f"Warning: No checkpoint found at '{model_path}'. Using untrained model.")
    
    model.eval()
    return model

def super_resolve(img, model_path, use_bn=False, use_tanh=False, upscale_2x=True):
    if img is None:
        return None
    
    try:
        # Ensure RGB (prevents 4-channel RGBA or 1-channel grayscale crashes)
        img = img.convert('RGB')
        
        # In SRCNN, input LR is first upscaled via Bicubic interpolation before passing through the network
        if upscale_2x:
            w, h = img.size
            img = img.resize((w * 2, h * 2), Image.BICUBIC)
        
        # Auto-resolve checkpoint if the provided path doesn't match the config
        if not model_path or not os.path.exists(model_path):
            resolved = find_checkpoint(use_bn, use_tanh)
            if resolved:
                model_path = resolved
                print(f"Auto-resolved checkpoint: {model_path}")
            else:
                raise FileNotFoundError(
                    "No trained checkpoint found. Please run train.py first "
                    "(e.g., python train.py --epochs 50)"
                )
        
        model = load_model(use_bn, use_tanh, model_path)
        
        # Process image
        transform = transforms.ToTensor()
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            output = torch.clamp(output, 0.0, 1.0)
            
        out_img = output.squeeze(0).cpu()
        out_pil = transforms.ToPILImage()(out_img)
        return out_pil
    
    except Exception as e:
        raise gr.Error(f"Super-resolution failed: {str(e)}")

def gradio_interface():
    # Determine default checkpoint path
    default_path = find_checkpoint(False, False)
    if not default_path:
        default_path = 'checkpoints/srcnn_lr0.001_wd0.0_bnFalse_tanhFalse.pth'
    
    demo = gr.Interface(
        fn=super_resolve,
        inputs=[
            gr.Image(type="pil", label="Input Low Resolution Image"),
            gr.Textbox(value=default_path, label="Model Checkpoint Path"),
            gr.Checkbox(value=False, label="Use BatchNorm"),
            gr.Checkbox(value=False, label="Use Tanh"),
            gr.Checkbox(value=True, label="2x Super-Resolution Upscaling (Bicubic Pre-upscale)")
        ],
        outputs=gr.Image(type="pil", label="Output Super Resolved Image"),
        title="SRCNN Super Resolution Demo",
        description=(
            "Upload an image to super-resolve it using the trained Residual SRCNN model. "
            "The model learns high-frequency residual details (edges, textures) on top of "
            "the bicubic baseline to produce sharper, cleaner output."
        )
    )
    demo.launch()

if __name__ == '__main__':
    gradio_interface()

