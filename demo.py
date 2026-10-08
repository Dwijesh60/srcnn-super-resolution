import torch
import gradio as gr
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.srcnn import SRCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(use_bn, use_tanh, model_path):
    model = SRCNN(use_bn=use_bn, use_tanh=use_tanh).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def super_resolve(img, model_path='checkpoints/srcnn_lr0.001_wd0.0_bnFalse_tanhFalse.pth', use_bn=False, use_tanh=False, upscale_2x=True):
    if img is None:
        return None
    
    # Ensure RGB (prevents 4-channel RGBA or 1-channel grayscale crashes)
    img = img.convert('RGB')
    
    # In SRCNN, input LR is first upscaled via Bicubic interpolation before passing through the network
    if upscale_2x:
        w, h = img.size
        img = img.resize((w * 2, h * 2), Image.BICUBIC)
    
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

def gradio_interface():
    demo = gr.Interface(
        fn=super_resolve,
        inputs=[
            gr.Image(type="pil", label="Input Low Resolution Image"),
            gr.Textbox(value='checkpoints/srcnn_lr0.001_wd0.0_bnFalse_tanhFalse.pth', label="Model Checkpoint Path"),
            gr.Checkbox(value=False, label="Use BatchNorm"),
            gr.Checkbox(value=False, label="Use Tanh"),
            gr.Checkbox(value=True, label="2x Super-Resolution Upscaling (Bicubic Pre-upscale)")
        ],
        outputs=gr.Image(type="pil", label="Output Super Resolved Image"),
        title="SRCNN Super Resolution Demo",
        description="Upload an image to super-resolve it using the trained SRCNN model."
    )
    demo.launch()

if __name__ == '__main__':
    gradio_interface()
