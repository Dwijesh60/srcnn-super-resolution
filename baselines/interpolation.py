import torch
import torch.nn.functional as F
import math

def calculate_psnr(img1, img2):
    """
    Computes PSNR between two images.
    Images should be PyTorch tensors with values in [0.0, 1.0].
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * math.log10(1.0 / mse.item())
    return psnr

def nearest_neighbor(lr_tensor):
    """
    Upsamples the 112x112 tensor to 224x224 using nearest neighbor interpolation.
    """
    return F.interpolate(lr_tensor, size=(224, 224), mode='nearest')

def bilinear(lr_tensor):
    """
    Upsamples the 112x112 tensor to 224x224 using bilinear interpolation.
    """
    return F.interpolate(lr_tensor, size=(224, 224), mode='bilinear', align_corners=False)

def bicubic(lr_tensor):
    """
    Upsamples the 112x112 tensor to 224x224 using bicubic interpolation.
    """
    return F.interpolate(lr_tensor, size=(224, 224), mode='bicubic', align_corners=False)
