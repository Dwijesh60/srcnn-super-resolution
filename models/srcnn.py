import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, use_bn=False, use_tanh=False):
        super(SRCNN, self).__init__()
        self.use_bn = use_bn
        self.use_tanh = use_tanh
        
        # Layer 1 (Patch Extraction & Representation)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        self.relu1 = nn.ReLU()
        
        # Layer 2 (Non-linear Mapping)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.relu2 = nn.ReLU()
        
        # Layer 3 (Reconstruction)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)
        self.tanh = nn.Tanh() if use_tanh else nn.Identity()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.tanh(x)
        
        return x
