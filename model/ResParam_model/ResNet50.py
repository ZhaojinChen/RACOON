import torch
import torch.nn as nn

class Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual_block, self).__init__()
        # 1x1x1 bottleneck convolution (projection)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        # 3x3x3 spatial convolution (responsible for downsampling via stride)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 1x1x1 bottleneck convolution (expansion/identity mapping)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = self.relu(out)
        
        return out

class ResNet50(nn.Module):
    def __init__(self, input_size, dropout=0.7):
        super(ResNet50, self).__init__()
        
        # Initial Stem: 7x7x7 conv and maxpooling for aggressive downsampling
        self.stemNet1 = nn.Sequential(
            nn.Conv3d(input_size[0], 64, kernel_size=7, stride=2, padding=3, bias=False), 
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1), 
        )

        # ResNet Layers: (In, Out, Blocks, Stride)
        self.layer1 = self._makelayer(64, 64, 3, strides=1)
        self.layer2 = self._makelayer(64, 128, 4, strides=2)   # Downsample 1/8
        self.layer3 = self._makelayer(128, 256, 6, strides=2)  # Downsample 1/16
        self.layer4 = self._makelayer(256, 512, 3, strides=2)  # Downsample 1/32

        # Adaptive pool handles any input spatial size and outputs 1x1x1
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fully Connected Network
        self.fnn = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 9)
        )

    def _makelayer(self, in_channels, out_channels, blocks, strides=1):
        layers = []
        # First block of the layer handles the stride and channel expansion
        layers.append(Residual_block(in_channels, out_channels, strides))
        
        # Subsequent blocks maintain the same dimensions
        for _ in range(1, blocks):
            layers.append(Residual_block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (Batch, Channels, Depth, Height, Width)
        out = self.stemNet1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1) # Flatten for Linear layers
        out = self.fnn(out)
        return out