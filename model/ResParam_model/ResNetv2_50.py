import torch
import torch.nn as nn

class ResidualBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockV2, self).__init__()
        
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)

        self.use_projection = (stride != 1 or in_channels != out_channels)
        if self.use_projection:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Pre-activation (BN + ReLU) before the first convolution
        preact = self.relu(self.bn1(x))
        
        # If shortcut is not identity, it should branch from the pre-activated signal
        if self.use_projection:
            shortcut = self.shortcut(preact)
        else:
            shortcut = x

        out = self.conv1(preact)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        # 3. Residual connection (No ReLU after addition in V2)
        return out + shortcut

class ResNetV2_50(nn.Module):
    def __init__(self, input_size, dropout=0.7):
        super(ResNetV2_50, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv3d(input_size[0], 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._makelayer(64, 64, 3, strides=1)
        self.layer2 = self._makelayer(64, 128, 4, strides=2)
        self.layer3 = self._makelayer(128, 256, 6, strides=2)
        self.layer4 = self._makelayer(256, 512, 3, strides=2)

        # V2 requires a final BN and ReLU before pooling
        self.post_bn = nn.BatchNorm3d(512)
        self.post_relu = nn.ReLU(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.fnn = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 9)
        )

    def _makelayer(self, in_channels, out_channels, blocks, strides=1):
        layers = []
        layers.append(ResidualBlockV2(in_channels, out_channels, strides))
        for _ in range(1, blocks):
            layers.append(ResidualBlockV2(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Final pre-activation step
        out = self.post_relu(self.post_bn(out))
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fnn(out)
        return out