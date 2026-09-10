import torch.nn as nn

class Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual_block, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, input_size,dropout=0.7):
        super(ResNet18, self).__init__()
        assert len(input_size) == 4, "input must be in 3d with the corresponding number of channels"
        self.stemNet1 = nn.Sequential(
            nn.Conv3d(input_size[0], 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2,padding=1),
        )
        self.layer1 = self._makelayer(64, 64, 2, strides=1)
        self.layer2 = self._makelayer(64, 128, 2, strides=2)
        self.layer3 = self._makelayer(128, 256, 2, strides=2)
        self.layer4 = self._makelayer(256, 512, 2, strides=2)

        #down sample to 1x1x1, then use fnn to predict params.
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fnn = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 9)
        )

    def _makelayer(self, in_channels, out_channels, blocks, strides=1):
        layers = []
        layers.append(Residual_block(in_channels, out_channels, strides))
        for _ in range(1, blocks):
            layers.append(Residual_block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stemNet1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fnn(out)
        return out