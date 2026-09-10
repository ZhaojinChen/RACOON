import torch.nn as nn
import torch

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
    def __init__(self, input_size, dropout=0.3, spatial_dropout=0.15,addon=True):
        super(ResNet18, self).__init__()
        assert len(input_size) == 4, "input must be in 3d with the corresponding number of channels"
        self.stemNet1 = nn.Sequential(
            nn.Conv3d(input_size[0], 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2,padding=1),
        )
        self.use_addon = bool(addon)
        self.layer1 = self._makelayer(64, 64, 2, strides=1)
        self.layer2 = self._makelayer(64, 128, 2, strides=2)
        self.layer3 = self._makelayer(128, 256, 2, strides=2)
        self.layer4 = self._makelayer(256, 512, 2, strides=2)

        if self.use_addon:
            self.addon = self._makelayer(512, 512, 3, strides=1)
            for m in self.addon.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        self.spatial_dropout = nn.Dropout3d(p=spatial_dropout)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fnn = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
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
        if self.use_addon:
            out = self.addon(out)
            out = self.spatial_dropout(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fnn(out)
        return out
    
'''
class ResNet18_param(nn.Module):
    def __init__(self, input_size,dropout=0.7):
        super(ResNet18_param, self).__init__()
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

        self.avgpool = nn.AvgPool3d((6,7,6),stride=1)
        self.fnn = nn.Sequential(
            nn.Linear(512+9, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1)
        )

    def _makelayer(self, in_channels, out_channels, blocks, strides=1):
        layers = []
        layers.append(Residual_block(in_channels, out_channels, strides))
        for _ in range(1, blocks):
            layers.append(Residual_block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x,param):
        out = self.stemNet1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = torch.cat([out, param], dim=1)
        out = self.fnn(out)
        return out

'''
'''
In the function ResNet18_param_add_template, we will add template features (the template will be transformed by residual parameters we predicted) as another channel to the input 
and then cancatenate the template features with the output of the ResNet before passing it to the fully connected layers. 
This way, we can leverage both the image features and the template features for classification.
'''
'''    
class ResNet18_param_add_template(nn.Module):
    def __init__(self, input_size,dropout=0.7):
        super(ResNet18_param_add_template, self).__init__()
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

        self.avgpool = nn.AvgPool3d((6,7,6),stride=1)
        self.fnn = nn.Sequential(
            nn.Linear(512+512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1)
        )

    def _makelayer(self, in_channels, out_channels, blocks, strides=1):
        layers = []
        layers.append(Residual_block(in_channels, out_channels, strides))
        for _ in range(1, blocks):
            layers.append(Residual_block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, x2):
        # Process the image features
        out = self.stemNet1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)

        # Process the template features
        out2 = self.stemNet1(x2)
        out2 = self.layer1(out2)
        out2 = self.layer2(out2)
        out2 = self.layer3(out2)
        out2 = self.layer4(out2)
        out2 = self.avgpool(out2)

        out = out.view(out.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out = torch.cat([out, out2], dim=1)  # Concatenate image and template features
        out = self.fnn(out)
        return out


# fcn (use addon with full connected network)
class ResNet18_param_template_fcn(nn.Module):
    def __init__(self, input_size, dropout=0.3, spatial_dropout=0.15, istemplate=True):
        super().__init__()
        assert len(input_size) == 4, "input must be in 3d with the corresponding number of channels"
        self.istemplate = istemplate
        self.channel = 2 if self.istemplate else 1

        self.stemNet1 = nn.Sequential(
            nn.Conv3d(input_size[0], 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._makelayer(64, 64, 2, strides=1)
        self.layer2 = self._makelayer(64, 128, 2, strides=2)
        self.layer3 = self._makelayer(128, 256, 2, strides=2)
        self.layer4 = self._makelayer(256, 512, 2, strides=2)
        
        # decide to just add more resnet layer
        self.addon = self._makelayer(512 * self.channel, 512 * self.channel, 3, strides=1)

        for m in self.addon.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.spatial_dropout = nn.Dropout3d(p=spatial_dropout)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fnn = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(p=dropout),
            nn.Linear(512 * self.channel, 1),
        )

    def _makelayer(self, in_channels, out_channels, blocks, strides=1):
        layers = [Residual_block(in_channels, out_channels, strides)]
        for _ in range(1, blocks):
            layers.append(Residual_block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, x2=None):
        out = self.stemNet1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.istemplate:
            if x2 is None:
                raise ValueError("x2 must be provided when istemplate=True")
            out2 = self.stemNet1(x2)
            out2 = self.layer1(out2)
            out2 = self.layer2(out2)
            out2 = self.layer3(out2)
            out2 = self.layer4(out2)
            out = torch.cat([out, out2], dim=1)

        out = self.addon(out)
        out = self.spatial_dropout(out)
        out = self.avgpool(out)
        out = self.fnn(out)
        return out
'''