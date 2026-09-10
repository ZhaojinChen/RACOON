import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np  # Make sure numpy is imported
from model.ResParam_model.PadMaxPool3d import PadMaxPool3d

# Don't apply Flatten class, directly flatten the feature in forward
class CNN_5layers(nn.Module):
    def __init__(self, input_size, dropout=0.7):
        super(CNN_5layers, self).__init__()

        # Main model framework, architecture refers to:
        # "Automatic quality control of brain T1-weighted magnetic resonance images for a clinical data warehouse"
        self.features = nn.Sequential(
            nn.Conv3d(in_channels=input_size[0], out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.Dropout3d(p=dropout),

            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.Dropout3d(p=dropout),

            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.Dropout3d(p=dropout),

            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.Dropout3d(p=dropout),

            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.Dropout3d(p=dropout)
        )

        # Calculate the flattened size after passing through CNN layers
        # Input size is (batch_size, channels, depth, height, width)
        # Here, we're assuming the input is 3D image data
        output_size = np.ceil(np.array(input_size[1:]) / 2 ** 5)
        flattened_size = int(np.prod(output_size) * 128)  # 128 is the output channel size after last conv layer

        # Fully connected layers for regression
        self.fnn = nn.Sequential(
            nn.Linear(flattened_size, 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, 9)  # Output layer for regression
        )

    def forward(self, x):
        # Forward pass through CNN layers
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Forward pass through the fully connected layers
        x = self.fnn(x)
        return x
