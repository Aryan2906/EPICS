import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class CNNtoSNNConverter(nn.Module):
    def __init__(self, beta=0.95, slope=25):
        super(CNNtoSNNConverter, self).__init__()
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(16, 32, 5, 2, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Flatten(),
        )
        spike_grad = surrogate.fast_sigmoid(slope=slope)
        self.spike_converter = snn.Leaky(
            beta=beta,
            threshold=0.5,
            spike_grad=spike_grad,
            init_hidden=True,
            output=True,
        )

    def forward(self, x):
        spatial_features = self.cnn_extractor(x)
        spikes, membrane_potential = self.spike_converter(spatial_features)
        return spikes, membrane_potential

class DementiaSNN(nn.Module):
    def __init__(self, beta=0.95, slope=25, num_classes=4):
        super().__init__()

        # CNN Feature Extractor
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

        spike_grad = surrogate.fast_sigmoid(slope=slope)

        # Feature → Spike conversion
        self.lif1 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )

        # Classification Head
        self.fc1 = nn.Linear(128, 64)

        self.lif2 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )

        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):

        features = self.cnn_extractor(x)

        # Initialize membrane states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # First spike layer
        spk1, mem1 = self.lif1(features, mem1)

        # Hidden layer
        hidden = self.fc1(spk1)

        # Second spike layer
        spk2, mem2 = self.lif2(hidden, mem2)

        # Final classification
        logits = self.fc2(spk2)

        return logits