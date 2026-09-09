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
