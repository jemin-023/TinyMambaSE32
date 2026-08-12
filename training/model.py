import torch
import torch.nn as nn


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 2)):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)

        return x

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            # 257 → 129
            DepthwiseSeparableConv2d(2, 32),
            # 129 → 65
            DepthwiseSeparableConv2d(32, 64),
            # 65 → 33
            DepthwiseSeparableConv2d(64, 104),
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":

    model = Encoder()

    # Fake complex spectrogram
    # B=4, complex=2, T=1000, F=257
    x = torch.randn(4, 2, 1000, 257)

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)