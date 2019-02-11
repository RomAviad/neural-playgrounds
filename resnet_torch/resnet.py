from torch.nn import AvgPool2d, BatchNorm2d, Conv2d, Linear, MaxPool2d, Module, ReLU, Sequential

from .res_block import ResBlock


class ResNet34(Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64

        # First - Conv2D 7x7 with 2 stride -> BatchNorm2D -> ReLU -> MaxPool2D
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace=True)
        self.max_pool = MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Setup residual parts block groups
        self.layer64 = self._make_residual_layer(out_channels=64, num_blocks=3)
        self.layer128 = self._make_residual_layer(out_channels=128, num_blocks=4, stride=2)
        self.layer256 = self._make_residual_layer(out_channels=256, num_blocks=6, stride=2)
        self.layer512 = self._make_residual_layer(out_channels=512, num_blocks=3, stride=2)

        self.avg_pool = AvgPool2d(kernel_size=(1, 1))
        self.fc = Linear(512, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_residual_layer(self, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = Sequential(
                Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                BatchNorm2d(out_channels)
            )
        layers = [ResBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        layers.extend([ResBlock(self.in_channels, out_channels) for _ in range(1, num_blocks)])
        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer64(x)
        x = self.layer128(x)
        x = self.layer256(x)
        x = self.layer512(x)

        x = self.avg_pool(x)
        # flatten x
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
