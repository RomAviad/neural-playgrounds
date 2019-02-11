from torch.nn import Module, BatchNorm2d, ReLU, Conv2d


class ResBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # stride kept in the "self" level for debugging
        self._stride = stride
        self.downsample = downsample
        self.conv1 = ResBlock.conv3x3(in_channels, out_channels, stride)
        self.bn1 = BatchNorm2d(out_channels)
        self.relu = ReLU(inplace=True)
        self.conv2 = ResBlock.conv3x3(out_channels, out_channels)
        self.bn2 = BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        # conv3x3 -> BatchNorm -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # conv3x3 -> BatchNorm
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(identity)

        # add the skip-connection to the output of the second BatchNorm, then run it through ReLU
        out += identity
        out = self.relu(out)
        return out

    @staticmethod
    def conv3x3(in_channels, out_channels, stride=1):
        return Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
