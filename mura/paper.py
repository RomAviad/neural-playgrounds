from torch.nn import Linear, Sigmoid
from torch.nn import functional as F
from torchvision.models.densenet import DenseNet


class MuraDenseNet(DenseNet):
    def __init__(self, growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=1):
        super().__init__(growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes)
        self.mura_classifier = Linear(self.classifier.in_features, num_classes, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.mura_classifier(out)
        out = self.sigmoid(out)
        return out
