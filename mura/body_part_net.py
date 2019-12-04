from torch.nn import Softmax, Linear
from torch.utils import model_zoo
from torchvision.models.resnet import Bottleneck, ResNet, model_urls


class BodyPartResNet(ResNet):
    def __init__(self):
        num_classes = 7
        layers = [3, 4, 6, 3]
        block = Bottleneck
        super().__init__(block, layers, num_classes)
        self.fc = Linear(512 * block.expansion, num_classes, bias=False)
        self.softmax = Softmax(-1)

    def forward(self, inputs):
        resnet_output = super().forward(inputs)
        out = self.softmax(resnet_output)
        return out


def load_pretrained_with_imagenet():
    """Load a BodyPartResNet model with imagenet weights"""
    model = BodyPartResNet()
    state_dict = model_zoo.load_url(model_urls['resnet50'])
    del state_dict["fc.bias"]
    del state_dict["fc.weight"]
    model.load_state_dict(state_dict, strict=False)
    return model
