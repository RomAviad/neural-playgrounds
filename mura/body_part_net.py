from torchvision.models.resnet import resnet50


model = resnet50(pretrained=True, num_classes=7)
model.cuda()

