import numpy as np
import torch
import torchvision

from matplotlib import pyplot as plt
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10

from resnet_torch.resnet_torch import ResNet34


def imshow(img):
    img = img / 2 + 0.5  # un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_cifar10_loader(root, train, download, transform, batch_size=4):
    shuffle = train
    dataset = CIFAR10(root=root, train=train, download=download, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def train_resnet(train_loader, num_epochs=2):
    """
    :param train_loader:
    :param num_epochs: number of epochs on the training set. default is 2 because training on my laptop takes forever
    :return:
    """
    net = ResNet34(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(num_epochs):
        # even 2 epochs take too long on my laptop; but I'd probably do ~50 for a ball-park estimation if the
        # model I
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    return net


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cifar_train_loader = get_cifar10_loader("./data", train=True, download=True, transform=transform)
    test_loader = get_cifar10_loader("./data", train=False, download=True, transform=transform)

    cifar_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = train_resnet(cifar_train_loader, num_epochs=2)

    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % cifar_classes[labels[j]] for j in range(4)))
    torch.save(net, "./model.pickle")