import torch
import torch.nn as nn


# ResNet feature extractor code
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identityDownsample = None, stride = 1):
        super(Block, self).__init__()

        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()
        self.identityDownsample = identityDownsample


    def forward(self, x):
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Adding the previous output to this layer
        if self.identityDownsample is not None:
            identity = self.identityDownsample(identity)

        x += identity
        x = self.relu(x)

        return x
    

class ResNet(nn.Module):
    def __init__(self, block, layer, img_channels):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Common beginning layers
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers
        self.layer1 = self._make_layer(block, layer[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layer[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layer[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layer[3], out_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        return x


    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identityDownsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identityDownsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4)
            )

        layers.append(block(self.in_channels, out_channels, identityDownsample, stride))
        self.in_channels = out_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    

def ResNet50(img_channels=3):
    return ResNet(Block, [3, 4, 6, 3], img_channels)


def ResNet101(img_channels=3):
    return ResNet(Block, [3, 4, 23, 3], img_channels)


def ResNet50(img_channels=3):
    return ResNet(Block, [3, 8, 36, 3], img_channels)
