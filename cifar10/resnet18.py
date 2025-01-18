import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = F.batch_norm(
                input.view(-1, C * self.num_splits, H, W),
                running_mean_split,
                running_var_split,
                self.weight.repeat(self.num_splits),
                self.bias.repeat(self.num_splits),
                True,
                self.momentum,
                self.eps,
            ).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias, False, self.momentum, self.eps
            )

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, use_split_bn=False, num_splits=1, num_classes=10):
        super().__init__()

        # Define BatchNorm layer based on use_split_bn
        norm_layer = lambda num_features: (
            SplitBatchNorm(num_features, num_splits=num_splits)
            if use_split_bn
            else nn.BatchNorm2d(num_features)
        )

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()  # No max-pooling for CIFAR-10

        # ResNet layers
        self.layer1 = self._make_layer(BasicBlock, 64, 2, norm_layer=norm_layer)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, norm_layer=norm_layer)

        # Fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        norm_layer = norm_layer or nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, feature_only=False):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Feature extraction layers
        x = self.layer1(x)
        x = self.layer2(x)
        features = self.layer3(x)
        x = self.layer4(features)

        if feature_only:
            return features  # Return intermediate features

        # Classification layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
