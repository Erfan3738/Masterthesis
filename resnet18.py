import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet18, self).__init__()
        # Load the ResNet-18 model from torchvision
        self.resnet18 = models.resnet18(pretrained=pretrained)
        
        # Modify the first convolutional layer and remove max pooling
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Identity()

        # Remove the final fully connected layer (we'll handle this in the forward pass)
        self.resnet18.fc = nn.Identity()

    def forward(self, x, use_feature=False):
        # Forward pass through the initial layers
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        # Forward pass through the ResNet blocks
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        # Global average pooling
        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)

        if use_feature:
            # Return the features before the final fully connected layer
            return x
        else:
            # Pass through the final fully connected layer (if needed)
            # Since we removed the original fc layer, you can add your own here
            # For example, if you want to use the default ResNet-18 output:
            x = self.resnet18.fc(x)
            return x
