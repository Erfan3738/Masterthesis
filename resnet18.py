import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, use_feature=False):
        super(ResNet18, self).__init__()
        # Load the pre-trained ResNet18 model
        self.resnet18 = models.resnet18(pretrained=False)
        self.use_feature = use_feature

        # Remove the final fully connected layer (classifier)
        self.features = nn.Sequential(*list(self.resnet18.children())[:-1])

        # If not using features, add the classifier back
        if not self.use_feature:
            self.fc = self.resnet18.fc

    def forward(self, x):
        if self.use_feature:
            # Extract features from the last convolutional layer
            x = self.features(x)
            # Flatten the output
            x = torch.flatten(x, 1)
            return x
        else:
            # Use the full ResNet18 for classification
            x = self.resnet18(x)
            return x
