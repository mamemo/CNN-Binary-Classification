"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    Definition of Models

    * Implemented Models:
        - ResNet 18
        - EfficientNet B4
"""


from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


def resnet18():
    """
        efficientnet ResNet 18 model definition.
    """

    model = models.resnet18(pretrained=True)

    # To freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # New output layers
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model


def efficientnet():
    """
        efficientnet EfficientNet B4 model definition.
    """

    model = EfficientNet.from_pretrained('efficientnet-b4')

    # To freeze layers
    for param in model.parameters():
        param.requires_grad = False

    model._fc = nn.Linear(in_features=model._fc.in_features, out_features=2)

    return model
