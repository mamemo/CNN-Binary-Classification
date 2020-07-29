''' Definition of Models

Implemented Models:
 - ResNet 18

'''
import torch.nn as nn
from torchvision import models

def resnet18():
    ''' ResNet 18 Model. '''

    model = models.resnet18(pretrained=True)

    # To freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # New output layers
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model