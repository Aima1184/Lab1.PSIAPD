import torch.nn as nn
import torchvision.models as models
from torchvision.models import AlexNet_Weights

def get_alexnet_model(num_classes):
    model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model
