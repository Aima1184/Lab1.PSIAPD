import torch.nn as nn
import torchvision.models as models

def get_alexnet_model(num_classes):
    alexnet = models.alexnet(pretrained=True)
    alexnet.classifier[6] = nn.Linear(4096, num_classes)
    return alexnet
