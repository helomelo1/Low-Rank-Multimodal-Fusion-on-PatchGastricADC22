import torch 
import torch.nn as nn
from torchvision import models

model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children)[:-1])
