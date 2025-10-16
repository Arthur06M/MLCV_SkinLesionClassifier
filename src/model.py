from torchvision import models
import torch.nn as nn

def create_model(num_classes=7):
    
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
