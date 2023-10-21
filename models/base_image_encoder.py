from torch import nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B4_Weights
import torchvision.models as models


class BaseImageEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super(BaseImageEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.model = models.efficientnet_b4(
            weights=EfficientNet_B4_Weights.DEFAULT)
        # unfreeze the last layer only
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        self.fc1 = nn.Linear(1000, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, latent_dim)

    def forward(self, x):
        x = self.model(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
