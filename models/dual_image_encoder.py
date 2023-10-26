from torch import nn
import torch.nn.functional as F
import torch
from .base_image_encoder import BaseImageEncoder


class DualImageEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super(DualImageEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.first_encoder = BaseImageEncoder(latent_dim)
        self.second_encoder = BaseImageEncoder(latent_dim)
        self.fc1 = nn.Linear(latent_dim*2, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, latent_dim)

    def forward(self, img1, img2):
        img1 = self.first_encoder(img1)
        img2 = self.second_encoder(img2)
        x = torch.cat((img1, img2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
