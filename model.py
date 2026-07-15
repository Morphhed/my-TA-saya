import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class SimCLRModel(nn.Module):
    def __init__(self, projection_dim=128):
        super(SimCLRModel, self).__init__()
        self.backbone = resnet50(pretrained=False)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() 
        
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        representation = self.backbone(x)
        projections = self.projection_head(representation)
        return projections

class NTXentLoss(nn.Module):
    def __init__(self, device, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)
        similarity_matrix = torch.matmul(z, z.T)
        
        labels = torch.arange(batch_size).to(self.device)
        labels = torch.cat([labels + batch_size - 1, labels], dim=0)
        
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        similarity_matrix = similarity_matrix / self.temperature
        
        return self.criterion(similarity_matrix, labels)