import torch
import torch.nn as nn
from torchvision import models

class DiceDetector(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights='DEFAULT')
        self.backbone = nn.Sequential(
            resnet.conv1, 
            resnet.bn1, 
            resnet.relu, 
            resnet.maxpool,
            resnet.layer1, 
            resnet.layer2, 
            resnet.layer3
        )
        
        self.cls_head = nn.Conv2d(256, 7, kernel_size=3, padding=1)
        self.reg_head = nn.Conv2d(256, 3, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.backbone(x)
        return self.cls_head(features), self.reg_head(features)

class DiceLoss(nn.Module):
    def __init__(self, bg_weight=1e-4):
        super().__init__()
        weights = torch.ones(7)
        weights[6] = bg_weight
        self.cls_criterion = nn.CrossEntropyLoss(weight=weights)
        self.reg_criterion = nn.L1Loss(reduction='none')

    def forward(self, pred_cls, pred_reg, target_cls, target_reg, mask):
        loss_cls = self.cls_criterion(pred_cls, target_cls)
        
        reg_error = self.reg_criterion(pred_reg, target_reg)
        masked_error = reg_error * mask.unsqueeze(1)
        
        loss_reg = masked_error.sum() / (mask.sum() * 3 + 1e-6)
        
        return loss_cls + loss_reg