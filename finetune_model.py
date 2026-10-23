import os
import torch
import torch.nn as nn
from torchvision import models
from finetune_CBAM import CBAM

class ResNet50WithCBAM(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50WithCBAM, self).__init__()
        base_model = models.resnet50(weights=None)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.cbam = CBAM(in_planes=2048)
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def get_simclr_model_cbam(checkpoint_path, num_classes=2):
    model = ResNet50WithCBAM(num_classes=num_classes)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"File bobot SimCLR tidak ditemukan: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model_state = model.state_dict()
    compatible_state, skipped_keys = {}, []

    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped_keys.append(key)

    if len(compatible_state) == 0:
        raise RuntimeError("Tidak ada bobot backbone SimCLR yang cocok dengan model fine-tuning")

    result = model.load_state_dict(compatible_state, strict=False)

    print("\n========== SIMCLR CHECKPOINT ==========")
    print(f"Checkpoint                 : {checkpoint_path}")
    print(f"Tensor berhasil dimuat     : {len(compatible_state)}")
    print(f"Skipped checkpoint keys    : {skipped_keys}")
    print(f"Missing model keys         : {result.missing_keys}")
    print(f"Unexpected keys            : {result.unexpected_keys}")
    print("=======================================\n")

    return model

def get_baseline_model_cbam(num_classes=2):
    model = ResNet50WithCBAM(num_classes=num_classes)
    pretrained_base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    pretrained_dict = pretrained_base.state_dict()
    pretrained_dict.pop('fc.weight', None)
    pretrained_dict.pop('fc.bias', None)
    model.load_state_dict(pretrained_dict, strict=False)
    return model