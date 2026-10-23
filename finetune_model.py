import os
import torch
import torch.nn as nn
from torchvision import models
from finetune_CBAM import CBAM

class ResNet50WithCBAM(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50WithCBAM, self).__init__()
        # Membuat base model utuh agar urutan state_dict sama persis dengan SimCLR Anda
        base_model = models.resnet50(weights=None)
        
        # Ekstrak seluruh komponen konvolusi utama
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4  # Output channel 2048
        
        # Sisipkan modul CBAM tepat di akhir keluaran layer4
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

        # Fitur disaring oleh CBAM di ujung akhir backbone
        x = self.cbam(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_simclr_model_cbam(checkpoint_path, num_classes=2):
    model = ResNet50WithCBAM(num_classes=num_classes)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"File bobot SimCLR tidak ditemukan: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Bersihkan nama prefiks "module." jika sebelumnya dilatih dengan DataParallel
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    
    # strict=False digunakan karena model baru memiliki parameter cbam dan fc yang acak awal
    model.load_state_dict(state_dict, strict=False)
    return model

def get_baseline_model_cbam(num_classes=2):
    model = ResNet50WithCBAM(num_classes=num_classes)
    pretrained_base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    pretrained_dict = pretrained_base.state_dict()
    

    if 'fc.weight' in pretrained_dict:
        del pretrained_dict['fc.weight']
    if 'fc.bias' in pretrained_dict:
        del pretrained_dict['fc.bias']
    model.load_state_dict(pretrained_dict, strict=False)
    return model
