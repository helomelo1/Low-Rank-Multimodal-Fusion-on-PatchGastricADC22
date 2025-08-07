import os
import torch 
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from correlation_module import CorrelationModule

model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children)[:-1])

def extract_and_save_features(dataloader, device, feature_dir):
    patch_batch_size = 64
    for bag_tensor, label, scan_id in tqdm(dataloader):
        scan_id = scan_id[0]
        bag_tensor = bag_tensor.squeeze(0).to(device)

        all_features = []
        for i in range(0, bag_tensor.size(0), patch_batch_size):
            patch_batch = bag_tensor[i : i + patch_batch_size]
            with torch.no_grad():
                features = model(patch_batch)
                features = features.squeeze(-1).squeeze(-1)
            all_features.append(features.cpu())

        scan_features = torch.cat(all_features, dim=0)

        torch.save({
            "scan_id": scan_id,
            "label": label.item(),
            "features": scan_features
        }), os.path.join(feature_dir, f"{scan_id}.pt")