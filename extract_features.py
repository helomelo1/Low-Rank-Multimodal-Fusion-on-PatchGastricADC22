import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PatchGastricMILDataset
from torchvision import models
from tqdm import tqdm

image_dir = ""
label_csv = ""
feature_dir = ""

dataset = PatchGastricMILDataset(image_dir, label_csv)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

patch_batch_size = 64

for bag_tensor, label, scan_id in tqdm(dataloader):
    scan_id = scan_id[0]
    bag_tensor = bag_tensor.squeeze(0).to(device)  # shape: (#patches, 3, 224, 224)

    all_features = []
    for i in range(0, bag_tensor.size(0), patch_batch_size):
        patch_batch = bag_tensor[i : i + patch_batch_size]
        with torch.no_grad():
            features = model(patch_batch)  # shape: (batch_size, 2048, 1, 1)
            features = features.squeeze(-1).squeeze(-1)      # shape: (batch_size, 2048)
        all_features.append(features.cpu())

    scan_features = torch.cat(all_features, dim=0)

    # Save each scan's features to a file
    torch.save({
        "scan_id": scan_id,
        "label": label.item(),
        "features": scan_features
    }, os.path.join(feature_dir, f"{scan_id}.pt"))