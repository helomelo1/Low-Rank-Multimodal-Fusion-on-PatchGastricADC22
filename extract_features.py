import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PatchGastricMILDataset
from torchvision import models
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

image_dir = ""
label_csv = ""
histo_feature_dir = ""
caption_feature_dir = ""


dataset = PatchGastricMILDataset(image_dir, label_csv)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

os.makedirs(histo_feature_dir, exist_ok = True)
def extract_histo_features(dataloader):
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
        out_path = os.path.join(histo_feature_dir, f"{scan_id}.pt")

        # Save each scan's features to a file
        torch.save({
            "scan_id": scan_id,
            "label": label.item(),
            "features": scan_features
        }, out_path)


os.makedirs(caption_feature_dir, exist_ok=True)
captions = pd.read_csv("captions_filtered.csv")
scan_col = "id"
caption_col = "text"
scan_to_caption_mapping = dict(zip(captions[scan_col], captions[caption_col]))

def extract_caption_features(dataloader):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model.eval()

    for _, _, scan_id in tqdm(dataloader):
        scan_id = scan_id[0]

        if scan_id not in scan_to_caption_mapping:
            continue

        caption = scan_to_caption_mapping[scan_id]

        # Skip if already processed
        out_path = f"features_per_caption/{scan_id}.pt"
        if os.path.exists(out_path):
            continue

        with torch.no_grad():
            embedding = model.encode(caption, convert_to_tensor=True)

        torch.save({
            "scan_id": scan_id,
            "embedding": embedding.cpu()
        }, out_path)