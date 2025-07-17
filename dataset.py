import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from collections import defaultdict

class PatchGastricMILDataset(Dataset):
    def __init__(self, image_dir, label_csv, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(label_csv)

        # Column names in your CSV
        scan_col = "id"      # <<< CHANGE THIS if needed
        label_col = "subtype"       # <<< CHANGE THIS if needed

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Group patch paths by scan_id
        self.scan_to_images = defaultdict(list)
        for fname in os.listdir(image_dir):
            if fname.endswith(".jpg"):
                scan_id = fname.split("_")[0]  # Extract scan_id from file name
                self.scan_to_images[scan_id].append(os.path.join(image_dir, fname))

        # Keep only those scan_ids that appear in both CSV and image set
        self.scan_ids = [sid for sid in self.labels_df[scan_col] if sid in self.scan_to_images]

        # Convert labels to numerical format
        unique_labels = self.labels_df[label_col].unique()
        self.label_to_int = {label: i for i, label in enumerate(unique_labels)}
        self.label_map = dict(zip(self.labels_df[scan_col], self.labels_df[label_col].map(self.label_to_int)))


    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, idx):
        scan_id = self.scan_ids[idx]
        img_paths = self.scan_to_images[scan_id]

        images = []
        for path in img_paths:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            images.append(img)

        bag_tensor = torch.stack(images)  # Shape: (num_patches, 3, H, W)
        label = torch.tensor(self.label_map[scan_id], dtype=torch.long)

        return bag_tensor, label, scan_id