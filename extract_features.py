
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModel
from sentence_transformers import SentenceTransformer


class PatchDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def extract_histo_features(image_dir, csv_path, output_dir, patch_batch_size=256):
    """Extract per-scan histopathology features using Phikon (ViT-B, 768-d)."""
    model = AutoModel.from_pretrained("owkin/phikon")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    # Build scan_id -> image paths mapping
    scan_to_images = defaultdict(list)
    for fname in os.listdir(image_dir):
        if fname.endswith(".jpg"):
            scan_id = fname.split("_")[0]
            scan_to_images[scan_id].append(os.path.join(image_dir, fname))

    df = pd.read_csv(csv_path)
    unique_labels = sorted(df["subtype"].unique())
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    scan_to_label = dict(zip(df["id"], df["subtype"]))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for scan_id, img_paths in tqdm(scan_to_images.items(),
                                    desc="Histo features (Phikon)"):
        out_path = os.path.join(output_dir, f"{scan_id}.pt")
        if os.path.exists(out_path):
            continue

        dataset = PatchDataset(img_paths, transform)
        # num_workers=4 speeds up image loading, pin_memory=True speeds up CPU->GPU transfer
        loader = DataLoader(dataset, batch_size=patch_batch_size, num_workers=4, pin_memory=True)

        all_features = []
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = model(pixel_values=batch)
                features = outputs.last_hidden_state[:, 0, :]  # CLS token
            all_features.append(features.cpu())

        scan_features = torch.cat(all_features, dim=0)
        label_str = scan_to_label.get(scan_id)
        label_int = label_to_int[label_str] if label_str else -1

        torch.save({
            "scan_id": scan_id,
            "label": label_int,
            "features": scan_features,
        }, out_path)


def extract_caption_features(csv_path, output_dir):
    """Extract per-scan caption features using BAAI/bge-small-en-v1.5 (384-d)."""
    df = pd.read_csv(csv_path)
    scan_to_caption = dict(zip(df["id"], df["text"]))

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    for scan_id, caption in tqdm(scan_to_caption.items(),
                                  desc="Caption features (mpnet)"):
        out_path = os.path.join(output_dir, f"{scan_id}.pt")
        if os.path.exists(out_path):
            continue

        with torch.no_grad():
            embedding = model.encode(caption, convert_to_tensor=True)

        torch.save({
            "scan_id": scan_id,
            "embedding": embedding.cpu(),
        }, out_path)


def build_parser():
    p = argparse.ArgumentParser(description="Extract features for both modalities.")
    p.add_argument("--image_dir",           type=str, default="dataset/patches_captions")
    p.add_argument("--label_csv",           type=str, default="captions_final.csv")
    p.add_argument("--histo_feature_dir",   type=str, default="features_phikon")
    p.add_argument("--caption_feature_dir", type=str, default="features_mpnet")
    p.add_argument("--patch_batch_size",    type=int, default=64)
    return p


def main():
    args = build_parser().parse_args()

    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("Image encoder:  Phikon (owkin/phikon) → 768-d")
    print("Text encoder:   BAAI/bge-small-en-v1.5 → 384-d")

    extract_histo_features(
        args.image_dir, args.label_csv,
        args.histo_feature_dir, args.patch_batch_size,
    )
    extract_caption_features(args.label_csv, args.caption_feature_dir)
    print("Done.")


if __name__ == "__main__":
    main()