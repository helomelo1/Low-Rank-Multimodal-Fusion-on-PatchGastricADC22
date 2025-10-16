import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

image_dir = ""
label_csv = ""
histo_feature_dir = ""
caption_feature_dir = ""

class LMF(nn.Module):
    def __init__(self, rank=4, hidden_dims=[2048, 384], output_dim=128):
        super(LMF, self).__init__()
        self.rank = rank
        self.dim1 = hidden_dims[0] + 1  # +1 for bias
        self.dim2 = hidden_dims[1] + 1
        self.output_dim = output_dim

        self.factor1 = nn.Parameter(torch.randn(rank, self.dim1, output_dim))
        self.factor2 = nn.Parameter(torch.randn(rank, self.dim2, output_dim))

    def forward(self, x1, x2):
        # x1: [B, dim1], e.g. image features
        # x2: [B, dim2], e.g. caption features
        B = x1.size(0)

        ones = x1.new_ones(B, 1)
        x1 = torch.cat([x1, ones], dim=1)  # [B, dim1+1]
        x2 = torch.cat([x2, ones], dim=1)  # [B, dim2+1]

        fusion = 0
        for i in range(self.rank):
            proj1 = torch.matmul(x1, self.factor1[i])  # [B, output_dim]
            proj2 = torch.matmul(x2, self.factor2[i])  # [B, output_dim]
            fusion += proj1 * proj2  # Hadamard product

        return fusion  # [B, output_dim]


class MultimodalClassifier(nn.Module):
    def __init__(self, rank=4, fusion_dim=128, num_classes=2):
        super().__init__()
        self.lmf = LMF(rank=rank, hidden_dims=[2048, 384], output_dim=fusion_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, num_classes)
        )

    def forward(self, img_feat, cap_feat):
        fused = self.lmf(img_feat, cap_feat)
        return self.classifier(fused)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_l = 10000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_l, dim))

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class AttentionPooling(nn.Module):
    def __init__(self, dim=2048, head=4):
        super().__init__()
        self.pos_enc = PositionalEncoding(dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=head, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        x = self.pos_enc(x)
        x = self.encoder(x)

        return x.mean(dim=1)

# Training Pipeline
model = MultimodalClassifier(rank=64, fusion_dim=128, num_classes=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
attention_pool = AttentionPooling()
attention_pool.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

df = pd.read_csv("captions_filtered.csv")

train_df, test_df = train_test_split(
    df, test_size=0.4, stratify=df["subtype"], random_state=42
)
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

label_to_idx = {label: idx for idx, label in enumerate(sorted(train_df["subtype"].unique()))}
num_classes = len(label_to_idx)

for epoch in range(10):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"Epoch {epoch + 1}"):
        scan_id = row["id"]
        label = row["subtype"]

        try:
            img_patches = torch.load(f"{histo_feature_dir}/{scan_id}.pt")["features"]
            img_feat = attention_pool(img_patches.unsqueeze(0).squeeze(0))
            cap_feat = torch.load(f"{caption_feature_dir}/{scan_id}.pt")["embedding"]
        except Exception as e:
            print(f"Skipping {scan_id}: {e}")
            continue

        img_feat = img_feat.unsqueeze(0).to(device)
        cap_feat = cap_feat.unsqueeze(0).to(device)
        label = torch.tensor([label_to_idx[label]]).to(device)

        logits = model(img_feat, cap_feat)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = torch.argmax(logits, dim=1).item()
        all_preds.append(pred)
        all_labels.append(label.item())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch + 1} | Train Loss: {total_loss:.4f} | Train Accuracy: {acc:.2%}")

# Testing Pipeline
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for _, row in test_df.iterrows():
        scan_id = row["id"]
        label = row["subtype"]

        try:
            img_feat = torch.load(f"{histo_feature_dir}/{scan_id}.pt")["features"].mean(dim=0)
            cap_feat = torch.load(f"{caption_feature_dir}/{scan_id}.pt")["embedding"]
        except:
            continue

        img_feat = img_feat.unsqueeze(0).to(device)
        cap_feat = cap_feat.unsqueeze(0).to(device)

        logits = model(img_feat, cap_feat)
        pred = torch.argmax(logits, dim=1).item()

        all_preds.append(pred)
        all_labels.append(label_to_idx[label])

test_acc = accuracy_score(all_labels, all_preds)
print(f"Final Test Accuracy: {test_acc:.2%}")