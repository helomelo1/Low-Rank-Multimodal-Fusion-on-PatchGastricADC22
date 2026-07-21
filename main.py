"""
run_experiments.py

Experiment runner for:
    "Low-Rank Multimodal Fusion for Gastric Adenocarcinoma Subtype Classification"

Usage:
    python main.py --model full_method --epochs 10
    python main.py --model full_method --rank 4 --epochs 10
    python main.py --model late_fusion --epochs 10
    python main.py --model full_method --evaluate --checkpoint results/full_method_r64/checkpoint.pt
    python main.py --visualize tsne confusion_matrix attention
"""

import os
import json
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────
# Model Components
# ──────────────────────────────────────────────────────────────

class LowRankFusion(nn.Module):
    def __init__(self, dims, output_dim, rank):
        super().__init__()
        self.rank = rank
        self.factor_img = nn.Parameter(torch.empty(rank, dims[0] + 1, output_dim))
        self.factor_txt = nn.Parameter(torch.empty(rank, dims[1] + 1, output_dim))
        nn.init.xavier_uniform_(self.factor_img)
        nn.init.xavier_uniform_(self.factor_txt)

    def forward(self, x_img, x_txt):
        ones = x_img.new_ones(x_img.size(0), 1)
        x_img = torch.cat([x_img, ones], dim=1)
        x_txt = torch.cat([x_txt, ones], dim=1)
        return sum(
            torch.matmul(x_img, self.factor_img[r])
            * torch.matmul(x_txt, self.factor_txt[r])
            for r in range(self.rank)
        )


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=10_000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, dim))

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class AttentionPooling(nn.Module):
    def __init__(self, dim=768, nhead=4):
        super().__init__()
        self.pos_enc = PositionalEncoding(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x, return_weights=False):
        x = self.pos_enc(x)
        x = self.encoder(x)

        if return_weights:
            norms = x.norm(dim=-1)
            weights = norms / (norms.sum(dim=-1, keepdim=True) + 1e-8)
            return x.mean(dim=1), weights

        return x.mean(dim=1)


# ──────────────────────────────────────────────────────────────
# Model Variants
# ──────────────────────────────────────────────────────────────

class ImageOnlyClassifier(nn.Module):
    def __init__(self, img_dim=768, num_classes=3):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(img_dim, num_classes)

    def forward(self, img_feat, cap_feat=None):
        return self.fc(self.dropout(self.relu(img_feat)))

    def get_features(self, img_feat, cap_feat=None):
        return self.dropout(self.relu(img_feat))


class TextOnlyClassifier(nn.Module):
    def __init__(self, cap_dim=384, num_classes=3):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(cap_dim, num_classes)

    def forward(self, img_feat=None, cap_feat=None):
        return self.fc(self.dropout(self.relu(cap_feat)))

    def get_features(self, img_feat=None, cap_feat=None):
        return self.dropout(self.relu(cap_feat))


class ConcatFusionClassifier(nn.Module):
    def __init__(self, img_dim=768, cap_dim=384, num_classes=3):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(img_dim + cap_dim, num_classes)

    def forward(self, img_feat, cap_feat):
        fused = torch.cat([img_feat, cap_feat], dim=1)
        return self.fc(self.dropout(self.relu(fused)))

    def get_features(self, img_feat, cap_feat):
        fused = torch.cat([img_feat, cap_feat], dim=1)
        return self.dropout(self.relu(fused))


class MultimodalLMFClassifier(nn.Module):
    def __init__(self, img_dim=768, cap_dim=384, fusion_dim=128,
                 rank=64, num_classes=3):
        super().__init__()
        self.lmf = LowRankFusion(
            dims=[img_dim, cap_dim], output_dim=fusion_dim, rank=rank,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(fusion_dim, num_classes)

    def forward(self, img_feat, cap_feat):
        fused = self.lmf(img_feat, cap_feat)
        return self.fc(self.dropout(self.relu(fused)))

    def get_features(self, img_feat, cap_feat):
        fused = self.lmf(img_feat, cap_feat)
        return self.dropout(self.relu(fused))


# ──────────────────────────────────────────────────────────────
# Data Utilities
# ──────────────────────────────────────────────────────────────

def load_split(csv_path, train_size=0.2, seed=42):
    df = pd.read_csv(csv_path)
    train_df, rest_df = train_test_split(
        df, train_size=train_size, stratify=df["subtype"], random_state=seed,
    )
    val_df, test_df = train_test_split(
        rest_df, test_size=0.5, stratify=rest_df["subtype"], random_state=seed,
    )
    label_to_idx = {
        lbl: i for i, lbl in enumerate(sorted(df["subtype"].unique()))
    }
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        label_to_idx
    )


def load_features(scan_id, histo_dir, caption_dir):
    img_data = torch.load(
        os.path.join(histo_dir, f"{scan_id}.pt"),
        map_location="cpu", weights_only=False,
    )
    cap_data = torch.load(
        os.path.join(caption_dir, f"{scan_id}.pt"),
        map_location="cpu", weights_only=False,
    )
    return img_data["features"], cap_data["embedding"]


# ──────────────────────────────────────────────────────────────
# Image Feature Aggregation
# ──────────────────────────────────────────────────────────────

def aggregate_patches(patches, device, mode="mean", attention_pool=None,
                      return_weights=False):
    """
    Aggregate patch features into a single image-level vector.
    mode: "mean" | "attention" | "single"
    """
    if mode == "single":
        return patches[0].unsqueeze(0).to(device), None

    if mode == "attention":
        inp = patches.unsqueeze(0).to(device)
        if return_weights:
            feat, w = attention_pool(inp, return_weights=True)
            return feat, w
        return attention_pool(inp), None

    return patches.mean(dim=0, keepdim=True).to(device), None


# ──────────────────────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, attention_pool, df, label_to_idx, histo_dir,
                    caption_dir, optimizer, criterion, device,
                    pool_mode="mean"):
    model.train()
    if attention_pool is not None:
        attention_pool.train()

    total_loss = 0.0
    all_preds, all_labels = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
        scan_id, label_str = row["id"], row["subtype"]

        try:
            patches, cap_feat = load_features(scan_id, histo_dir, caption_dir)
        except Exception:
            continue

        img_feat, _ = aggregate_patches(
            patches, device, mode=pool_mode, attention_pool=attention_pool,
        )
        cap_feat = cap_feat.unsqueeze(0).to(device)
        label = torch.tensor([label_to_idx[label_str]], device=device)

        logits = model(img_feat, cap_feat)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(logits.argmax(1).item())
        all_labels.append(label.item())

    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return total_loss, acc


@torch.no_grad()
def evaluate(model, attention_pool, df, label_to_idx, histo_dir,
             caption_dir, device, pool_mode="mean",
             collect_embeddings=False, collect_attention=False):
    model.eval()
    if attention_pool is not None:
        attention_pool.eval()

    all_preds, all_labels = [], []
    embeddings, attention_records = [], []

    for _, row in df.iterrows():
        scan_id, label_str = row["id"], row["subtype"]

        try:
            patches, cap_feat = load_features(scan_id, histo_dir, caption_dir)
        except Exception:
            continue

        img_feat, attn_w = aggregate_patches(
            patches, device, mode=pool_mode, attention_pool=attention_pool,
            return_weights=collect_attention,
        )
        cap_feat = cap_feat.unsqueeze(0).to(device)

        logits = model(img_feat, cap_feat)
        all_preds.append(logits.argmax(1).item())
        all_labels.append(label_to_idx[label_str])

        if collect_embeddings:
            feat = model.get_features(img_feat, cap_feat)
            embeddings.append(feat.cpu().squeeze(0))
        if collect_attention and attn_w is not None:
            attention_records.append((scan_id, attn_w.cpu().squeeze(0)))

    metrics = {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall":    recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1":        f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }
    out = {"metrics": metrics, "preds": all_preds, "labels": all_labels}

    if collect_embeddings:
        out["embeddings"] = torch.stack(embeddings).numpy()
    if collect_attention:
        out["attention"] = attention_records
    return out


@torch.no_grad()
def evaluate_late_fusion(img_model, txt_model, df, label_to_idx,
                         histo_dir, caption_dir, device):
    img_model.eval()
    txt_model.eval()

    all_preds, all_labels = [], []

    for _, row in df.iterrows():
        scan_id, label_str = row["id"], row["subtype"]

        try:
            patches, cap_feat = load_features(scan_id, histo_dir, caption_dir)
        except Exception:
            continue

        img_feat = patches.mean(dim=0, keepdim=True).to(device)
        cap_feat = cap_feat.unsqueeze(0).to(device)

        p_img = F.softmax(img_model(img_feat), dim=1)
        p_txt = F.softmax(txt_model(cap_feat=cap_feat), dim=1)
        pred  = ((p_img + p_txt) / 2).argmax(1).item()

        all_preds.append(pred)
        all_labels.append(label_to_idx[label_str])

    return {
        "metrics": {
            "accuracy":  accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "recall":    recall_score(all_labels, all_preds, average="macro", zero_division=0),
            "f1":        f1_score(all_labels, all_preds, average="macro", zero_division=0),
        },
        "preds": all_preds,
        "labels": all_labels,
    }


# ──────────────────────────────────────────────────────────────
# Training Harness
# ──────────────────────────────────────────────────────────────

def train_and_evaluate(model, attention_pool, train_df, val_df, test_df,
                       label_to_idx, args, pool_mode="mean", tag=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if attention_pool is not None:
        attention_pool.to(device)

    params = list(model.parameters())
    if attention_pool is not None:
        params += list(attention_pool.parameters())

    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'─'*60}")
    print(f"  Training: {tag}")
    print(f"{'─'*60}")

    run_dir = os.path.join(args.output_dir, tag.replace(" ", "_"))
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    
    best_val_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(
            model, attention_pool, train_df, label_to_idx,
            args.histo_feature_dir, args.caption_feature_dir,
            optimizer, criterion, device, pool_mode=pool_mode,
        )
        
        val_res = evaluate(
            model, attention_pool, val_df, label_to_idx,
            args.histo_feature_dir, args.caption_feature_dir,
            device, pool_mode=pool_mode,
        )
        val_f1 = val_res["metrics"]["f1"]
        print(f"  Epoch {epoch:>3d}/{args.epochs}  loss={loss:.4f}  train_acc={acc:.4f}  val_f1={val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt = {"model": model.state_dict()}
            if attention_pool is not None:
                ckpt["attention_pool"] = attention_pool.state_dict()
            torch.save(ckpt, ckpt_path)

    print(f"  Loading best checkpoint (val_f1={best_val_f1:.4f}) for final Test set evaluation...")
    model, attention_pool = _load_checkpoint(model, attention_pool, ckpt_path, device)

    result = evaluate(
        model, attention_pool, test_df, label_to_idx,
        args.histo_feature_dir, args.caption_feature_dir,
        device, pool_mode=pool_mode,
    )

    m = result["metrics"]
    print(f"  ── Test  acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
          f"rec={m['recall']:.4f}  f1={m['f1']:.4f}")

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)

    report = classification_report(
        result["labels"], result["preds"],
        target_names=sorted(label_to_idx, key=label_to_idx.get),
    )
    with open(os.path.join(run_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    print(f"  Saved → {run_dir}/")
    return result["metrics"], model, attention_pool


# ──────────────────────────────────────────────────────────────
# Experiment Dispatch
# ──────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "image_only", "text_only", "concat_fusion", "late_fusion", "full_method",
    "no_attn_pool", "no_lmf", "no_mil",
}

TAG_MAP = {
    "image_only":    "image_only",
    "text_only":     "text_only",
    "concat_fusion": "concat_fusion",
    "late_fusion":   "late_fusion",
    "no_attn_pool":  "ablation_no_attn_pool",
    "no_lmf":        "ablation_no_lmf",
    "no_mil":        "ablation_no_mil",
}


def get_tag(model_name, rank=64):
    if model_name == "full_method":
        return f"full_method_r{rank}"
    return TAG_MAP[model_name]


def run_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    train_df, val_df, test_df, label_to_idx = load_split(args.csv, seed=args.seed)
    nc = len(label_to_idx)
    name = args.model

    if name == "image_only":
        model = ImageOnlyClassifier(num_classes=nc)
        return train_and_evaluate(
            model, None, train_df, val_df, test_df, label_to_idx, args,
            pool_mode="mean", tag=get_tag(name),
        )

    if name == "text_only":
        model = TextOnlyClassifier(num_classes=nc)
        return train_and_evaluate(
            model, None, train_df, val_df, test_df, label_to_idx, args,
            pool_mode="mean", tag=get_tag(name),
        )

    if name == "concat_fusion":
        model = ConcatFusionClassifier(num_classes=nc)
        return train_and_evaluate(
            model, None, train_df, val_df, test_df, label_to_idx, args,
            pool_mode="mean", tag=get_tag(name),
        )

    if name == "late_fusion":
        seed_everything(args.seed)
        img_model = ImageOnlyClassifier(num_classes=nc)
        _, img_model, _ = train_and_evaluate(
            img_model, None, train_df, val_df, test_df, label_to_idx, args,
            pool_mode="mean", tag="late_fusion_img_branch",
        )

        seed_everything(args.seed)
        txt_model = TextOnlyClassifier(num_classes=nc)
        _, txt_model, _ = train_and_evaluate(
            txt_model, None, train_df, val_df, test_df, label_to_idx, args,
            pool_mode="mean", tag="late_fusion_txt_branch",
        )

        result = evaluate_late_fusion(
            img_model, txt_model, test_df, label_to_idx,
            args.histo_feature_dir, args.caption_feature_dir, device,
        )
        m = result["metrics"]
        print(f"\n  ── Late Fusion Test  acc={m['accuracy']:.4f}  "
              f"prec={m['precision']:.4f}  rec={m['recall']:.4f}  "
              f"f1={m['f1']:.4f}")

        run_dir = os.path.join(args.output_dir, "late_fusion")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(m, f, indent=2)

        return m, None, None

    if name == "full_method":
        model = MultimodalLMFClassifier(rank=args.rank, num_classes=nc)
        return train_and_evaluate(
            model, None, train_df, val_df, test_df, label_to_idx, args,
            pool_mode="mean", tag=get_tag(name, args.rank),
        )

    if name == "no_attn_pool":
        model = MultimodalLMFClassifier(rank=args.rank, num_classes=nc)
        return train_and_evaluate(
            model, None, train_df, val_df, test_df, label_to_idx, args,
            pool_mode="mean", tag=get_tag(name),
        )

    if name == "no_lmf":
        model = ConcatFusionClassifier(num_classes=nc)
        pool = AttentionPooling()
        return train_and_evaluate(
            model, pool, train_df, val_df, test_df, label_to_idx, args,
            pool_mode="attention", tag=get_tag(name),
        )

    if name == "no_mil":
        model = MultimodalLMFClassifier(rank=args.rank, num_classes=nc)
        return train_and_evaluate(
            model, None, train_df, val_df, test_df, label_to_idx, args,
            pool_mode="single", tag=get_tag(name),
        )

    raise ValueError(f"Unknown model: {name}")


# ──────────────────────────────────────────────────────────────
# Checkpoint Utilities
# ──────────────────────────────────────────────────────────────

def _load_checkpoint(model, attention_pool, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    if attention_pool is not None and "attention_pool" in ckpt:
        attention_pool.load_state_dict(ckpt["attention_pool"])
        attention_pool.to(device)
    return model, attention_pool


def _ensure_checkpoint(tag, args):
    path = os.path.join(args.output_dir, tag, "checkpoint.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Train the model first before running evaluation or visualisation."
        )
    return path


# ──────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────

def generate_tsne(args):
    """t-SNE: image-only vs text-only vs full method (3-panel)."""
    print("\nGenerating t-SNE…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_df, label_to_idx = load_split(args.csv, seed=args.seed)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    nc = len(label_to_idx)
    class_names = [idx_to_label[i] for i in range(nc)]

    # image-only embeddings
    ckpt = _ensure_checkpoint("image_only", args)
    img_model = ImageOnlyClassifier(num_classes=nc)
    img_model, _ = _load_checkpoint(img_model, None, ckpt, device)
    res_img = evaluate(
        img_model, None, test_df, label_to_idx,
        args.histo_feature_dir, args.caption_feature_dir, device,
        pool_mode="mean", collect_embeddings=True,
    )

    # text embeddings (raw sentence-transformer)
    txt_embeds, txt_labels = [], []
    for _, row in test_df.iterrows():
        try:
            _, cap = load_features(row["id"], args.histo_feature_dir, args.caption_feature_dir)
        except Exception:   
            continue
        txt_embeds.append(cap.numpy())
        txt_labels.append(label_to_idx[row["subtype"]])
    emb_txt = np.stack(txt_embeds)
    lbl_txt = np.array(txt_labels)

    # full method embeddings
    tag = f"full_method_r{args.rank}"
    ckpt = _ensure_checkpoint(tag, args)
    full_model = MultimodalLMFClassifier(rank=args.rank, num_classes=nc)
    full_model, _ = _load_checkpoint(full_model, None, ckpt, device)
    res_full = evaluate(
        full_model, None, test_df, label_to_idx,
        args.histo_feature_dir, args.caption_feature_dir, device,
        pool_mode="mean", collect_embeddings=True,
    )

    panels = [
        ("Image Only",  res_img["embeddings"],  np.array(res_img["labels"])),
        ("Text Only",   emb_txt,                lbl_txt),
        ("Full Method", res_full["embeddings"], np.array(res_full["labels"])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (title, emb, lbl) in zip(axes, panels):
        proj = TSNE(n_components=2, random_state=args.seed, perplexity=30).fit_transform(emb)
        for c in range(nc):
            mask = lbl == c
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       label=class_names[c], alpha=0.7, s=20)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("t-SNE of Learned Representations", fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(args.output_dir, "tsne_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def generate_confusion_matrix(args):
    print("\n▶ Generating confusion matrix …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_df, label_to_idx = load_split(args.csv, seed=args.seed)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    nc = len(label_to_idx)
    class_names = [idx_to_label[i] for i in range(nc)]

    tag = f"full_method_r{args.rank}"
    ckpt = _ensure_checkpoint(tag, args)
    model = MultimodalLMFClassifier(rank=args.rank, num_classes=nc)
    model, _ = _load_checkpoint(model, None, ckpt, device)

    result = evaluate(
        model, None, test_df, label_to_idx,
        args.histo_feature_dir, args.caption_feature_dir, device,
        pool_mode="mean",
    )

    cm = confusion_matrix(result["labels"], result["preds"])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(nc), yticks=range(nc),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted", ylabel="True",
        title="Confusion Matrix — Full Method",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(nc):
        for j in range(nc):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    path = os.path.join(args.output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def generate_attention_map(args):
    print("\nGenerating attention heatmap…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_df, label_to_idx = load_split(args.csv, seed=args.seed)
    nc = len(label_to_idx)

    tag = f"full_method_r{args.rank}"
    ckpt = _ensure_checkpoint(tag, args)
    model = MultimodalLMFClassifier(rank=args.rank, num_classes=nc)
    pool = AttentionPooling()
    model, pool = _load_checkpoint(model, pool, ckpt, device)

    result = evaluate(
        model, pool, test_df, label_to_idx,
        args.histo_feature_dir, args.caption_feature_dir, device,
        pool_mode="attention", collect_attention=True,
    )

    records = result["attention"]
    n_show = min(5, len(records))
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 3))
    if n_show == 1:
        axes = [axes]

    for ax, (sid, w) in zip(axes, records[:n_show]):
        w_np = w.numpy()
        ax.bar(range(len(w_np)), w_np, color="steelblue", width=1.0)
        ax.set_title(f"{sid[:12]}…", fontsize=10)
        ax.set_xlabel("Patch index")
        ax.set_ylabel("Attention weight")

    fig.suptitle("Per-Patch Attention Weights", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(args.output_dir, "attention_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


VISUALIZATIONS = {
    "tsne":             generate_tsne,
    "confusion_matrix": generate_confusion_matrix,
    "attention":        generate_attention_map,
}


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Experiment runner for LMF multimodal gastric ADC classification.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", type=str, choices=sorted(MODEL_REGISTRY))
    mode.add_argument("--visualize", type=str, nargs="+",
                      choices=list(VISUALIZATIONS.keys()))

    p.add_argument("--histo_feature_dir",   type=str, required=True)
    p.add_argument("--caption_feature_dir", type=str, required=True)
    p.add_argument("--csv",        type=str,   default="captions_filtered.csv")
    p.add_argument("--output_dir", type=str,   default="results")
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--rank",       type=int,   default=64)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--evaluate",   action="store_true")
    p.add_argument("--checkpoint", type=str,   default=None)

    return p


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Seed   : {args.seed}")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model:
        if args.evaluate:
            seed_everything(args.seed)
            _, test_df, label_to_idx = load_split(args.csv, seed=args.seed)
            nc = len(label_to_idx)

            # late fusion is a special case: two separate checkpoints
            if args.model == "late_fusion":
                img_ckpt = _ensure_checkpoint("late_fusion_img_branch", args)
                txt_ckpt = _ensure_checkpoint("late_fusion_txt_branch", args)

                img_model = ImageOnlyClassifier(num_classes=nc)
                img_model, _ = _load_checkpoint(img_model, None, img_ckpt, device)
                txt_model = TextOnlyClassifier(num_classes=nc)
                txt_model, _ = _load_checkpoint(txt_model, None, txt_ckpt, device)

                result = evaluate_late_fusion(
                    img_model, txt_model, test_df, label_to_idx,
                    args.histo_feature_dir, args.caption_feature_dir, device,
                )
                m = result["metrics"]
                print(f"\n  acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
                      f"rec={m['recall']:.4f}  f1={m['f1']:.4f}")
            else:
                tag = get_tag(args.model, args.rank)
                ckpt_path = args.checkpoint or _ensure_checkpoint(tag, args)

                pool_mode = "mean"
                pool = None

                if args.model == "image_only":
                    model = ImageOnlyClassifier(num_classes=nc)
                elif args.model == "text_only":
                    model = TextOnlyClassifier(num_classes=nc)
                elif args.model == "concat_fusion":
                    model = ConcatFusionClassifier(num_classes=nc)
                elif args.model == "full_method":
                    model = MultimodalLMFClassifier(rank=args.rank, num_classes=nc)
                    pool_mode = "mean"
                elif args.model == "no_attn_pool":
                    model = MultimodalLMFClassifier(rank=args.rank, num_classes=nc)
                elif args.model == "no_lmf":
                    model = ConcatFusionClassifier(num_classes=nc)
                    pool = AttentionPooling()
                    pool_mode = "attention"
                elif args.model == "no_mil":
                    model = MultimodalLMFClassifier(rank=args.rank, num_classes=nc)
                    pool_mode = "single"

                model, pool = _load_checkpoint(model, pool, ckpt_path, device)
                result = evaluate(
                    model, pool, test_df, label_to_idx,
                    args.histo_feature_dir, args.caption_feature_dir,
                    device, pool_mode=pool_mode,
                )
                m = result["metrics"]
                print(f"\n  acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
                      f"rec={m['recall']:.4f}  f1={m['f1']:.4f}")
        else:
            run_model(args)

    elif args.visualize:
        for vis_name in args.visualize:
            VISUALIZATIONS[vis_name](args)

    print("\nDone.")


if __name__ == "__main__":
    main()
