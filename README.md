# MKD-LMF: Multimodal Knowledge Distillation for Gastric Adenocarcinoma Classification from Whole-Slide Images

<p align="center">
  <a href="https://github.com/helomelo1/MKD-LMF">
    <img src="https://img.shields.io/badge/CVIP-2026-blue" alt="CVIP 2026"/>
  </a>
  <a href="https://github.com/helomelo1/MKD-LMF/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="PyTorch 2.0+"/>
</p>

> **Multimodal Knowledge Distillation for Gastric Adenocarcinoma Classification from Whole-Slide Images**  
> Shrihari Dumbre and Bikash Santra  
> Indian Institute of Technology Jodhpur  
> CVIP 2026

---

## Overview

This repository provides the official PyTorch implementation of **MKD-LMF**, a lightweight multimodal knowledge distillation framework for classifying gastric adenocarcinoma (GA) subtypes from whole-slide images (WSIs).

**Key idea:** A teacher model fuses WSI patch features with pathology report captions via [Low-Rank Multimodal Fusion (LMF)](https://aclanthology.org/P18-1209/). A student model then distils this multimodal knowledge to enable accurate **image-only** inference at test time—no pathology report needed.

---

## Method

### Two-Stage Framework

**Stage 1 — Multimodal Teacher Training**

- Each WSI is partitioned into patches; patch embeddings are extracted via [Phikon](https://huggingface.co/owkin/phikon) (ViT-B/16, `d_v = 768`) and mean-pooled to a bag-level representation `z_img`.
- The pathology report caption is encoded via [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (`d_t = 384`) to give `z_txt`.
- `z_img` and `z_txt` are fused via LMF (rank `r = 128`) into a joint representation `z_fused ∈ R^128`.
- A linear classifier is trained on `z_fused` with cross-entropy loss.

**Stage 2 — Knowledge Distillation for Image-Only Inference**

- A residual MLP Knowledge Distiller `H` is trained to map image-only features `z_img → ẑ_fused`, approximating the teacher's fused representation.
- The shared (frozen) classifier is applied to `ẑ_fused` for final prediction.
- Combined distillation loss:

$$\mathcal{L}_{distill} = \mathcal{L}_{CE} + \lambda_{feat}\,\mathcal{L}_{feat}(\hat{z}_{fused},\, z_{fused}) + \lambda_{KD}\,\mathcal{L}_{KD}(\hat{y}_S,\, \hat{y}_T)$$

where `λ_feat = 1.0`, `λ_KD = 0.5`, `τ = 2`.

### Architecture

| Component | Model | Output Dim |
|---|---|---|
| Image Encoder | Phikon (ViT-B/16), frozen | 768 |
| Text Encoder | BGE-small-en-v1.5, frozen | 384 |
| LMF Module | rank=128 | 128 |
| Knowledge Distiller | Residual MLP (768→512→512→128) | 128 |
| Classifier (shared) | Linear + ReLU + Dropout(0.3) | 3 |

---

## Installation

```bash
git clone https://github.com/helomelo1/MKD-LMF.git
cd MKD-LMF
conda create -n mkd-lmf python=3.10 -y
conda activate mkd-lmf
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
transformers>=4.38.0
sentence-transformers>=2.7.0
huggingface-hub
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
h5py
pillow
```

---

## Dataset

We use the [PatchGastric](https://github.com/med-air/PatchGastric) dataset. Download it and set up the directory as follows:

```
data/
└── PatchGastric/
    ├── patches/          # pre-extracted patches (300×300 px, 20× magnification)
    │   ├── <wsi_id>/
    │   │   ├── patch_0001.png
    │   │   └── ...
    ├── captions.csv      # WSI-level diagnostic captions
    └── labels.csv        # WSI-level subtype labels
```

We use **3 subtypes** (following CITE and PathM3):
- `0` — Well-differentiated tubular adenocarcinoma
- `1` — Moderately differentiated tubular adenocarcinoma
- `2` — Poorly differentiated adenocarcinoma

Data split follows prior work: **20% train / 40% val / 40% test**.

---

## Usage

### 1. Pre-extract Image Features

```bash
python extract_features.py \
  --patch_dir data/PatchGastric/patches \
  --output_dir data/PatchGastric/features \
  --encoder phikon
```

### 2. Stage 1 — Train the Teacher (Multimodal)

```bash
python train_teacher.py \
  --feature_dir data/PatchGastric/features \
  --caption_csv data/PatchGastric/captions.csv \
  --label_csv data/PatchGastric/labels.csv \
  --lmf_rank 128 \
  --epochs 20 \
  --lr 1e-4 \
  --save_dir checkpoints/teacher
```

### 3. Stage 2 — Train the Student (Knowledge Distillation)

```bash
python train_student.py \
  --feature_dir data/PatchGastric/features \
  --label_csv data/PatchGastric/labels.csv \
  --teacher_ckpt checkpoints/teacher/best.pth \
  --lambda_feat 1.0 \
  --lambda_kd 0.5 \
  --temperature 2.0 \
  --epochs 50 \
  --lr 1e-4 \
  --save_dir checkpoints/student
```

### 4. Evaluate (Image-Only Inference)

```bash
python evaluate.py \
  --feature_dir data/PatchGastric/features \
  --label_csv data/PatchGastric/labels.csv \
  --student_ckpt checkpoints/student/best.pth \
  --teacher_ckpt checkpoints/teacher/best.pth
```

---

## Repository Structure

```
MKD-LMF/
├── models/
│   ├── lmf.py              # Low-Rank Multimodal Fusion module
│   ├── knowledge_distiller.py  # Residual MLP Knowledge Distiller
│   ├── teacher.py          # Teacher network (image + text + LMF + classifier)
│   └── student.py          # Student network (image + KD + shared classifier)
├── data/
│   └── dataset.py          # PatchGastric dataset loader
├── extract_features.py     # Pre-extract Phikon patch features
├── train_teacher.py        # Stage 1 training script
├── train_student.py        # Stage 2 training script
├── evaluate.py             # Evaluation script
├── utils/
│   ├── losses.py           # CE, Smooth L1, KL distillation losses
│   └── metrics.py          # Accuracy, DBI computation
├── requirements.txt
└── README.md
```

---

## Hardware

Experiments were run on:
- GPU: NVIDIA RTX 6000 ADA (48 GB)
- RAM: 512 GB
- CPU: Intel Xeon Gold 6426Y

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{dumbre2026mkdlmf,
  title={Multimodal Knowledge Distillation for Gastric Adenocarcinoma Classification from Whole-Slide Images},
  author={Dumbre, Shrihari and Santra, Bikash},
  booktitle={Proceedings of the Conference on Computer Vision and Image Processing (CVIP)},
  year={2026},
  institution={Indian Institute of Technology Jodhpur}
}
```

---

## Acknowledgements

We thank the authors of [PatchGastric](https://github.com/med-air/PatchGastric), [Phikon](https://huggingface.co/owkin/phikon), [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), and [PathM3](https://arxiv.org/abs/2403.08967) for their open-source contributions.

---

## License

This project is released under the [MIT License](LICENSE).
