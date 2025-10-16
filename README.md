# Low-Rank Multimodal Fusion on PatchGastricADC22

## 📜 Project Description

This project presents a deep learning pipeline for the classification of gastric adenocarcinoma subtypes by leveraging multimodal data. Specifically, it uses a **Low-Rank Multimodal Fusion (LMF)** model to combine information from histopathology images and their corresponding textual descriptions. The goal is to achieve a more accurate and robust classification by integrating both visual and textual features.

The model is designed to work with the **PatchGastricADC22 dataset**, classifying cancer subtypes based on patches from whole-slide images and their medical captions.

---

## 🚀 Key Features

* **Multimodal Fusion:** Fuses image and text data for a comprehensive analysis.
* **Low-Rank Multimodal Fusion (LMF):** An efficient technique for combining multimodal features.
* **Attention Pooling:** Uses a Transformer-based attention mechanism to aggregate image patch features.
* **Pretrained Models:** Leverages powerful pretrained models for feature extraction: **ResNet-50** for images and **Sentence-Transformers** for text.
* **End-to-End Pipeline:** Includes scripts for data loading, feature extraction, training, and evaluation.

---

## 📂 File Descriptions

* **`dataset.py`**: Contains the `PatchGastricMILDataset` class, a custom PyTorch dataset to load the image patches and labels.
* **`extract_features.py`**: A script to extract features from both image patches (using ResNet-50) and text captions (using Sentence-Transformers).
* **`pipeline.py`**: The main script that defines the model architecture (LMF, MultimodalClassifier, AttentionPooling) and includes the training and testing pipelines.
* **`requirements.txt`**: A list of all the Python libraries and dependencies required to run the project.
* **`captions_filtered.csv`**: The CSV file containing the `id`, `subtype`, and `text` for each sample in the dataset.

---

## ⚙️ Methodology

The pipeline follows these steps:

1.  **Feature Extraction:**
    * **Histopathology Images**: A pretrained **ResNet-50** model is used to extract a feature vector from each image patch.
    * **Text Captions**: A **SentenceTransformer** model is used to encode the descriptive captions into dense vector embeddings.

2.  **Attention Pooling:**
    * The features from all patches belonging to a single slide are fed into a **Transformer Encoder with positional encoding**.
    * This attention mechanism allows the model to weigh the importance of different patches, and the output is then mean-pooled to get a single representative feature vector for the entire slide.

3.  **Low-Rank Multimodal Fusion (LMF):**
    * The image feature vector and the caption feature vector are fused using the **LMF** model. This technique is an efficient way to model the complex interactions between the two modalities.

4.  **Classification:**
    * The fused feature vector is passed to a classifier, which is a feed-forward neural network, to predict the gastric adenocarcinoma subtype.

---

## 🔧 Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/helomelo1/low-rank-multimodal-fusion-on-patchgastricadc22.git](https://github.com/helomelo1/low-rank-multimodal-fusion-on-patchgastricadc22.git)
    cd low-rank-multimodal-fusion-on-patchgastricadc22
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃‍♀️ How to Run

1.  **Extract Features:**
    * Run the `extract_features.py` script to generate the feature files for both images and captions. You'll need to set the `image_dir` and `label_csv` paths in the script.
    ```bash
    python extract_features.py
    ```

2.  **Train the Model:**
    * Run the `pipeline.py` script to train the multimodal classifier.
    ```bash
    python pipeline.py
    ```
    This script will also split the data, train the model, and print the test accuracy.

---

## 📦 Dependencies

The main dependencies are:

* `torch`
* `torchvision`
* `pandas`
* `scikit-learn`
* `sentence-transformers`
* `tqdm`
* `Pillow`
* `matplotlib`