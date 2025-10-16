import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

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

