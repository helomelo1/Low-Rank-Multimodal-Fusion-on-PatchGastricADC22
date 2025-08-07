import torch
import torch.nn as nn

class CorrelationModule(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout:float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(d_model)

    
    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        attn_output, _  = self.self_attn(query=image_embeddings, key=image_embeddings, value=image_embeddings)
        x = image_embeddings + self.dropout(attn_output)
        x = self.norm(x)

        return x