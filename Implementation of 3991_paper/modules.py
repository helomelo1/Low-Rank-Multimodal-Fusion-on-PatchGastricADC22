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
    

class ClassificationModule(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, num_classes)


    def forward(self, query_outputs: torch.Tensor) ->  torch.Tensor:
        k_logits = self.classifier(query_outputs)
        final_logits = k_logits.mean(dim=1)
        
        return final_logits