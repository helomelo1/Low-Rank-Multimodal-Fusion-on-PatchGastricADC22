import torch
import torch.nn as nn
from query_based_transformer import QueryTransformer
from modules import CorrelationModule, ClassificationModule

class PathM3(nn.Module):
    def __init__(self, num_classes, d_model, nhead, num_layers, num_queries, patch_dim):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.correlation_module = CorrelationModule(d_model=d_model, nhead=nhead)

        self.query_transformer = QueryTransformer(
            num_queries=num_queries,
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead
        )

        self.classifier = ClassificationModule(d_model=d_model, num_classes=num_classes)

    
    def forward(self, image_embeddings, caption_tokens=None):
        image_embeddings = self.patch_proj(image_embeddings)
        fused_image_embeddings = self.correlation_module(image_embeddings)
        query_outputs = self.query_transformer(fused_image_embeddings, caption_tokens)
        logits = self.classifier(query_outputs)

        return logits