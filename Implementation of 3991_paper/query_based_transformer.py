import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    
    def forward(self, query, text_embed, image_features):
        # SELF ATTENTION BLOCK
        if text_embed is not None: 
            self_attn_input = torch.cat([query, text_embed], dim = 1)
        else:
            self_attn_input = query

        q_norm = self.norm1(query)
        self_attn_output, _ = self.self_attn(query=q_norm, key=self_attn_input, value=self_attn_input)
        query = query + self.dropout1(self_attn_output)

        # CROSS ATTENTION BLOCK
        q_norm = self.norm2(query)
        cross_attn_output, _ = self.cross_attn(query=q_norm, key=image_features, value=image_features)
        query = query + self.dropout2(cross_attn_output)

        # FEED FORWARD NETWORK BLOCK
        q_norm = self.norm3(query)
        ffn_output = self.feed_forward(q_norm)
        query = query + self.dropout3(ffn_output)

        return query
    

class QueryTransformer(nn.Module):
    def __init__(self, num_queries, num_layers, d_model, nhead, vocab_size=None, max_seq_line=128):
        super().__init__()
        self.learnable_queries = nn.Embedding(num_queries, d_model)
        self.txt = vocab_size is not None
        if self.txt:
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.positional_embedding = nn.Embedding(max_seq_line, d_model)

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    
    def forward(self, image_features, caption_features=None):
        batch_size = image_features.shape[0]
        device = image_features.device

        query = self.learnable_queries.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        text_embeds = None
        if self.txt and caption_features is not None:
            token_embeds = self.token_embedding(caption_features)
            positions = torch.arange(0, caption_features.shape[0], device=device).unsqueeze(0)
            pos_embeds = self.positional_embedding(positions)
            text_embeds = token_embeds + pos_embeds

        for layer in self.layers:
            query = layer(query, text_embeds, image_features)

        return self.norm(query)