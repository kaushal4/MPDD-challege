import torch
import torch.nn as nn
import math
from typing import Optional

# Assumes PositionalEncoding class is available (defined in previous answers)
# Or define it here:
class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model, requires_grad=False)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Shape: [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: Tensor, shape [batch_size, seq_len, embedding_dim]"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# --- Attentional Fusion Model ---
class AttentionalFusionTransformer(nn.Module):
    """
    Processes audio/video with Transformers, then fuses aggregated outputs
    with personalized features using an attention mechanism (Transformer Encoder Layer).
    """
    def __init__(self,
                 audio_dim: int,
                 video_dim: int,
                 pers_dim: int,
                 num_classes: int,
                 # Encoder params
                 transformer_embed_dim: int = 128,
                 transformer_nhead: int = 4,
                 transformer_num_layers: int = 2,
                 transformer_dropout: float = 0.2,
                 # Fusion params
                 fusion_nhead: int = 4,        # Heads for the fusion layer
                 fusion_dropout: float = 0.2,    # Dropout for the fusion layer
                 # MLP params
                 mlp_hidden_dim: int = 256,
                 mlp_dropout: float = 0.5,
                 # Other
                 max_seq_len: int = 26 # Use the value from your config
                 ):
        super().__init__()

        if transformer_embed_dim % transformer_nhead != 0 or \
           transformer_embed_dim % fusion_nhead != 0:
             raise ValueError(f"transformer_embed_dim ({transformer_embed_dim}) must be divisible by "
                              f"both transformer_nhead ({transformer_nhead}) and fusion_nhead ({fusion_nhead})")

        self.transformer_embed_dim = transformer_embed_dim
        self.pers_dim = pers_dim

        # --- Input Projections ---
        self.audio_proj = nn.Linear(audio_dim, transformer_embed_dim)
        self.video_proj = nn.Linear(video_dim, transformer_embed_dim)
        # Project personalized features to the same dimension for fusion
        self.pers_proj = nn.Linear(pers_dim, transformer_embed_dim)

        # --- Positional Encoding ---
        self.pos_encoder = PositionalEncoding(transformer_embed_dim, transformer_dropout, max_seq_len)

        # --- Modality Type Embeddings (Learnable) ---
        # One embedding for each modality (A, V, P) before fusion attention
        self.modality_embeddings = nn.Parameter(torch.randn(3, 1, transformer_embed_dim)) # Shape [NumModalities, 1, EmbedDim]

        # --- Audio Transformer Encoder ---
        audio_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim, nhead=transformer_nhead,
            dim_feedforward=transformer_embed_dim * 4, dropout=transformer_dropout,
            activation='gelu', batch_first=True, norm_first=True # norm_first often more stable
        )
        self.audio_transformer_encoder = nn.TransformerEncoder(audio_encoder_layer, num_layers=transformer_num_layers)

        # --- Video Transformer Encoder ---
        video_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim, nhead=transformer_nhead,
            dim_feedforward=transformer_embed_dim * 4, dropout=transformer_dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.video_transformer_encoder = nn.TransformerEncoder(video_encoder_layer, num_layers=transformer_num_layers)

        # --- Fusion Attention Layer (using a Transformer Encoder Layer) ---
        # Treats the 3 modalities (A_agg, V_agg, P_proj) as a sequence of length 3
        self.fusion_norm = nn.LayerNorm(transformer_embed_dim) # Norm before attention
        self.fusion_attention_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim, nhead=fusion_nhead,
            dim_feedforward=transformer_embed_dim * 4, # Can be smaller here if needed
            dropout=fusion_dropout,
            activation='gelu', batch_first=True, norm_first=True
        )

        # --- Final MLP Classifier ---
        # Input dimension is the output dim of the fusion layer
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(transformer_embed_dim),
            nn.Linear(transformer_embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, num_classes)
        )

        self._init_weights()
        # Avoid printing during Optuna runs
        # self._print_info()

    def _print_info(self):
        print(f"Initialized AttentionalFusionTransformer:")
        print(f"  - Input Dims: Audio={self.audio_proj.in_features}, Video={self.video_proj.in_features}, Pers={self.pers_dim}")
        print(f"  - Encoder Transformer: Embed={self.transformer_embed_dim}, Heads={self.audio_transformer_encoder.layers[0].self_attn.num_heads}, Layers={len(self.audio_transformer_encoder.layers)}, Dropout={self.audio_transformer_encoder.layers[0].dropout.p:.2f}")
        print(f"  - Fusion Transformer Layer: Heads={self.fusion_attention_layer.self_attn.num_heads}, Dropout={self.fusion_attention_layer.dropout.p:.2f}")
        print(f"  - Final MLP: Hidden={self.mlp_head[1].out_features}, Dropout={self.mlp_head[3].p:.2f}")
        print(f"  - Output Classes: {self.mlp_head[-1].out_features}")


    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                 nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                 nn.init.zeros_(param)
            elif 'modality_embeddings' in name:
                 nn.init.normal_(param, mean=0.0, std=0.02) # Initialize modality embeddings

    def aggregate_features(self, features: torch.Tensor, method: str = 'mean') -> torch.Tensor:
        """Aggregates features over the sequence length dimension (dim=1)."""
        if features.ndim != 3: raise ValueError(f"Expected 3D input [B, L, D], got {features.ndim}D")
        if method == 'mean': return torch.mean(features, dim=1)
        # Add other methods like 'cls' if needed
        else: raise ValueError(f"Unknown aggregation method: {method}")

    def forward(self, A_feat: torch.Tensor, V_feat: torch.Tensor, P_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_feat (torch.Tensor): Audio features [B, L_a, D_a].
            V_feat (torch.Tensor): Video features [B, L_v, D_v].
            P_feat (torch.Tensor): Personalized features [B, D_p].
        Returns:
            torch.Tensor: Output logits [B, num_classes].
        """
        batch_size = A_feat.size(0)
        if V_feat.size(0) != batch_size or P_feat.size(0) != batch_size:
            raise ValueError("Batch sizes of inputs do not match!")
        if P_feat.ndim == 1: P_feat = P_feat.unsqueeze(0) # Handle batch size 1

        # 1. Project Inputs
        A_proj = self.audio_proj(A_feat) # [B, L_a, E]
        V_proj = self.video_proj(V_feat) # [B, L_v, E]
        P_proj = self.pers_proj(P_feat)  # [B, E]

        # 2. Add Positional Encoding (only to sequences)
        A_in = self.pos_encoder(A_proj) # [B, L_a, E]
        V_in = self.pos_encoder(V_proj) # [B, L_v, E]

        # 3. Pass sequences through Transformer Encoders
        A_encoded_seq = self.audio_transformer_encoder(A_in) # [B, L_a, E]
        V_encoded_seq = self.video_transformer_encoder(V_in) # [B, L_v, E]

        # 4. Aggregate sequence outputs
        A_agg = self.aggregate_features(A_encoded_seq, method='mean') # [B, E]
        V_agg = self.aggregate_features(V_encoded_seq, method='mean') # [B, E]

        # 5. Prepare inputs for Fusion Attention
        # Unsqueeze to add sequence dimension (L=1) for stacking
        A_agg_seq = A_agg.unsqueeze(1) # [B, 1, E]
        V_agg_seq = V_agg.unsqueeze(1) # [B, 1, E]
        P_proj_seq = P_proj.unsqueeze(1) # [B, 1, E]

        # Add modality type embeddings (shape [3, 1, E] broadcasted to [B, 1, E])
        # Ensure embeddings are on the correct device
        mod_A = self.modality_embeddings[0].unsqueeze(0).expand(batch_size, -1, -1)
        mod_V = self.modality_embeddings[1].unsqueeze(0).expand(batch_size, -1, -1)
        mod_P = self.modality_embeddings[2].unsqueeze(0).expand(batch_size, -1, -1)

        A_f = A_agg_seq + mod_A
        V_f = V_agg_seq + mod_V
        P_f = P_proj_seq + mod_P

        # Concatenate along sequence dimension -> [B, 3, E]
        fusion_input_unnorm = torch.cat([A_f, V_f, P_f], dim=1)
        fusion_input = self.fusion_norm(fusion_input_unnorm) # Apply LayerNorm before attention

        # 6. Pass through Fusion Attention Layer
        # Input shape [B, L=3, E], Output shape [B, L=3, E]
        fusion_output_seq = self.fusion_attention_layer(fusion_input)

        # 7. Aggregate fusion output (e.g., mean pooling over the 3 modalities)
        fused_representation = torch.mean(fusion_output_seq, dim=1) # [B, E]

        # 8. Pass through final MLP classifier
        logits = self.mlp_head(fused_representation) # [B, num_classes]

        return logits