import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np # Keep numpy for compatibility with the Dataset/Dataloader structure provided

# Helper Module: Positional Encoding
class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Transpose pe to shape [1, max_len, d_model] for easier addition with batch_first=True inputs
        self.register_buffer('pe', pe.transpose(0, 1)) # Shape: [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x.size(1) is the sequence length
        # self.pe[:, :x.size(1)] selects the positional encodings up to the length of the input sequence
        # The positional encodings are added to the input tensor x
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# Custom Cross-Modal Transformer Layer
class CrossModalTransformerLayer(nn.Module):
    """
    A single layer of the Cross-Modal Transformer Encoder.
    It includes self-attention for audio, self-attention for video,
    cross-attention (audio query, video key/value),
    cross-attention (video query, audio key/value),
    and feed-forward networks for both modalities.

    Uses pre-layer normalization (Norm -> Attention/FFN -> Dropout -> + Residual).
    """
    def __init__(self, embed_dim: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout

        # --- Self-Attention Components ---
        self.norm_a1 = nn.LayerNorm(embed_dim)
        self.self_attn_a = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.dropout_sa_a = nn.Dropout(dropout)

        self.norm_v1 = nn.LayerNorm(embed_dim)
        self.self_attn_v = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.dropout_sa_v = nn.Dropout(dropout)

        # --- Cross-Attention Components ---
        self.norm_a2 = nn.LayerNorm(embed_dim)
        self.norm_v_for_a_cross = nn.LayerNorm(embed_dim) # Norm V before A queries it
        self.cross_attn_a = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True) # A queries V
        self.dropout_ca_a = nn.Dropout(dropout)

        self.norm_v2 = nn.LayerNorm(embed_dim)
        self.norm_a_for_v_cross = nn.LayerNorm(embed_dim) # Norm A before V queries it
        self.cross_attn_v = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True) # V queries A
        self.dropout_ca_v = nn.Dropout(dropout)

        # --- Feed-Forward Components ---
        self.norm_a3 = nn.LayerNorm(embed_dim)
        self.linear1_a = nn.Linear(embed_dim, dim_feedforward)
        self.activation_a = nn.ReLU() # Consider nn.GELU as well
        self.dropout_ffn1_a = nn.Dropout(dropout)
        self.linear2_a = nn.Linear(dim_feedforward, embed_dim)
        self.dropout_ffn2_a = nn.Dropout(dropout)

        self.norm_v3 = nn.LayerNorm(embed_dim)
        self.linear1_v = nn.Linear(embed_dim, dim_feedforward)
        self.activation_v = nn.ReLU() # Consider nn.GELU as well
        self.dropout_ffn1_v = nn.Dropout(dropout)
        self.linear2_v = nn.Linear(dim_feedforward, embed_dim)
        self.dropout_ffn2_v = nn.Dropout(dropout)


    def forward(self, audio_src: torch.Tensor, video_src: torch.Tensor,
                audio_key_padding_mask: torch.Tensor = None,
                video_key_padding_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the cross-modal layer.

        Args:
            audio_src: Audio sequence tensor [batch, seq_len_a, embed_dim].
            video_src: Video sequence tensor [batch, seq_len_v, embed_dim].
            audio_key_padding_mask: Mask for audio padding [batch, seq_len_a].
            video_key_padding_mask: Mask for video padding [batch, seq_len_v].

        Returns:
            Tuple containing the updated audio and video sequences.
        """
        audio_res = audio_src
        video_res = video_src

        # 1. Self-Attention (Audio)
        norm_a1 = self.norm_a1(audio_res)
        sa_a_out, _ = self.self_attn_a(query=norm_a1, key=norm_a1, value=norm_a1,
                                       key_padding_mask=audio_key_padding_mask)
        audio_res = audio_res + self.dropout_sa_a(sa_a_out)

        # 1. Self-Attention (Video)
        norm_v1 = self.norm_v1(video_res)
        sa_v_out, _ = self.self_attn_v(query=norm_v1, key=norm_v1, value=norm_v1,
                                       key_padding_mask=video_key_padding_mask)
        video_res = video_res + self.dropout_sa_v(sa_v_out)

        # Intermediate states after self-attention (used for cross-attention keys/values)
        audio_after_sa = audio_res
        video_after_sa = video_res

        # 2. Cross-Attention (Audio queries Video)
        norm_a2 = self.norm_a2(audio_res) # Normalize audio residual for query
        norm_v_cross = self.norm_v_for_a_cross(video_after_sa) # Normalize video state for key/value
        ca_a_out, _ = self.cross_attn_a(query=norm_a2, key=norm_v_cross, value=norm_v_cross,
                                        key_padding_mask=video_key_padding_mask) # Use video mask here
        audio_res = audio_res + self.dropout_ca_a(ca_a_out)

        # 2. Cross-Attention (Video queries Audio)
        norm_v2 = self.norm_v2(video_res) # Normalize video residual for query
        norm_a_cross = self.norm_a_for_v_cross(audio_after_sa) # Normalize audio state for key/value
        ca_v_out, _ = self.cross_attn_v(query=norm_v2, key=norm_a_cross, value=norm_a_cross,
                                        key_padding_mask=audio_key_padding_mask) # Use audio mask here
        video_res = video_res + self.dropout_ca_v(ca_v_out)

        # 3. Feed-Forward (Audio)
        norm_a3 = self.norm_a3(audio_res)
        ffn_a = self.linear2_a(self.dropout_ffn1_a(self.activation_a(self.linear1_a(norm_a3))))
        audio_res = audio_res + self.dropout_ffn2_a(ffn_a)

        # 3. Feed-Forward (Video)
        norm_v3 = self.norm_v3(video_res)
        ffn_v = self.linear2_v(self.dropout_ffn1_v(self.activation_v(self.linear1_v(norm_v3))))
        video_res = video_res + self.dropout_ffn2_v(ffn_v)

        return audio_res, video_res


# Main Model: Cross-Modal Transformer Encoder
class CrossModalTransformerEncoder(nn.Module):
    """
    Multimodal architecture using a Cross-Modal Transformer Encoder.

    Processes audio and video sequences through stacked CrossModalTransformerLayer
    layers, allowing interaction between modalities. Aggregates the final sequences
    and fuses them with personalized features before classification.
    """
    def __init__(self,
                 audio_dim: int,
                 video_dim: int,
                 pers_dim: int,
                 embed_dim: int,       # Common dimension for transformer layers
                 num_heads: int,       # Number of attention heads
                 num_layers: int,      # Number of CrossModalTransformerLayer blocks
                 dim_feedforward: int, # Dimension of the FFN inside transformer layers
                 num_classes: int,     # Number of output classes
                 max_seq_len: int = 10, # Max sequence length for positional encoding
                 dropout: float = 0.1):
        """
        Initializes the CrossModalTransformerEncoder model.

        Args:
            audio_dim: Input dimension of audio features.
            video_dim: Input dimension of video features.
            pers_dim: Input dimension of personalized features.
            embed_dim: The dimension transformers will work with.
            num_heads: Number of attention heads in MultiheadAttention.
            num_layers: Number of stacked cross-modal transformer layers.
            dim_feedforward: Hidden dimension of the feed-forward networks in transformer layers.
            num_classes: Number of output classes for classification.
            max_seq_len: Maximum sequence length expected (for PositionalEncoding).
            dropout: Dropout rate used throughout the model.
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.pers_dim = pers_dim

        # 1. Input Projection Layers (to match embed_dim)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)
        self.video_proj = nn.Linear(video_dim, embed_dim)

        # 2. Positional Encoding
        self.pos_encoder_a = PositionalEncoding(embed_dim, dropout, max_seq_len)
        self.pos_encoder_v = PositionalEncoding(embed_dim, dropout, max_seq_len)

        # 3. Stacked Cross-Modal Transformer Layers
        self.transformer_layers = nn.ModuleList([
            CrossModalTransformerLayer(embed_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # 4. Final Layer Normalization (optional but often helpful)
        self.final_norm_a = nn.LayerNorm(embed_dim)
        self.final_norm_v = nn.LayerNorm(embed_dim)

        # 5. Classifier Head
        # Input dimension: aggregated audio (embed_dim) + aggregated video (embed_dim) + personalized (pers_dim)
        self.classifier_input_dim = embed_dim + embed_dim + pers_dim
        # Example simple MLP head - could be made deeper
        self.fc1 = nn.Linear(self.classifier_input_dim, embed_dim) # Intermediate layer
        self.classifier_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim, num_classes) # Output layer

        # Optional: Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, A_feat: torch.Tensor, V_feat: torch.Tensor, P_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CrossModalTransformerEncoder.

        Args:
            A_feat: Audio features (batch, seq_len, audio_dim).
            V_feat: Video features (batch, seq_len, video_dim).
            P_feat: Personalized features (batch, pers_dim).

        Returns:
            Output logits (batch, num_classes).
        """
        # --- Input Checks and Preparation ---
        # Ensure inputs have 3 dimensions (batch, seq, feat) for sequences
        if A_feat.ndim != 3:
            raise ValueError(f"Expected A_feat to have 3 dimensions [batch, seq, dim], but got {A_feat.ndim}")
        if V_feat.ndim != 3:
             raise ValueError(f"Expected V_feat to have 3 dimensions [batch, seq, dim], but got {V_feat.ndim}")
        if P_feat.ndim != 2:
            # Handle potential case where batch size is 1 and loader drops dim
            if P_feat.ndim == 1 and A_feat.shape[0] == 1:
                P_feat = P_feat.unsqueeze(0)
            else:
                raise ValueError(f"Expected P_feat to have 2 dimensions [batch, dim], but got {P_feat.ndim}")

        # TODO: Implement Padding Mask Generation if needed
        # If your sequences might have padding based on original lengths before pad_or_truncate,
        # you would create boolean masks here (True where padded) and pass them to layers.
        # E.g., audio_key_padding_mask = (A_feat.sum(dim=-1) == 0) # Example if padding is zero vectors
        audio_key_padding_mask = None
        video_key_padding_mask = None

        # --- Feature Projection and Positional Encoding ---
        # Project features to the transformer's embedding dimension
        audio_emb = self.audio_proj(A_feat) # Shape: [batch, seq_len, embed_dim]
        video_emb = self.video_proj(V_feat) # Shape: [batch, seq_len, embed_dim]

        # Add positional encoding
        audio_input = self.pos_encoder_a(audio_emb)
        video_input = self.pos_encoder_v(video_emb)

        # --- Cross-Modal Transformer Layers ---
        audio_seq = audio_input
        video_seq = video_input
        for layer in self.transformer_layers:
            audio_seq, video_seq = layer(audio_seq, video_seq,
                                         audio_key_padding_mask=audio_key_padding_mask,
                                         video_key_padding_mask=video_key_padding_mask)

        # --- Final Normalization and Aggregation ---
        audio_output = self.final_norm_a(audio_seq) # Shape: [batch, seq_len, embed_dim]
        video_output = self.final_norm_v(video_seq) # Shape: [batch, seq_len, embed_dim]

        # Aggregate sequence outputs (e.g., mean pooling over sequence length)
        # Note: If using padding masks, masked mean pooling would be more accurate.
        audio_agg = torch.mean(audio_output, dim=1) # Shape: [batch, embed_dim]
        video_agg = torch.mean(video_output, dim=1) # Shape: [batch, embed_dim]

        # --- Fusion with Personalized Features ---
        fused_features = torch.cat((audio_agg, video_agg, P_feat), dim=1)
        # Shape: [batch, embed_dim + embed_dim + pers_dim]

        # --- Classifier ---
        x = F.relu(self.fc1(fused_features))
        x = self.classifier_dropout(x)
        logits = self.fc2(x) # Shape: [batch, num_classes]

        return logits