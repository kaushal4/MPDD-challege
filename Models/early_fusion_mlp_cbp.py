import torch
import torch.nn as nn
import torch.fft
import math
import torch.nn.functional as F



class CompactBilinearPooling(nn.Module):
    """
    Compact Bilinear Pooling layer using Count Sketch approximation via scatter_add_.
    """
    def __init__(self, input_dim1, input_dim2, output_dim, sum_pool=False):
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool

        # Generate random projection indices and signs
        for i, d in enumerate([input_dim1, input_dim2]):
            h = torch.randint(0, output_dim, (d,), dtype=torch.int64) # Indices must be Long
            s = (torch.randint(0, 2, (d,)) * 2 - 1).float()
            self.register_buffer(f'h{i+1}', h)
            self.register_buffer(f's{i+1}', s)

    def _sketch(self, x, h, s):
        """ Applies Count Sketch using hashing (h) and signs (s). """
        batch_size = x.size(0)
        output_sketch = torch.zeros(batch_size, self.output_dim, device=x.device, dtype=x.dtype)

        x_signed = x * s # Shape: [batch_size, input_dim]

        index = h.repeat(batch_size, 1)
        output_sketch.scatter_add_(dim=1, index=index, src=x_signed)

        return output_sketch # Shape: [batch_size, output_dim]

    def forward(self, x1, x2):
        """
        x1 shape: [batch_size, input_dim1]
        x2 shape: [batch_size, input_dim2]
        """
        # --- Sketching using Count Sketch ---
        sketch1 = self._sketch(x1, self.h1, self.s1) # Shape: [batch, output_dim]
        sketch2 = self._sketch(x2, self.h2, self.s2) # Shape: [batch, output_dim]

        # --- Approximate Outer Product using FFT ---
        fft1 = torch.fft.fft(sketch1, dim=1) # FFT along the output_dim dimension
        fft2 = torch.fft.fft(sketch2, dim=1)

        fft_product = fft1 * fft2
        cbp_output = torch.fft.ifft(fft_product, dim=1) # IFFT along the output_dim dimension

        cbp_real = cbp_output.real # Shape: [batch, output_dim]

        if self.sum_pool:
            cbp_real = cbp_real.sum(dim=1)

        # --- Normalization (Signed Sqrt + L2) ---
        eps = 1e-9 # Epsilon for numerical stability
        cbp_norm_sign = torch.sign(cbp_real)
        cbp_norm_sqrt = torch.sqrt(torch.abs(cbp_real) + eps)
        cbp_normalized = cbp_norm_sign * cbp_norm_sqrt

        # L2 norm per sample vector
        norm = torch.norm(cbp_normalized, p=2, dim=1, keepdim=True).clamp(min=eps) # Clamp norm to avoid division by zero
        cbp_final = cbp_normalized / norm # Shape: [batch, output_dim] (if sum_pool=False)

        if torch.isnan(cbp_final).any():
            print("Warning: NaN detected after CBP normalization. Input might have been zero.")
            cbp_final = torch.nan_to_num(cbp_final, nan=0.0)


        return cbp_final


class EarlyFusionMLPWithCBP(nn.Module):
    """
    An Early Fusion MLP model that uses Compact Bilinear Pooling (CBP)
    to fuse aggregated sequential audio/video features and personalized features,
    followed by a shallow MLP for classification.
    """
    def __init__(self, audio_dim, video_dim, pers_dim, cbp_output_dim, hidden_dim, num_classes, dropout_rate=0.5):
        """
        Initializes the EarlyFusionMLPWithCBP model.
        Args are the same as before.
        """
        super().__init__()

        if not all(isinstance(d, int) and d > 0 for d in [audio_dim, video_dim, pers_dim, cbp_output_dim, hidden_dim, num_classes]):
             raise ValueError("Invalid non-positive or non-integer dimension provided to EarlyFusionMLPWithCBP")

        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.pers_dim = pers_dim
        self.cbp_output_dim = cbp_output_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.cbp_av = CompactBilinearPooling(audio_dim, video_dim, cbp_output_dim, sum_pool=False)
        self.cbp_ap = CompactBilinearPooling(audio_dim, pers_dim, cbp_output_dim, sum_pool=False)
        self.cbp_vp = CompactBilinearPooling(video_dim, pers_dim, cbp_output_dim, sum_pool=False)

        self.fused_dim = cbp_output_dim * 3

        # --- MLP Layers ---
        self.fc1 = nn.Linear(self.fused_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

        print(f"Initialized EarlyFusionMLPWithCBP:")
        print(f"  - Input Dims: Audio={audio_dim}, Video={video_dim}, Pers={pers_dim}")
        print(f"  - CBP Output Dim (per pair): {cbp_output_dim}")
        print(f"  - Total Fused Dim (Input to MLP): {self.fused_dim}")
        print(f"  - MLP Hidden Dim: {hidden_dim}")
        print(f"  - Output Classes: {num_classes}")
        print(f"  - Dropout Rate: {dropout_rate}")

    def aggregate_features(self, features):
        """Aggregates features over the sequence length dimension (dim=1) using mean pooling."""
        if features.ndim == 3:
            return torch.mean(features, dim=1)
        elif features.ndim == 2:
            return features
        else:
            raise ValueError(f"Unsupported feature dimension: {features.ndim}. Expected 2 or 3.")

    def forward(self, A_feat, V_feat, P_feat):
        """ Performs the forward pass of the model using CBP fusion. """
        # 1. Aggregate Audio and Video features
        A_feat_agg = self.aggregate_features(A_feat)
        V_feat_agg = self.aggregate_features(V_feat)

        # Ensure P_feat is 2D
        if P_feat.ndim == 1: P_feat = P_feat.unsqueeze(0)
        elif P_feat.ndim != 2: raise ValueError(f"Unexpected P_feat dim: {P_feat.ndim}.")

        # Optional Dimension checks
        if A_feat_agg.shape[1] != self.audio_dim: raise ValueError(f"Audio dim mismatch")
        if V_feat_agg.shape[1] != self.video_dim: raise ValueError(f"Video dim mismatch")
        if P_feat.shape[1] != self.pers_dim: raise ValueError(f"Pers dim mismatch")

        # 2. Pairwise Compact Bilinear Pooling
        fused_av = self.cbp_av(A_feat_agg, V_feat_agg)
        fused_ap = self.cbp_ap(A_feat_agg, P_feat)
        fused_vp = self.cbp_vp(V_feat_agg, P_feat)

        # 3. Concatenate the fused features
        final_fused_features = torch.cat((fused_av, fused_ap, fused_vp), dim=1)

        # 4. Pass through MLP
        x = self.fc1(final_fused_features)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits