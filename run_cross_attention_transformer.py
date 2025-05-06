# -*- coding: utf-8 -*-
"""
COMPLETE Training Script for Cross-Modal Transformer Encoder Model
- All parameters defined locally within the script.
- Audio and Video sequences interact via cross-attention layers.
- Uses CLS Token Aggregation for final A/V representations.
- Personalized features fused via concatenation + MLP at the end.
- Uses WeightedRandomSampler for handling class imbalance.
- Uses Warmup + Cosine LR Scheduling.
- Saves the best model based on Validation Macro F1-score.
- Implements Early Stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR
from collections import Counter
import numpy as np
import os
import json
import time
import traceback
import math
import torch.nn.functional as F
from datetime import datetime # For timestamping output file

# External Imports (Ensure these exist in your project structure)
from sklearn.metrics import accuracy_score, classification_report, f1_score
try:
    # Assuming these paths are correct relative to your script's location
    # If these are in subdirectories, ensure Python can find them (e.g., via __init__.py or adjusting PYTHONPATH)
    from DataSets.audioVisualDataset import AudioVisualDataset
    # Model defined below
    from Utils.focal_loss import FocalLoss
    from Utils.test_val_split import train_val_split1, train_val_split2
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure DataSets, Utils directories are correctly structured and importable.")
    exit(1)

# Optional: torchinfo for model summary
try:
    import torchinfo
except ImportError:
    torchinfo = None
    print("torchinfo not found. Install it (`pip install torchinfo`) for model summaries.")


# --- Model Definition (Cross-Modal Attention Version) ---

class PositionalEncoding(nn.Module):
    """Injects positional information into the sequence embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if d_model % 2 != 0: print("Warning: d_model in PositionalEncoding is odd.")

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model dimension carefully when assigning cosine
        if d_model % 2 != 0: pe[:, 0, 1::2] = torch.cos(position * div_term)[:,:pe[:, 0, 1::2].size(1)]
        else: pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2) # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class FeedForward(nn.Module):
    """Simple Feed Forward Network used in Transformer blocks."""
    def __init__(self, embed_dim, ff_dim, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.gelu(x) # Use GELU activation
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class CrossModalEncoderBlock(nn.Module):
    """
    A Transformer Encoder block with self-attention for each modality
    and cross-attention between modalities. Uses Pre-LayerNorm structure.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        # Using Pre-LN structure (Norm -> Attention/FFN -> Residual)
        self.norm_sa_a = nn.LayerNorm(embed_dim)
        self.norm_sa_v = nn.LayerNorm(embed_dim)
        self.norm_ca_a = nn.LayerNorm(embed_dim) # Norm before cross-attn query A
        self.norm_ca_v = nn.LayerNorm(embed_dim) # Norm before cross-attn query V
        self.norm_ffn_a = nn.LayerNorm(embed_dim)
        self.norm_ffn_v = nn.LayerNorm(embed_dim)

        self.self_attn_a = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_v = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.cross_attn_av = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True) # A queries V
        self.cross_attn_va = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True) # V queries A

        self.ffn_a = FeedForward(embed_dim, ff_dim, dropout)
        self.ffn_v = FeedForward(embed_dim, ff_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, audio_seq, video_seq, audio_mask=None, video_mask=None):
        # --- Self-Attention (Pre-LN) ---
        # Audio
        res_a = audio_seq
        sa_a_in = self.norm_sa_a(audio_seq)
        sa_a_out, _ = self.self_attn_a(sa_a_in, sa_a_in, sa_a_in, key_padding_mask=audio_mask)
        audio_seq = res_a + self.dropout(sa_a_out)

        # Video
        res_v = video_seq
        sa_v_in = self.norm_sa_v(video_seq)
        sa_v_out, _ = self.self_attn_v(sa_v_in, sa_v_in, sa_v_in, key_padding_mask=video_mask)
        video_seq = res_v + self.dropout(sa_v_out)

        # --- Cross-Attention (Pre-LN for Query) ---
        # Audio attends to Video (A -> V)
        res_a = audio_seq
        # Norm Audio (Query), use un-normed Video (Key/Value) as is standard in decoder-like cross-attn
        ca_a_in = self.norm_ca_a(audio_seq)
        ca_av_out, _ = self.cross_attn_av(ca_a_in, video_seq, video_seq, key_padding_mask=video_mask)
        audio_seq = res_a + self.dropout(ca_av_out)

        # Video attends to Audio (V -> A)
        res_v = video_seq
        ca_v_in = self.norm_ca_v(video_seq)
        ca_va_out, _ = self.cross_attn_va(ca_v_in, audio_seq, audio_seq, key_padding_mask=audio_mask)
        video_seq = res_v + self.dropout(ca_va_out)

        # --- Feed Forward (Pre-LN) ---
        # Audio
        res_a = audio_seq
        ffn_a_in = self.norm_ffn_a(audio_seq)
        ffn_a_out = self.ffn_a(ffn_a_in)
        audio_seq = res_a + self.dropout(ffn_a_out)

        # Video
        res_v = video_seq
        ffn_v_in = self.norm_ffn_v(video_seq)
        ffn_v_out = self.ffn_v(ffn_v_in)
        video_seq = res_v + self.dropout(ffn_v_out)

        return audio_seq, video_seq

class CrossModalTransformer(nn.Module):
    """
    Cross-Modal Transformer that processes audio and video sequences jointly
    through cross-modal attention blocks. Fuses with personalized features at the end.
    """
    def __init__(self, audio_dim, video_dim, pers_dim, embed_dim, nhead, num_cross_modal_layers,
                 dim_feedforward, hidden_dim_mlp, num_classes, max_len,
                 fusion_hidden_dim, dropout_rate=0.1):
        super().__init__()

        # --- Input Processing ---
        self.audio_projection = nn.Linear(audio_dim, embed_dim)
        self.video_projection = nn.Linear(video_dim, embed_dim)

        self.cls_token_audio = nn.Parameter(torch.zeros(1, 1, embed_dim)); nn.init.normal_(self.cls_token_audio, std=0.02)
        self.cls_token_video = nn.Parameter(torch.zeros(1, 1, embed_dim)); nn.init.normal_(self.cls_token_video, std=0.02)

        # Positional encoders for audio/video sequences (including CLS token)
        self.audio_pos_encoder = PositionalEncoding(embed_dim, dropout_rate, max_len + 1)
        self.video_pos_encoder = PositionalEncoding(embed_dim, dropout_rate, max_len + 1)

        self.input_dropout = nn.Dropout(dropout_rate)

        # --- Cross-Modal Encoder Blocks ---
        self.cross_modal_layers = nn.ModuleList([
            CrossModalEncoderBlock(embed_dim, nhead, dim_feedforward, dropout_rate)
            for _ in range(num_cross_modal_layers)
        ])
        # Final LayerNorm after cross-modal blocks (using Pre-LN within block now)
        # self.norm_a_final = nn.LayerNorm(embed_dim) # Not needed if using Pre-LN within blocks
        # self.norm_v_final = nn.LayerNorm(embed_dim)

        # --- Personalized Feature Branch ---
        self.mlp_pers_layers = nn.Sequential(
            nn.Linear(pers_dim, hidden_dim_mlp),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # Project personalized features to embed_dim for final fusion
        self.pers_projection = nn.Linear(hidden_dim_mlp, embed_dim)

        # --- Final Fusion & Classification Head (Using Concatenation + MLP) ---
        # Input dim: embed_dim (A CLS) + embed_dim (V CLS) + embed_dim (P projected)
        fusion_input_dim = embed_dim * 3
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, num_classes)
        )

    def forward(self, A_feat, V_feat, P_feat, A_mask=None, V_mask=None): # Add optional masks
        batch_size = A_feat.size(0)

        # --- Initial Processing ---
        A_proj = self.audio_projection(A_feat) # [B, SeqA, D]
        V_proj = self.video_projection(V_feat) # [B, SeqV, D]

        # Prepend CLS tokens
        cls_audio = self.cls_token_audio.expand(batch_size, -1, -1) # [B, 1, D]
        cls_video = self.cls_token_video.expand(batch_size, -1, -1) # [B, 1, D]
        A_seq = torch.cat((cls_audio, A_proj), dim=1) # [B, SeqA+1, D]
        V_seq = torch.cat((cls_video, V_proj), dim=1) # [B, SeqV+1, D]

        # Add Positional Encoding & Dropout
        A_seq = self.input_dropout(self.audio_pos_encoder(A_seq))
        V_seq = self.input_dropout(self.video_pos_encoder(V_seq))

        # --- Cross-Modal Encoding ---
        # TODO: Create padding masks if necessary based on original sequence lengths
        # Example: audio_padding_mask = (A_feat.sum(dim=-1) == 0) # Create mask BEFORE projection/CLS token
        for layer in self.cross_modal_layers:
            # Pass masks if they exist
            A_seq, V_seq = layer(A_seq, V_seq) # Add masks here if implemented: audio_mask=A_pad_mask, video_mask=V_pad_mask

        # Aggregate using CLS tokens (output after the last block)
        A_agg = A_seq[:, 0] # [B, D]
        V_agg = V_seq[:, 0] # [B, D]

        # --- Personalized Features ---
        if P_feat.ndim == 1: P_feat = P_feat.unsqueeze(0)
        elif P_feat.ndim != 2: raise ValueError(f"Unexpected P_feat shape: {P_feat.shape}")
        P_proc = self.mlp_pers_layers(P_feat)
        P_agg = self.pers_projection(P_proc) # [B, D]

        # --- Final Fusion (Concatenation) ---
        combined_features = torch.cat((A_agg, V_agg, P_agg), dim=1) # [B, D*3]
        logits = self.fusion_mlp(combined_features) # [B, num_classes]

        return logits


# --- Training and Evaluation Functions ---
# (Keep the train_epoch and evaluate functions from the previous iteration)
def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad_norm=1.0, scheduler=None):
    """Trains the model for one epoch with gradient clipping and optional LR step."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_samples = 0
    if dataloader is None or len(dataloader.dataset) == 0: return 0.0, 0.0

    processed_batches = 0
    for batch_idx, batch in enumerate(dataloader):
        if not isinstance(batch, dict) or 'A_feat' not in batch or 'V_feat' not in batch or 'personalized_feat' not in batch or 'emo_label' not in batch:
            continue
        try:
            audio_feat = batch['A_feat'].to(device, non_blocking=True)
            video_feat = batch['V_feat'].to(device, non_blocking=True)
            pers_feat = batch['personalized_feat'].to(device, non_blocking=True)
            labels = batch['emo_label'].to(device, non_blocking=True)

            batch_size = labels.size(0)
            if batch_size == 0: continue

            optimizer.zero_grad(set_to_none=True)
            # Pass None for masks initially, implement mask creation if needed
            outputs = model(audio_feat, video_feat, pers_feat, A_mask=None, V_mask=None)
            loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    WARNING: NaN or Inf loss encountered in train batch {batch_idx}. Skipping backward/step.")
                continue

            loss.backward()

            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()
            if scheduler is not None:
                 # Step step-based schedulers
                 if not hasattr(scheduler, 'is_epoch_scheduler') or not scheduler.is_epoch_scheduler:
                      scheduler.step()

            total_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_samples += batch_size
            processed_batches += 1

        except Exception as e:
            print(f"    ERROR during train batch {batch_idx}: {e}")
            # traceback.print_exc()
            continue

    if num_samples == 0: return 0.0, 0.0
    avg_loss = total_loss / num_samples
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, num_classes):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_samples = 0
    if dataloader is None or len(dataloader.dataset) == 0: return 0.0, 0.0, {}

    processed_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if not isinstance(batch, dict) or 'A_feat' not in batch or 'V_feat' not in batch or 'personalized_feat' not in batch or 'emo_label' not in batch:
                continue
            try:
                audio_feat = batch['A_feat'].to(device, non_blocking=True)
                video_feat = batch['V_feat'].to(device, non_blocking=True)
                pers_feat = batch['personalized_feat'].to(device, non_blocking=True)
                labels = batch['emo_label'].to(device, non_blocking=True)

                batch_size = labels.size(0)
                if batch_size == 0: continue

                outputs = model(audio_feat, video_feat, pers_feat, A_mask=None, V_mask=None) # Pass masks if needed
                loss = criterion(outputs, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"    WARNING: NaN or Inf loss encountered in eval batch {batch_idx}. Contribution ignored.")
                    continue

                total_loss += loss.item() * batch_size
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                num_samples += batch_size
                processed_batches += 1
            except Exception as e:
                print(f"    ERROR during eval batch {batch_idx}: {e}")
                # traceback.print_exc()
                continue

    if num_samples == 0: return 0.0, 0.0, {}
    avg_loss = total_loss / num_samples
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0

    try:
        report = classification_report(
            all_labels, all_preds, zero_division=0, output_dict=True,
            labels=list(range(num_classes)), target_names=[f'Class_{i}' for i in range(num_classes)]
        )
    except Exception as e:
         print(f"    Error generating classification report: {e}")
         report = {}

    return avg_loss, accuracy, report


# --- Learning Rate Scheduler Function ---
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Creates cosine LR schedule with linear warmup. """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    scheduler = LambdaLR(optimizer, lr_lambda, last_epoch)
    scheduler.is_epoch_scheduler = False # Mark as step-based
    return scheduler


# --- Main Execution Block ---
if __name__ == '__main__':

    # ==========================================================================
    # ==                         PARAMETER DEFINITIONS                        ==
    # ==========================================================================
    # Using parameters from user's previous successful runs where appropriate
    print(f"--- Script Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print("--- Defining Experiment Parameters (Cross-Modal Transformer) ---")

    # --- Paths ---
    DATA_ROOT_PATH = "./data"
    MODEL_SAVE_PATH = './best_cross_modal_transformer_v1.pth' # New name

    # --- Data & Feature Settings ---
    WINDOW_SPLIT_TIME = 1
    AUDIO_FEATURE_METHOD = "wav2vec"  # Match directory name
    VIDEO_FEATURE_METHOD = "openface" # Match directory name
    FEATURE_MAX_LEN = 26              # Max length of INPUT sequence (before modality CLS token)
    PERS_DIM = 1024

    # --- Classification & Splitting ---
    LABEL_COUNT = 2
    TRACK_OPTION = "Track1"
    VAL_RATIO = 0.1
    VAL_PERCENTAGE = 0.1 # Used only if TRACK_OPTION="Track2"
    SEED = 32

    # --- Model Architecture (Cross-Modal) ---
    EMBED_DIM = 128
    N_HEAD = 4                    # Heads for self/cross-attention in blocks
    NUM_CROSS_MODAL_LAYERS = 2    # Number of cross-modal interaction blocks (Try 1-4)
    DIM_FEEDFORWARD = 512         # FFN dim inside blocks
    HIDDEN_DIM_MLP = 256          # MLP for personalized features
    FUSION_HIDDEN_DIM = 128       # Hidden dim for the *final* MLP fusion head
    DROPOUT_RATE = 0.3            # Dropout rate (Adjust based on previous results)

    # --- Training & Optimization ---
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-5          # Base LR for scheduler
    WEIGHT_DECAY = 1e-4           # Weight decay from previous successful runs
    NUM_EPOCHS_FINAL = 75
    DEVICE_NAME = "mps"           # "cuda", "mps", or "cpu"
    CLIP_GRAD_NORM = 1.0
    WARMUP_PROPORTION = 0.1

    # --- Loss Function & Balancing ---
    FOCAL_LOSS_GAMMA = 3.0
    MANUAL_CLASS_WEIGHTS = [1.0, 2.5] # Try weights that gave best balance previously
    USE_WEIGHTED_SAMPLER = True

    # --- Early Stopping ---
    EARLY_STOPPING_PATIENCE = 20 # Slightly increased patience

    # ==========================================================================
    # ==                       END PARAMETER DEFINITIONS                      ==
    # ==========================================================================

    # --- Derived Paths ---
    DEV_JSON_PATH = os.path.join(DATA_ROOT_PATH, 'Training', 'labels', 'Training_Validation_files.json')
    PERSONALIZED_FEATURE_PATH = os.path.join(DATA_ROOT_PATH, 'Training', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')
    AUDIO_FEATURE_DIR = os.path.join(DATA_ROOT_PATH, 'Training', f"{WINDOW_SPLIT_TIME}s", 'Audio', f"{AUDIO_FEATURE_METHOD}")
    VIDEO_FEATURE_DIR = os.path.join(DATA_ROOT_PATH, 'Training', f"{WINDOW_SPLIT_TIME}s", 'Visual', f"{VIDEO_FEATURE_METHOD}")

    # --- Verify Parameters ---
    print("\n--- Verifying Parameters ---")
    if EMBED_DIM % N_HEAD != 0: raise ValueError(f"EMBED_DIM ({EMBED_DIM}) must be divisible by N_HEAD ({N_HEAD})")
    if MANUAL_CLASS_WEIGHTS is not None and len(MANUAL_CLASS_WEIGHTS) != LABEL_COUNT: raise ValueError(f"MANUAL_CLASS_WEIGHTS length must match LABEL_COUNT")
    print("Parameters verified.")

    # --- Setup Device and Seed ---
    print("\n--- Setting Up Device and Seed ---")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    DEVICE = DEVICE_NAME
    if DEVICE == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if not torch.backends.mps.is_built(): DEVICE = 'cpu'; print("Warning: MPS not built. Using CPU.")
        else: print(f"Using MPS device.")
    elif DEVICE == 'cuda' and torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED); print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else: DEVICE = 'cpu'; print("Using CPU.")
    print(f"Selected device: {DEVICE}")

    # --- Basic Path Checks ---
    print("\n--- Checking Paths ---")
    paths_ok = True
    for p, name in [(DEV_JSON_PATH, "Dev JSON"), (PERSONALIZED_FEATURE_PATH, "Personalized Features"), (AUDIO_FEATURE_DIR, "Audio Dir"), (VIDEO_FEATURE_DIR, "Video Dir")]:
        if not os.path.exists(p): print(f"ERROR: {name} not found: {p}"); paths_ok = False
        else: print(f"Found {name}: {p}")
    if not paths_ok: exit(1)

    # --- Split Data ---
    print("\n--- Splitting Data ---")
    train_data, val_data = [], []
    try:
        if TRACK_OPTION=='Track1': train_data, val_data, _, _ = train_val_split1(DEV_JSON_PATH, val_ratio=VAL_RATIO, random_seed=SEED)
        elif TRACK_OPTION=='Track2': train_data, val_data, _, _ = train_val_split2(DEV_JSON_PATH, val_percentage=VAL_PERCENTAGE, seed=SEED)
        else: print(f"Error: Invalid TRACK_OPTION '{TRACK_OPTION}'."); exit(1)
        print(f"Data split: Train={len(train_data)}, Val={len(val_data)}")
    except Exception as e: print(f"Error splitting data: {e}"); exit(1)
    if not train_data or not val_data: print("Error: Data splitting failed."); exit(1)

    # --- Determine Feature Dimensions ---
    print("\n--- Determining Feature Dimensions ---")
    audio_dim, video_dim = None, None
    try:
        for fname in sorted(os.listdir(AUDIO_FEATURE_DIR)):
            if fname.lower().endswith('.npy'):
                try: audio_dim = np.load(os.path.join(AUDIO_FEATURE_DIR, fname)).shape[-1]; print(f"  Audio Dim: {audio_dim} (from {fname})"); break
                except Exception as load_e: print(f"    Warn: Could not load {fname}: {load_e}")
        if audio_dim is None: print(f"ERR: No loadable .npy found in {AUDIO_FEATURE_DIR}."); exit(1)
    except Exception as e: print(f"ERR determining audio dim: {e}"); exit(1)
    try:
        for fname in sorted(os.listdir(VIDEO_FEATURE_DIR)):
             if fname.lower().endswith('.npy'):
                try: video_dim = np.load(os.path.join(VIDEO_FEATURE_DIR, fname)).shape[-1]; print(f"  Video Dim: {video_dim} (from {fname})"); break
                except Exception as load_e: print(f"    Warn: Could not load {fname}: {load_e}")
        if video_dim is None: print(f"ERR: No loadable .npy found in {VIDEO_FEATURE_DIR}."); exit(1)
    except Exception as e: print(f"ERR determining video dim: {e}"); exit(1)
    NUM_CLASSES = LABEL_COUNT
    print(f"Dims Used: Audio={audio_dim}, Video={video_dim}, Pers={PERS_DIM}, Classes={NUM_CLASSES}")

    # --- Create Datasets ---
    print("\n--- Creating Datasets ---");
    try:
        full_train_dataset = AudioVisualDataset(json_data=train_data, label_count=LABEL_COUNT, personalized_feature_file=PERSONALIZED_FEATURE_PATH, max_len=FEATURE_MAX_LEN, audio_path=AUDIO_FEATURE_DIR, video_path=VIDEO_FEATURE_DIR)
        val_dataset = AudioVisualDataset(json_data=val_data, label_count=LABEL_COUNT, personalized_feature_file=PERSONALIZED_FEATURE_PATH, max_len=FEATURE_MAX_LEN, audio_path=AUDIO_FEATURE_DIR, video_path=VIDEO_FEATURE_DIR)
        print(f"Dataset sizes: Train={len(full_train_dataset)}, Val={len(val_dataset)}")
    except Exception as e: print(f"Error creating datasets: {e}"); traceback.print_exc(); exit(1)
    if len(full_train_dataset) == 0 or len(val_dataset) == 0: print("Error: Datasets are empty."); exit(1)

    # --- Calculate Class Weights and Sampler ---
    focal_loss_weights_tensor = None
    sampler = None
    if NUM_CLASSES > 1:
        print("\n--- Processing Class Distribution ---")
        train_labels = []
        label_extraction_failed = False
        try:
            print("  Extracting labels...")
            for i in range(len(full_train_dataset)):
                try:
                    sample = full_train_dataset[i]
                    if isinstance(sample, dict) and 'emo_label' in sample: train_labels.append(int(sample['emo_label']))
                    else: print(f"Warn: Bad label format index {i}")
                except Exception as sample_e: print(f"Warn: Error access sample {i}: {sample_e}")
            print(f"  Extracted {len(train_labels)} labels.")
            if len(train_labels) != len(full_train_dataset): print(f"  Warn: Label count mismatch.")
            if not train_labels: label_extraction_failed = True
        except Exception as e: print(f"  Error extracting labels: {e}."); label_extraction_failed = True

        if not label_extraction_failed and train_labels:
            class_counts = Counter(train_labels)
            print(f"  Train counts: {dict(sorted(class_counts.items()))}")
            num_train_samples = len(train_labels)
            # --- Loss Weights ---
            if MANUAL_CLASS_WEIGHTS is not None:
                focal_loss_weights_tensor = torch.tensor(MANUAL_CLASS_WEIGHTS, dtype=torch.float).to(DEVICE)
                print(f"  Using Manual Loss Weights: {focal_loss_weights_tensor.cpu().numpy()}")
            # --- Sampler ---
            if USE_WEIGHTED_SAMPLER and class_counts and num_train_samples > 0:
                print("  Creating WeightedRandomSampler...")
                class_sample_weights = {cls: num_train_samples / count if count > 0 else 0 for cls, count in class_counts.items()}
                sample_weights_list = [class_sample_weights.get(label, 0) for label in train_labels]
                if sample_weights_list and sum(sample_weights_list) > 0:
                    min_pos_weight = min((w for w in sample_weights_list if w > 0), default=1e-9)
                    safe_sample_weights = [max(w, min_pos_weight * 1e-6) for w in sample_weights_list]
                    sampler = WeightedRandomSampler(torch.DoubleTensor(safe_sample_weights), len(safe_sample_weights), replacement=True)
                    print(f"  Sampler created.")
                else: print("  Warn: Cannot create sampler weights."); sampler = None
            elif USE_WEIGHTED_SAMPLER: print("  Warn: Sampler enabled but cannot create."); sampler = None
            else: print("  Sampler disabled."); sampler = None
        else: print("  Skipping weights/sampler."); sampler = None

    # --- Model Instantiation ---
    print("\n--- Instantiating Model (Cross-Modal Transformer Version) ---")
    try:
        final_model = CrossModalTransformer(
            audio_dim=audio_dim, video_dim=video_dim, pers_dim=PERS_DIM,
            embed_dim=EMBED_DIM, nhead=N_HEAD,
            num_cross_modal_layers=NUM_CROSS_MODAL_LAYERS, # Use cross-modal layers param
            dim_feedforward=DIM_FEEDFORWARD, hidden_dim_mlp=HIDDEN_DIM_MLP,
            num_classes=NUM_CLASSES, max_len=FEATURE_MAX_LEN,
            fusion_hidden_dim=FUSION_HIDDEN_DIM, # For final fusion MLP
            dropout_rate=DROPOUT_RATE,
        ).to(DEVICE)
        print("Model instantiated successfully.")
    except Exception as e: print(f"ERROR: Model instantiation error: {e}"); traceback.print_exc(); exit(1)

    # --- Model Summary ---
    if torchinfo:
        print("\n--- Model Architecture & Parameters ---")
        example_audio_shape = (BATCH_SIZE, FEATURE_MAX_LEN, audio_dim)
        example_video_shape = (BATCH_SIZE, FEATURE_MAX_LEN, video_dim)
        example_pers_shape = (BATCH_SIZE, PERS_DIM)
        try:
            # Note: Input size for summary doesn't include CLS token explicitly
            torchinfo.summary(final_model,
                              input_size=[example_audio_shape, example_video_shape, example_pers_shape],
                              col_names=["input_size", "output_size", "num_params", "mult_adds"],
                              depth=6, # Slightly deeper summary
                              device=DEVICE, verbose=0)
        except Exception as e_summary:
            print(f"Could not generate torchinfo summary: {e_summary}")
            total_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
            print(f"Total Trainable Parameters (Manual Count): {total_params:,}")

    # --- Create DataLoaders ---
    print(f"\n--- Creating DataLoaders (Batch Size: {BATCH_SIZE}, Sampler: {sampler is not None}) ---")
    num_workers = 0 # Adjust based on system/OS
    try:
        final_train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=num_workers, shuffle=False if sampler else True, pin_memory=True if DEVICE == 'cuda' else False, drop_last=False)
        # Use larger batch size for validation if possible
        val_batch_size = BATCH_SIZE * 2
        final_val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if DEVICE == 'cuda' else False)
        if len(final_train_loader) == 0 or len(final_val_loader) == 0: print("Error: DataLoaders are empty."); exit(1)
        print(f"Train DataLoader length: {len(final_train_loader)}, Val DataLoader length: {len(final_val_loader)}")
    except Exception as e: print(f"Error creating DataLoaders: {e}"); traceback.print_exc(); exit(1)

    # --- Optimizer ---
    print(f"\n--- Setting up Optimizer (AdamW) ---")
    print(f"Using AdamW optimizer with Base LR={LEARNING_RATE}, Weight Decay={WEIGHT_DECAY}")
    final_optimizer = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Criterion (Loss Function) ---
    print(f"\n--- Setting up Loss Function (FocalLoss) ---")
    print(f"Using FocalLoss with Gamma={FOCAL_LOSS_GAMMA}, Weights={focal_loss_weights_tensor}")
    final_criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA, weight=focal_loss_weights_tensor).to(DEVICE)

    # --- Learning Rate Scheduler ---
    scheduler = None
    lr_scheduler_active = False
    if len(final_train_loader) > 0:
        try:
            steps_per_epoch = len(final_train_loader)
            TOTAL_TRAINING_STEPS = steps_per_epoch * NUM_EPOCHS_FINAL
            NUM_WARMUP_STEPS = int(TOTAL_TRAINING_STEPS * WARMUP_PROPORTION)
            print(f"\n--- Setting up LR Scheduler ---")
            print(f"Using Cosine schedule with Warmup.")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"Total steps: {TOTAL_TRAINING_STEPS}, Warmup steps: {NUM_WARMUP_STEPS}")
            if NUM_WARMUP_STEPS >= TOTAL_TRAINING_STEPS:
                 print("Warning: Warmup steps >= Total steps. Adjust WARMUP_PROPORTION or NUM_EPOCHS_FINAL.")
                 NUM_WARMUP_STEPS = max(1, int(TOTAL_TRAINING_STEPS * 0.05)) # Fallback to shorter warmup
                 print(f"Using adjusted Warmup steps: {NUM_WARMUP_STEPS}")

            scheduler = get_cosine_schedule_with_warmup(
                final_optimizer,
                num_warmup_steps=NUM_WARMUP_STEPS,
                num_training_steps=TOTAL_TRAINING_STEPS
            )
            print("LR Scheduler created.")
            lr_scheduler_active = True
        except Exception as e: print(f"Error setting up LR scheduler: {e}. Proceeding without scheduler.")
    else: print("Training loader has length 0. Cannot setup LR scheduler.")

    # --- Training Loop Setup ---
    best_final_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    training_log = []

    # --- Final Training Loop ---
    print(f"\n--- Starting Final Training Loop ({NUM_EPOCHS_FINAL} epochs) ---")
    start_train_time = time.time()

    for epoch in range(NUM_EPOCHS_FINAL):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS_FINAL}")
        current_lr = final_optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.3e}")

        # --- Train ---
        print("  Starting training...")
        train_loss, train_acc = train_epoch(final_model, final_train_loader, final_optimizer, final_criterion, DEVICE, clip_grad_norm=CLIP_GRAD_NORM, scheduler=scheduler if lr_scheduler_active else None)

        # --- Validate ---
        print("  Starting validation...")
        val_loss, val_acc, val_report = evaluate(final_model, final_val_loader, final_criterion, DEVICE, NUM_CLASSES)

        current_macro_f1 = 0.0
        f1_extracted = False
        if val_report and 'macro avg' in val_report and isinstance(val_report['macro avg'], dict) and 'f1-score' in val_report['macro avg']:
             f1_val = val_report['macro avg']['f1-score']
             if f1_val is not None and not np.isnan(f1_val):
                 current_macro_f1 = float(f1_val)
                 f1_extracted = True
             else: print("  Warning: Validation Macro F1 score is NaN or None.")
        else: print("  Warning: Macro F1 score not found/valid in validation report.")

        # --- Print epoch results ---
        print(f"  Results: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, Macro F1={current_macro_f1:.4f}")

        # Log results
        epoch_log = {'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'val_macro_f1': current_macro_f1, 'lr': current_lr }
        training_log.append(epoch_log)

        # --- Check for Best Model & Saving ---
        if f1_extracted and current_macro_f1 > best_final_macro_f1:
            delta = current_macro_f1 - best_final_macro_f1 if best_final_macro_f1 > -1.0 else current_macro_f1
            print(f"  -> Macro F1 improved by {delta:.4f} (from {max(0, best_final_macro_f1):.4f} to {current_macro_f1:.4f}). Saving model...")
            best_final_macro_f1 = current_macro_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            try:
                torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
                print(f"  -> Saved best model to {MODEL_SAVE_PATH}")
                best_val_report_path = MODEL_SAVE_PATH.replace('.pth', '_best_report.json')
                with open(best_val_report_path, 'w') as f: json.dump(val_report, f, indent=2)
                print(f"  -> Saved best validation report to {best_val_report_path}")
            except Exception as e_save: print(f"  -> Error saving model/report: {e_save}")
        else:
            epochs_without_improvement += 1
            print(f"  -> Macro F1 did not improve for {epochs_without_improvement} epoch(s). Best F1: {max(0, best_final_macro_f1):.4f} at epoch {best_epoch if best_epoch != -1 else 'N/A'}.")

        # --- Early Stopping Check ---
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without validation Macro F1 improvement.")
            break

        epoch_duration = time.time() - epoch_start_time
        print(f"  Epoch completed in {epoch_duration:.2f}s")

    # --- After the loop ---
    total_training_time = time.time() - start_train_time
    print("\n--- Final Training Complete ---")
    print(f"Total training time: {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")
    if best_epoch != -1: print(f"Best Validation Macro F1 score ({best_final_macro_f1:.4f}) achieved at epoch {best_epoch}")
    else: print("No model saved. Macro F1 may not have improved.")

    # Save training log
    log_path = MODEL_SAVE_PATH.replace('.pth', '_training_log.json')
    try:
        with open(log_path, 'w') as f: json.dump(training_log, f, indent=2)
        print(f"Training log saved to {log_path}")
    except Exception as e: print(f"Error saving training log: {e}")

    # --- Evaluation ---
    print("\n--- Evaluating Best Saved Model on Validation Set ---")
    model_eval_path = MODEL_SAVE_PATH
    if best_epoch != -1 and os.path.exists(model_eval_path):
        try:
            print(f"Loading best model state from epoch {best_epoch} saved at {model_eval_path}")
            # Instantiate the CrossModalTransformer model
            eval_model = CrossModalTransformer(
                 audio_dim=audio_dim, video_dim=video_dim, pers_dim=PERS_DIM,
                 embed_dim=EMBED_DIM, nhead=N_HEAD,
                 num_cross_modal_layers=NUM_CROSS_MODAL_LAYERS, # Use correct param
                 dim_feedforward=DIM_FEEDFORWARD, hidden_dim_mlp=HIDDEN_DIM_MLP,
                 num_classes=NUM_CLASSES, max_len=FEATURE_MAX_LEN,
                 fusion_hidden_dim=FUSION_HIDDEN_DIM, # For final fusion MLP
                 dropout_rate=DROPOUT_RATE,
            ).to(DEVICE)
            eval_model.load_state_dict(torch.load(model_eval_path, map_location=DEVICE))
            eval_model.eval()
            print("Model loaded successfully.")

            # Re-create criterion for evaluation loss calculation
            eval_criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA, weight=focal_loss_weights_tensor).to(DEVICE)

            # Perform evaluation
            print("Running final evaluation...")
            final_loss, final_acc, final_report = evaluate(eval_model, final_val_loader, eval_criterion, DEVICE, NUM_CLASSES)

            print(f"\nFinal Evaluation Results (using model from epoch {best_epoch}):")
            print(f"  Loss: {final_loss:.4f}")
            print(f"  Accuracy: {final_acc:.4f}")
            if final_report and 'macro avg' in final_report:
                print(f"  Macro F1: {final_report['macro avg'].get('f1-score', 0.0):.4f}")
                print("  Classification Report:")
                print(json.dumps(final_report, indent=2))
            else:
                print("  Classification report could not be generated or was empty.")

        except Exception as e:
            print(f"Error evaluating best model from {model_eval_path}: {e}")
            traceback.print_exc()
    else:
        if best_epoch == -1: print("No best epoch recorded. Skipping final evaluation.")
        else: print(f"Best model path ({model_eval_path}) not found.")

    print(f"\n--- Script End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")