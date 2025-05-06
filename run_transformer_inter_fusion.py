"""
Late Fusion Transformer Model (CLS Token Aggregation)
- Uses WeightedRandomSampler for handling class imbalance.
- Uses Warmup + Cosine LR Scheduling.
- Saves the best model based on Validation Macro F1-score.
- Implements Early Stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR # For custom scheduler
from collections import Counter
import numpy as np
import os
import json
import time
import traceback
import math

from sklearn.metrics import accuracy_score, classification_report, f1_score
from DataSets.audioVisualDataset import AudioVisualDataset
from Utils.focal_loss import FocalLoss
from Utils.test_val_split import train_val_split1, train_val_split2
import torchinfo


# CLS token aggregation and positional encoding
class PositionalEncoding(nn.Module):
    """Injects positional information into the sequence embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if d_model % 2 != 0:
             print("Warning: d_model in PositionalEncoding is odd. This might lead to slight inaccuracies.")

        position = torch.arange(max_len).unsqueeze(1)
        # Corrected div_term calculation
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model dimension carefully when assigning cosine
        if d_model % 2 != 0:
             pe[:, 0, 1::2] = torch.cos(position * div_term)[:,:pe[:, 0, 1::2].size(1)] # Match dimensions
        else:
             pe[:, 0, 1::2] = torch.cos(position * div_term)

        pe = pe.permute(1, 0, 2) # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, embedding_dim]
        # Add positional encoding, ensuring size match
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LateFusionTransformerWithCLS(nn.Module):
    """Late Fusion Transformer using CLS token aggregation."""
    def __init__(self, audio_dim, video_dim, pers_dim, embed_dim, nhead, num_encoder_layers,
                 dim_feedforward, hidden_dim_mlp, num_classes, max_len,
                 fusion_hidden_dim, dropout_rate=0.1):
        super().__init__()

        self.cls_token_audio = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_video = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token_audio, std=0.02)
        nn.init.normal_(self.cls_token_video, std=0.02)

        # Positional encoding needs size max_len + 1 for CLS token
        self.pos_encoder = PositionalEncoding(embed_dim, dropout_rate, max_len + 1)

        self.audio_projection = nn.Linear(audio_dim, embed_dim)
        self.video_projection = nn.Linear(video_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout_rate, activation=torch.nn.functional.gelu, # Use GELU
            batch_first=True, norm_first=False
        )
        self.audio_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.video_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.mlp_pers_layers = nn.Sequential(
            nn.Linear(pers_dim, hidden_dim_mlp),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Add LayerNorm before fusion MLP input
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim + embed_dim + hidden_dim_mlp),
            nn.Linear(embed_dim + embed_dim + hidden_dim_mlp, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, num_classes)
        )

    def forward(self, A_feat, V_feat, P_feat):
        batch_size = A_feat.size(0)

        A_proj = self.audio_projection(A_feat)
        cls_audio_expanded = self.cls_token_audio.expand(batch_size, -1, -1)
        A_with_cls = torch.cat((cls_audio_expanded, A_proj), dim=1)
        A_enc = self.pos_encoder(A_with_cls)
        A_out = self.audio_transformer_encoder(A_enc)
        A_agg = A_out[:, 0] # Use CLS token output

        V_proj = self.video_projection(V_feat)
        cls_video_expanded = self.cls_token_video.expand(batch_size, -1, -1)
        V_with_cls = torch.cat((cls_video_expanded, V_proj), dim=1)
        V_enc = self.pos_encoder(V_with_cls)
        V_out = self.video_transformer_encoder(V_enc)
        V_agg = V_out[:, 0] # Use CLS token output

        if P_feat.ndim == 1: P_feat = P_feat.unsqueeze(0)
        elif P_feat.ndim != 2: raise ValueError(f"Unexpected P_feat shape: {P_feat.shape}")
        P_proc = self.mlp_pers_layers(P_feat)

        combined_features = torch.cat((A_agg, V_agg, P_proc), dim=1)
        logits = self.fusion_mlp(combined_features)
        return logits

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
            audio_feat = batch['A_feat'].to(device, non_blocking=True) # Use non_blocking for potential speedup with pin_memory
            video_feat = batch['V_feat'].to(device, non_blocking=True)
            pers_feat = batch['personalized_feat'].to(device, non_blocking=True)
            labels = batch['emo_label'].to(device, non_blocking=True)

            batch_size = labels.size(0)
            if batch_size == 0: continue

            optimizer.zero_grad(set_to_none=True)
            outputs = model(audio_feat, video_feat, pers_feat)
            loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    WARNING: NaN or Inf loss encountered in train batch {batch_idx}. Skipping backward/step.")
                continue

            loss.backward()

            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()

            # *** FIX: Step LR scheduler (if using step-based scheduler like cosine) ***
            if scheduler is not None:
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
            continue

    if num_samples == 0:
        print("  Warning: No samples processed in training epoch.")
        return 0.0, 0.0
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

                outputs = model(audio_feat, video_feat, pers_feat)
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
                continue

    avg_loss = total_loss / num_samples
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0

    try:
        report = classification_report(
            all_labels,
            all_preds,
            zero_division=0,
            output_dict=True,
            labels=list(range(num_classes)),
            target_names=[f'Class_{i}' for i in range(num_classes)]
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
    # Add flag to distinguish from ReduceLROnPlateau if needed elsewhere (though not strictly necessary now)
    scheduler.is_epoch_scheduler = False
    return scheduler


# --- Main Execution Block ---
if __name__ == '__main__':

    # ==========================================================================
    # ==                         PARAMETER DEFINITIONS                        ==
    # ==========================================================================
    print(f"--- Script Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    print("--- Defining Experiment Parameters ---")

    # --- Paths ---
    DATA_ROOT_PATH = "./data"
    MODEL_SAVE_PATH = './best_late_fusion_transformer_cls_v4.pth'

    # --- Data & Feature Settings ---
    WINDOW_SPLIT_TIME = 1
    AUDIO_FEATURE_METHOD = "wav2vec"
    VIDEO_FEATURE_METHOD = "openface"
    FEATURE_MAX_LEN = 26
    PERS_DIM = 1024

    # --- Classification & Splitting ---
    LABEL_COUNT = 2
    TRACK_OPTION = "Track1"
    VAL_RATIO = 0.1
    VAL_PERCENTAGE = 0.1 # Used only if TRACK_OPTION="Track2"
    SEED = 32

    # --- Model Architecture (Using CLS Token) ---
    EMBED_DIM = 128
    N_HEAD = 4
    NUM_ENCODER_LAYERS = 2
    DIM_FEEDFORWARD = 512
    HIDDEN_DIM_MLP = 256
    FUSION_HIDDEN_DIM = 128
    DROPOUT_RATE = 0.4

    # --- Training & Optimization ---
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS_FINAL = 75
    DEVICE_NAME = "mps"
    CLIP_GRAD_NORM = 1.0
    WARMUP_PROPORTION = 0.1

    # --- Loss Function & Balancing ---
    FOCAL_LOSS_GAMMA = 3.0
    MANUAL_CLASS_WEIGHTS = [1.0, 3.0] # Weight Class 0=1.0, Class 1=3.0
    USE_WEIGHTED_SAMPLER = True

    # --- Early Stopping ---
    EARLY_STOPPING_PATIENCE = 15

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
    if MANUAL_CLASS_WEIGHTS is not None and len(MANUAL_CLASS_WEIGHTS) != LABEL_COUNT: raise ValueError(f"MANUAL_CLASS_WEIGHTS length ({len(MANUAL_CLASS_WEIGHTS)}) must match LABEL_COUNT ({LABEL_COUNT})")
    print("Parameters verified.")

    # --- Setup Device and Seed ---
    print("\n--- Setting Up Device and Seed ---")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    DEVICE = DEVICE_NAME
    if DEVICE == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("Warning: MPS available but not built. Falling back to CPU.")
            DEVICE = 'cpu'
        else:
            print(f"Using MPS device.")
            # torch.mps.manual_seed(SEED) # Seed MPS if needed
    elif DEVICE == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        if DEVICE != 'cpu': print(f"Warning: Device '{DEVICE}' specified but unavailable. Using CPU.")
        DEVICE = 'cpu'
    print(f"Selected device: {DEVICE}")

    # --- Basic Path Checks ---
    print("\n--- Checking Paths ---")
    paths_ok = True
    for p, name in [(DEV_JSON_PATH, "Dev JSON"), (PERSONALIZED_FEATURE_PATH, "Personalized Features"), (AUDIO_FEATURE_DIR, "Audio Features Dir"), (VIDEO_FEATURE_DIR, "Video Features Dir")]:
        if not os.path.exists(p):
            print(f"ERROR: {name} not found at: {p}")
            paths_ok = False
        else:
             print(f"Found {name}: {p}")
    if not paths_ok: exit(1)

    # --- Split Data ---
    print("\n--- Splitting Data ---")
    train_data, val_data = [], []
    try:
        if TRACK_OPTION=='Track1':
            train_data, val_data, _, _ = train_val_split1(DEV_JSON_PATH, val_ratio=VAL_RATIO, random_seed=SEED)
        elif TRACK_OPTION=='Track2':
            train_data, val_data, _, _ = train_val_split2(DEV_JSON_PATH, val_percentage=VAL_PERCENTAGE, seed=SEED)
        else:
            print(f"Error: Invalid TRACK_OPTION '{TRACK_OPTION}'."); exit(1)
        print(f"Data split: Train={len(train_data)}, Val={len(val_data)}")
    except Exception as e: print(f"Error splitting data: {e}"); exit(1)
    if not train_data or not val_data: print("Error: Data splitting failed."); exit(1)

    # --- Determine Feature Dimensions ---
    print("\n--- Determining Feature Dimensions ---")
    audio_dim, video_dim = None, None
    try:
        # Find first .npy file to get dimension
        for fname in sorted(os.listdir(AUDIO_FEATURE_DIR)): # Sort for consistency
            if fname.lower().endswith('.npy'):
                try:
                    audio_dim = np.load(os.path.join(AUDIO_FEATURE_DIR, fname), allow_pickle=False).shape[-1]
                    print(f"  Determined Audio Dim: {audio_dim} (from {fname})")
                    break
                except Exception as load_e:
                    print(f"    Could not load {fname}: {load_e}")
        if audio_dim is None: print(f"ERR: No loadable .npy files found in {AUDIO_FEATURE_DIR}."); exit(1)
    except Exception as e: print(f"ERR determining audio dim: {e}"); exit(1)
    try:
        for fname in sorted(os.listdir(VIDEO_FEATURE_DIR)):
             if fname.lower().endswith('.npy'):
                try:
                    video_dim = np.load(os.path.join(VIDEO_FEATURE_DIR, fname), allow_pickle=False).shape[-1]
                    print(f"  Determined Video Dim: {video_dim} (from {fname})")
                    break
                except Exception as load_e:
                    print(f"    Could not load {fname}: {load_e}")
        if video_dim is None: print(f"ERR: No loadable .npy files found in {VIDEO_FEATURE_DIR}."); exit(1)
    except Exception as e: print(f"ERR determining video dim: {e}"); exit(1)

    NUM_CLASSES = LABEL_COUNT
    print(f"Final Dims Used: Audio={audio_dim}, Video={video_dim}, Pers={PERS_DIM}, Classes={NUM_CLASSES}")

    # --- Create Datasets ---
    print("\n--- Creating Datasets ---");
    try:
        full_train_dataset = AudioVisualDataset(json_data=train_data, label_count=LABEL_COUNT, personalized_feature_file=PERSONALIZED_FEATURE_PATH, max_len=FEATURE_MAX_LEN, audio_path=AUDIO_FEATURE_DIR, video_path=VIDEO_FEATURE_DIR)
        val_dataset = AudioVisualDataset(json_data=val_data, label_count=LABEL_COUNT, personalized_feature_file=PERSONALIZED_FEATURE_PATH, max_len=FEATURE_MAX_LEN, audio_path=AUDIO_FEATURE_DIR, video_path=VIDEO_FEATURE_DIR)
        print(f"Dataset sizes: Train={len(full_train_dataset)}, Val={len(val_dataset)}")
    except Exception as e: print(f"Error creating datasets: {e}"); traceback.print_exc(); exit(1)
    if len(full_train_dataset) == 0 or len(val_dataset) == 0: print("Error: Datasets are empty after creation."); exit(1)

    # --- Calculate Class Weights and Sampler ---
    focal_loss_weights_tensor = None
    sampler = None
    if NUM_CLASSES > 1:
        print("\n--- Processing Class Distribution for Loss Weights and Sampler ---")
        train_labels = []
        label_extraction_failed = False
        try:
            print("  Extracting labels from training dataset...")
            # Assuming __getitem__ returns a dict with 'emo_label' as integer
            for i in range(len(full_train_dataset)):
                try:
                    sample = full_train_dataset[i]
                    if isinstance(sample, dict) and 'emo_label' in sample:
                        train_labels.append(int(sample['emo_label']))
                    else:
                         # Handle cases where sample might not be dict or key missing
                         # Attempting to access sample.label etc. might be needed depending on Dataset impl.
                         print(f"Warning: Could not extract label from training sample index {i}. Format: {type(sample)}")
                except Exception as sample_e:
                    print(f"Warning: Error accessing sample index {i}: {sample_e}")
            print(f"  Attempted extraction for {len(full_train_dataset)} samples, got {len(train_labels)} labels.")
            if len(train_labels) != len(full_train_dataset):
                print(f"  Warning: Number of extracted labels ({len(train_labels)}) does not match dataset size ({len(full_train_dataset)}). Check dataset implementation.")
                if not train_labels: # If completely failed
                     label_extraction_failed = True

        except Exception as e:
            print(f"  Could not extract training labels: {e}.")
            label_extraction_failed = True

        if label_extraction_failed:
            print("  Label extraction failed. Proceeding without weights/sampler.")
            train_labels = [] # Ensure it's empty
            class_counts = None
            num_train_samples = 0
        elif train_labels:
            class_counts = Counter(train_labels)
            print(f"  Training class counts: {dict(sorted(class_counts.items()))}")
            num_train_samples = len(train_labels)
        else:
            print("  No labels extracted, proceeding without weights/sampler.")
            class_counts = None
            num_train_samples = 0


        # --- Determine Focal Loss Weights ---
        if MANUAL_CLASS_WEIGHTS is not None and not label_extraction_failed:
             if len(MANUAL_CLASS_WEIGHTS) == NUM_CLASSES:
                  focal_loss_weights_tensor = torch.tensor(MANUAL_CLASS_WEIGHTS, dtype=torch.float).to(DEVICE)
                  print(f"  Using MANUALLY DEFINED class weights for FocalLoss: {focal_loss_weights_tensor.cpu().numpy()}")
             else:
                  print(f"  Warning: Manual class_weights length != num_classes. Loss weights set to None.")
        else:
             print("  Manual class weights not defined or label extraction failed. Loss weights set to None.")

        # --- Create Sampler ---
        if USE_WEIGHTED_SAMPLER and class_counts and num_train_samples > 0:
            print("  Attempting to create WeightedRandomSampler...")
            class_sample_weights = {cls: num_train_samples / count if count > 0 else 0 for cls, count in class_counts.items()}
            sample_weights_list = [class_sample_weights.get(label, 0) for label in train_labels]

            if sample_weights_list and sum(sample_weights_list) > 0:
                min_pos_weight = min((w for w in sample_weights_list if w > 0), default=1e-9) # Avoid zero division
                safe_sample_weights = [max(w, min_pos_weight * 1e-6) for w in sample_weights_list] # Ensure positive weights

                sampler = WeightedRandomSampler(torch.DoubleTensor(safe_sample_weights), len(safe_sample_weights), replacement=True)
                print(f"  Created WeightedRandomSampler for training. Num samples: {len(safe_sample_weights)}")
            else:
                print("  Warning: Could not create valid sample weights. Sampler disabled.")
                sampler = None
        elif USE_WEIGHTED_SAMPLER:
            print("  WeightedRandomSampler enabled but cannot be created (no class counts or labels).")
            sampler = None
        else:
            print("  WeightedRandomSampler is disabled by configuration.")
            sampler = None

    # --- Model Instantiation ---
    print("\n--- Instantiating Model (CLS Token Version) ---")
    try:
        final_model = LateFusionTransformerWithCLS(
            audio_dim=audio_dim, video_dim=video_dim, pers_dim=PERS_DIM,
            embed_dim=EMBED_DIM, nhead=N_HEAD, num_encoder_layers=NUM_ENCODER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD, hidden_dim_mlp=HIDDEN_DIM_MLP,
            num_classes=NUM_CLASSES, max_len=FEATURE_MAX_LEN,
            dropout_rate=DROPOUT_RATE, fusion_hidden_dim=FUSION_HIDDEN_DIM,
        ).to(DEVICE)
        print("Model instantiated successfully.")
    except Exception as e: print(f"Final model instantiation error: {e}"); traceback.print_exc(); exit(1)

    # --- Model Summary ---
    if torchinfo:
        print("\n--- Model Architecture & Parameters ---")
        example_audio_shape = (BATCH_SIZE, FEATURE_MAX_LEN, audio_dim)
        example_video_shape = (BATCH_SIZE, FEATURE_MAX_LEN, video_dim)
        example_pers_shape = (BATCH_SIZE, PERS_DIM)
        try:
            torchinfo.summary(final_model,
                              input_size=[example_audio_shape, example_video_shape, example_pers_shape],
                              col_names=["input_size", "output_size", "num_params", "mult_adds"],
                              depth=5, device=DEVICE, verbose=0)
        except Exception as e_summary:
            print(f"Could not generate torchinfo summary: {e_summary}")
            total_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
            print(f"Total Trainable Parameters (Manual Count): {total_params:,}")

    # --- Create DataLoaders ---
    print(f"\n--- Creating DataLoaders (Batch Size: {BATCH_SIZE}, Sampler: {sampler is not None}) ---")
    num_workers = 0 # default 0 cuz of mps
    final_train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=num_workers, shuffle=False if sampler else True, pin_memory=True if DEVICE == 'cuda' else False, drop_last=False)
    final_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=num_workers, pin_memory=True if DEVICE == 'cuda' else False)
    if len(final_train_loader) == 0 or len(final_val_loader) == 0: print("Error: DataLoaders are empty."); exit(1)
    print(f"Train DataLoader length: {len(final_train_loader)}, Val DataLoader length: {len(final_val_loader)}")

    print(f"\n--- Setting up Optimizer (AdamW) ---")
    print(f"Using AdamW optimizer with Base LR={LEARNING_RATE}, Weight Decay={WEIGHT_DECAY}")
    final_optimizer = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"\n--- Setting up Loss Function (FocalLoss) ---")
    print(f"Using FocalLoss with Gamma={FOCAL_LOSS_GAMMA}, Weights={focal_loss_weights_tensor}")
    final_criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA, weight=focal_loss_weights_tensor).to(DEVICE)

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

            scheduler = get_cosine_schedule_with_warmup(
                final_optimizer,
                num_warmup_steps=NUM_WARMUP_STEPS,
                num_training_steps=TOTAL_TRAINING_STEPS
            )
            print("LR Scheduler created.")
            lr_scheduler_active = True
        except Exception as e:
            print(f"Error setting up LR scheduler: {e}. Proceeding without scheduler.")
    else:
        print("Training loader has length 0. Cannot setup LR scheduler.")


    # --- Training Loop Setup ---
    best_final_macro_f1 = -1.0 # Initialize to -1 to ensure first valid F1 saves
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
             else:
                 print("  Warning: Validation Macro F1 score is NaN or None.")
        else:
             print("  Warning: Macro F1 score not found in validation report structure.")

        print(f"  Results: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, Macro F1={current_macro_f1:.4f}")

        epoch_log = {'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'val_macro_f1': current_macro_f1, 'lr': current_lr }
        training_log.append(epoch_log)

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

    total_training_time = time.time() - start_train_time
    print("\n--- Final Training Complete ---")
    print(f"Total training time: {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")
    if best_epoch != -1: print(f"Best Validation Macro F1 score ({best_final_macro_f1:.4f}) achieved at epoch {best_epoch}")
    else: print("No model saved. Macro F1 may not have improved.")

    log_path = MODEL_SAVE_PATH.replace('.pth', '_training_log.json')
    try:
        with open(log_path, 'w') as f: json.dump(training_log, f, indent=2)
        print(f"Training log saved to {log_path}")
    except Exception as e: print(f"Error saving training log: {e}")

    print("\n--- Evaluating Best Saved Model on Validation Set ---")
    model_eval_path = MODEL_SAVE_PATH
    if best_epoch != -1 and os.path.exists(model_eval_path):
        try:
            print(f"Loading best model state from epoch {best_epoch} saved at {model_eval_path}")
            eval_model = LateFusionTransformerWithCLS( # Use the correct class
                audio_dim=audio_dim, video_dim=video_dim, pers_dim=PERS_DIM,
                embed_dim=EMBED_DIM, nhead=N_HEAD, num_encoder_layers=NUM_ENCODER_LAYERS,
                dim_feedforward=DIM_FEEDFORWARD, hidden_dim_mlp=HIDDEN_DIM_MLP,
                num_classes=NUM_CLASSES, max_len=FEATURE_MAX_LEN,
                dropout_rate=DROPOUT_RATE, fusion_hidden_dim=FUSION_HIDDEN_DIM,
            ).to(DEVICE)
            eval_model.load_state_dict(torch.load(model_eval_path, map_location=DEVICE))
            eval_model.eval()
            print("Model loaded successfully.")

            eval_criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA, weight=focal_loss_weights_tensor).to(DEVICE)

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

    print(f"\n--- Script End Time: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")