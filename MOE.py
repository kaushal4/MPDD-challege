# -*- coding: utf-8 -*-
"""
COMPLETE Training Script for Adapter-Based Multimodal Fusion Model
using a Mixture of Experts (MoE) layer for fusion.

- Assumes input features are from FROZEN backbones.
- Trains small ADAPTER modules for each modality.
- Fuses adapted features using a DENSE MoE layer and a final classifier.
- All parameters are defined locally within this script.
- Uses WeightedRandomSampler for handling class imbalance.
- Uses Warmup + Cosine LR Scheduling.
- Saves the best model based on Validation Macro F1-score.
- Implements Early Stopping.
- Personalized feature dimension fixed at 1024.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F # Needed for softmax in MoE
from collections import Counter
import numpy as np
import os
import json
import time
import traceback
import math
from datetime import datetime

# External Imports (Ensure these exist in your project structure)
from sklearn.metrics import accuracy_score, classification_report, f1_score
try:
    # Assuming these paths are correct relative to your script's location
    from DataSets.audioVisualDataset import AudioVisualDataset
    # Models defined below
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


# --- Model Definition ---

class Adapter(nn.Module):
    """A simple MLP adapter module."""
    def __init__(self, input_dim, adapter_hidden_dim, output_dim, dropout_rate):
        super().__init__()
        # Simple 2-layer MLP adapter
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, adapter_hidden_dim),
            nn.GELU(), # Use GELU activation
            nn.Dropout(dropout_rate),
            nn.Linear(adapter_hidden_dim, output_dim)
            # Optional: Add LayerNorm? nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        # Input x is expected to be [B, input_dim]
        return self.adapter(x)

class MoELayer(nn.Module):
    """
    A Dense Mixture of Experts (MoE) Layer.

    Note: This implementation uses a dense weighted sum of all experts,
    not sparse routing (like top-k) typically used for computational savings
    in very large models. It demonstrates the architectural concept.
    """
    def __init__(self, input_dim, expert_hidden_dim, expert_output_dim, num_experts, dropout_rate):
        super().__init__()
        self.num_experts = num_experts
        self.expert_output_dim = expert_output_dim

        # Expert Networks (list of MLPs)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(expert_hidden_dim, expert_output_dim)
            ) for _ in range(num_experts)
        ])

        # Gating Network (simple linear layer)
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # x shape: [B, input_dim]

        # Gating logits -> probabilities
        gate_logits = self.gate(x) # [B, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1) # [B, num_experts]

        # Get outputs from all experts
        # expert_outputs shape initially: [num_experts, B, expert_output_dim]
        expert_outputs = torch.stack([expert(x) for expert in self.experts])

        # Weighted sum (Dense MoE combination)
        # Align dimensions for broadcasting/multiplication
        # gate_probs: [B, num_experts] -> [B, num_experts, 1]
        # expert_outputs: [num_experts, B, expert_output_dim] -> [B, num_experts, expert_output_dim] (permute)
        gate_probs_expanded = gate_probs.unsqueeze(-1) # [B, num_experts, 1]
        expert_outputs_permuted = expert_outputs.permute(1, 0, 2) # [B, num_experts, expert_output_dim]

        # Element-wise multiplication and sum across experts
        weighted_expert_outputs = expert_outputs_permuted * gate_probs_expanded # [B, num_experts, expert_output_dim]
        final_output = torch.sum(weighted_expert_outputs, dim=1) # [B, expert_output_dim]

        # Optional: Add auxiliary loss here for load balancing in sparse MoE (not implemented here)

        return final_output # Shape [B, expert_output_dim]

class MoEAdapterFusionModel(nn.Module):
    """
    Model using frozen backbones (implicit via input features),
    trainable adapters for each modality, fused by an MoE layer.
    """
    def __init__(self, audio_backbone_dim, video_backbone_dim, pers_backbone_dim,
                 adapter_hidden_dim, adapter_output_dim, # Adapter specific dims
                 num_experts, expert_hidden_dim, # MoE specific dims
                 fusion_output_dim, # Output dim of MoE layer / input dim of final classifier
                 num_classes, dropout_rate):
        super().__init__()

        self.adapter_output_dim = adapter_output_dim

        # --- Trainable Adapters ---
        # Takes backbone output dim -> adapter output dim
        self.audio_adapter = Adapter(audio_backbone_dim, adapter_hidden_dim, adapter_output_dim, dropout_rate)
        self.video_adapter = Adapter(video_backbone_dim, adapter_hidden_dim, adapter_output_dim, dropout_rate)
        self.pers_adapter = Adapter(pers_backbone_dim, adapter_hidden_dim, adapter_output_dim, dropout_rate)

        # --- Fusion Head (LayerNorm + MoE Layer) ---
        fusion_input_dim = adapter_output_dim * 3
        self.fusion_norm = nn.LayerNorm(fusion_input_dim) # Normalize concatenated features before MoE

        self.moe_layer = MoELayer(
            input_dim=fusion_input_dim,
            expert_hidden_dim=expert_hidden_dim,
            expert_output_dim=fusion_output_dim, # MoE layer outputs features
            num_experts=num_experts,
            dropout_rate=dropout_rate
        )

        # --- Final Classifier Layer ---
        # Takes the output of the MoE layer
        self.classifier = nn.Linear(fusion_output_dim, num_classes)

    def forward(self, A_feat, V_feat, P_feat):
        # A_feat, V_feat, P_feat are assumed outputs from frozen backbones

        # --- Apply Adapters ---
        # Aggregate sequence features (if needed) before adapter
        # Assumes mean pooling if input is sequential [B, Seq, Dim]
        if A_feat.ndim > 2: A_feat_agg = A_feat.mean(dim=1)
        else: A_feat_agg = A_feat # Already aggregated [B, Dim]

        if V_feat.ndim > 2: V_feat_agg = V_feat.mean(dim=1)
        else: V_feat_agg = V_feat # Already aggregated [B, Dim]

        # Handle P_feat dimension if needed
        if P_feat.ndim == 1: P_feat = P_feat.unsqueeze(0) # Add batch dim if missing
        elif P_feat.ndim != 2: raise ValueError(f"Unexpected P_feat shape: {P_feat.shape}")

        A_adapted = self.audio_adapter(A_feat_agg) # [B, adapter_output_dim]
        V_adapted = self.video_adapter(V_feat_agg) # [B, adapter_output_dim]
        P_adapted = self.pers_adapter(P_feat)     # [B, adapter_output_dim]

        # --- Fusion using MoE ---
        combined_features = torch.cat((A_adapted, V_adapted, P_adapted), dim=1) # [B, 3 * adapter_output_dim]
        normalized_features = self.fusion_norm(combined_features) # Apply LayerNorm before MoE
        moe_output = self.moe_layer(normalized_features) # [B, fusion_output_dim]

        # --- Final Classification ---
        logits = self.classifier(moe_output) # [B, num_classes]

        return logits


# --- Training and Evaluation Functions (Identical to previous script) ---

def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad_norm=1.0, scheduler=None):
    """Trains the model for one epoch with gradient clipping and optional LR step."""
    model.train() # Set model to training mode
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_samples = 0
    if dataloader is None or len(dataloader.dataset) == 0: return 0.0, 0.0

    processed_batches = 0
    for batch_idx, batch in enumerate(dataloader):
        # Check batch format
        if not isinstance(batch, dict) or 'A_feat' not in batch or 'V_feat' not in batch or 'personalized_feat' not in batch or 'emo_label' not in batch:
            print(f"Warning: Skipping train batch {batch_idx} due to unexpected format.")
            continue
        try:
            # Move data first
            audio_feat = batch['A_feat'].to(device, non_blocking=True)
            video_feat = batch['V_feat'].to(device, non_blocking=True)
            pers_feat = batch['personalized_feat'].to(device, non_blocking=True)
            labels = batch['emo_label'].to(device, non_blocking=True)

            batch_size = labels.size(0)
            if batch_size == 0: continue # Skip empty batches

            optimizer.zero_grad(set_to_none=True) # More memory efficient zeroing

            # Forward pass
            outputs = model(audio_feat, video_feat, pers_feat)
            loss = criterion(outputs, labels)

            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"      WARNING: NaN or Inf loss encountered in train batch {batch_idx}. Skipping backward/step.")
                continue

            # Backward pass
            loss.backward()

            # Gradient Clipping
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            # Optimizer step
            optimizer.step()

            # LR Scheduler step (if step-based)
            if scheduler is not None and not scheduler.is_epoch_scheduler:
                 scheduler.step()

            # Accumulate results
            total_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_samples += batch_size
            processed_batches += 1

        except Exception as e:
            print(f"      ERROR during train batch {batch_idx}: {e}")
            traceback.print_exc() # Print full traceback for debugging
            continue # Continue to next batch

    # Calculate epoch metrics
    if num_samples == 0:
        print("   Warning: No samples processed in training epoch.")
        return 0.0, 0.0
    avg_loss = total_loss / num_samples
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    # print(f"   Train batches processed: {processed_batches}") # Optional debug
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, num_classes):
    """Evaluates the model on a given dataset."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_samples = 0
    if dataloader is None or len(dataloader.dataset) == 0: return 0.0, 0.0, {}

    processed_batches = 0
    with torch.no_grad(): # Disable gradient calculations
        for batch_idx, batch in enumerate(dataloader):
            # Check batch format
            if not isinstance(batch, dict) or 'A_feat' not in batch or 'V_feat' not in batch or 'personalized_feat' not in batch or 'emo_label' not in batch:
                print(f"Warning: Skipping eval batch {batch_idx} due to unexpected format.")
                continue
            try:
                # Move data
                audio_feat = batch['A_feat'].to(device, non_blocking=True)
                video_feat = batch['V_feat'].to(device, non_blocking=True)
                pers_feat = batch['personalized_feat'].to(device, non_blocking=True)
                labels = batch['emo_label'].to(device, non_blocking=True)

                batch_size = labels.size(0)
                if batch_size == 0: continue

                # Forward pass
                outputs = model(audio_feat, video_feat, pers_feat)
                loss = criterion(outputs, labels)

                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"      WARNING: NaN or Inf loss encountered in eval batch {batch_idx}. Contribution ignored.")
                    continue

                # Accumulate results
                total_loss += loss.item() * batch_size
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                num_samples += batch_size
                processed_batches += 1
            except Exception as e:
                print(f"      ERROR during eval batch {batch_idx}: {e}")
                traceback.print_exc() # Print full traceback for debugging
                continue # Continue to next batch

    # Calculate final metrics
    if num_samples == 0:
        print("   Warning: No samples processed in evaluation.")
        return 0.0, 0.0, {}
    avg_loss = total_loss / num_samples
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0

    # Generate classification report
    try:
        report = classification_report(
            all_labels, all_preds, zero_division=0, output_dict=True,
            labels=list(range(num_classes)), target_names=[f'Class_{i}' for i in range(num_classes)]
        )
    except Exception as e:
       print(f"     Error generating classification report: {e}")
       report = {} # Return empty dict on error

    # print(f"   Eval batches processed: {processed_batches}") # Optional debug
    return avg_loss, accuracy, report


# --- Learning Rate Scheduler Function (Identical to previous script) ---
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Creates cosine LR schedule with linear warmup. """
    def lr_lambda(current_step):
        # Linear warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        return max(0.0, cosine_decay) # Ensure non-negative

    scheduler = LambdaLR(optimizer, lr_lambda, last_epoch)
    scheduler.is_epoch_scheduler = False # Mark as step-based for train_epoch logic
    return scheduler


# --- Main Execution Block ---
if __name__ == '__main__':

    # ==========================================================================
    # ==                     PARAMETER DEFINITIONS                            ==
    # ==========================================================================
    print(f"--- Script Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print("--- Defining Experiment Parameters (MoE Adapter Fusion Model) ---")

    # --- Paths ---
    DATA_ROOT_PATH = "./data" # **REQUIRED: Set this path**
    MODEL_SAVE_PATH = './best_moe_adapter_fusion_model_v1.pth' # Model save path (Updated name)

    # --- Data & Feature Settings ---
    WINDOW_SPLIT_TIME = 1
    AUDIO_FEATURE_METHOD = "wav2vec" # Subdirectory name for audio features
    VIDEO_FEATURE_METHOD = "openface" # Subdirectory name for video features
    FEATURE_MAX_LEN = 26 # Max seq len (used by dataset for padding/truncation)
    # *** Personalized Feature Dimension is Fixed ***
    PERS_BACKBONE_DIM = 1024

    # --- Classification & Splitting ---
    LABEL_COUNT = 2
    TRACK_OPTION = "Track1" # Affects train/val split function
    VAL_RATIO = 0.1         # Used if TRACK_OPTION is 'Track1'
    VAL_PERCENTAGE = 0.1    # Used if TRACK_OPTION is 'Track2'
    SEED = 32

    # --- Model Architecture (MoE Adapter Fusion) ---
    ADAPTER_HIDDEN_DIM = 64     # Hidden dim within adapter MLPs
    ADAPTER_OUTPUT_DIM = 128    # Output dim of adapters (input to fusion stage)
    # --- MoE Specific ---
    NUM_EXPERTS = 4             # Number of expert networks in the MoE layer
    EXPERT_HIDDEN_DIM = 64      # Hidden dimension within each expert MLP
    FUSION_OUTPUT_DIM = 128     # Output dim of MoE layer (input to final classifier)
    # --- General ---
    DROPOUT_RATE = 0.3          # Dropout rate used in adapters and MoE experts

    # --- Training & Optimization ---
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4        # Base LR for adapters/MoE/classifier
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS_FINAL = 75       # Max number of training epochs
    DEVICE_NAME = "mps"         # "cuda", "mps", or "cpu"
    CLIP_GRAD_NORM = 1.0        # Max norm for gradient clipping (or None)
    WARMUP_PROPORTION = 0.1     # Proportion of total steps for LR warmup

    # --- Loss Function & Balancing ---
    FOCAL_LOSS_GAMMA = 3.0
    MANUAL_CLASS_WEIGHTS = [1.0, 2.5] # Class weights for loss [Class0, Class1, ...]
    USE_WEIGHTED_SAMPLER = True      # Use weighted sampling for training loader?

    # --- Early Stopping ---
    EARLY_STOPPING_PATIENCE = 20 # Stop after N epochs with no improvement

    # ==========================================================================
    # ==                   END PARAMETER DEFINITIONS                          ==
    # ==========================================================================

    # --- Derived Paths ---
    DEV_JSON_PATH = os.path.join(DATA_ROOT_PATH, 'Training', 'labels', 'Training_Validation_files.json')
    PERSONALIZED_FEATURE_PATH = os.path.join(DATA_ROOT_PATH, 'Training', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy') # Path still needed by dataset
    AUDIO_FEATURE_DIR = os.path.join(DATA_ROOT_PATH, 'Training', f"{WINDOW_SPLIT_TIME}s", 'Audio', f"{AUDIO_FEATURE_METHOD}")
    VIDEO_FEATURE_DIR = os.path.join(DATA_ROOT_PATH, 'Training', f"{WINDOW_SPLIT_TIME}s", 'Visual', f"{VIDEO_FEATURE_METHOD}")

    # --- Verify Parameters ---
    print("\n--- Verifying Parameters ---")
    if MANUAL_CLASS_WEIGHTS is not None and len(MANUAL_CLASS_WEIGHTS) != LABEL_COUNT:
        raise ValueError(f"ERROR: Length of MANUAL_CLASS_WEIGHTS ({len(MANUAL_CLASS_WEIGHTS)}) must match LABEL_COUNT ({LABEL_COUNT})")
    print("Parameters verified.")

    # --- Setup Device and Seed ---
    print("\n--- Setting Up Device and Seed ---")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    DEVICE = DEVICE_NAME
    if DEVICE == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            DEVICE = 'cpu'; print("Warning: MPS available but not built. Falling back to CPU.")
        else:
            # torch.mps.manual_seed(SEED) # Optional MPS seeding if needed
            print(f"Using MPS device.")
    elif DEVICE == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        if DEVICE != 'cpu': print(f"Warning: Device '{DEVICE}' not available/specified. Using CPU.")
        DEVICE = 'cpu'
    print(f"Selected device: {DEVICE}")

    # --- Basic Path Checks ---
    print("\n--- Checking Paths ---")
    paths_ok = True
    # Check all paths except personalized feature file (only needed by dataset)
    for p, name in [(DEV_JSON_PATH, "Dev JSON"), (AUDIO_FEATURE_DIR, "Audio Dir"), (VIDEO_FEATURE_DIR, "Video Dir"), (PERSONALIZED_FEATURE_PATH, "Personalized Features")]:
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
    except Exception as e:
        print(f"Error splitting data: {e}"); exit(1)
    if not train_data or not val_data:
        print("Error: Data splitting failed (returned empty lists)."); exit(1)

    # --- Determine Feature Dimensions (Backbone Output Dimensions) ---
    print("\n--- Determining Backbone Feature Dimensions ---")
    audio_backbone_dim, video_backbone_dim = None, None
    # *** Personalized Feature Dimension is Fixed ***
    pers_backbone_dim = PERS_BACKBONE_DIM # Use fixed value
    print(f"   Using fixed Pers Backbone Dim: {pers_backbone_dim}")
    try:
        # Find first .npy file in audio dir to get dimension
        for fname in sorted(os.listdir(AUDIO_FEATURE_DIR)):
            if fname.lower().endswith('.npy'):
                try:
                    feature_shape = np.load(os.path.join(AUDIO_FEATURE_DIR, fname)).shape
                    if len(feature_shape) < 1: raise ValueError("Feature array has no dimensions")
                    audio_backbone_dim = feature_shape[-1]
                    print(f"   Audio Backbone Dim: {audio_backbone_dim} (from {fname})")
                    break
                except Exception as load_e:
                    print(f"     Warn: Could not load/read shape from {fname}: {load_e}")
        if audio_backbone_dim is None: print(f"ERR: No loadable .npy files found in {AUDIO_FEATURE_DIR}."); exit(1)

        # Find first .npy file in video dir
        for fname in sorted(os.listdir(VIDEO_FEATURE_DIR)):
             if fname.lower().endswith('.npy'):
                 try:
                     feature_shape = np.load(os.path.join(VIDEO_FEATURE_DIR, fname)).shape
                     if len(feature_shape) < 1: raise ValueError("Feature array has no dimensions")
                     video_backbone_dim = feature_shape[-1]
                     print(f"   Video Backbone Dim: {video_backbone_dim} (from {fname})")
                     break
                 except Exception as load_e:
                     print(f"     Warn: Could not load/read shape from {fname}: {load_e}")
        if video_backbone_dim is None: print(f"ERR: No loadable .npy files found in {VIDEO_FEATURE_DIR}."); exit(1)

    except Exception as e: print(f"ERR determining feature dims: {e}"); exit(1)

    NUM_CLASSES = LABEL_COUNT
    print(f"Backbone Dims: Audio={audio_backbone_dim}, Video={video_backbone_dim}, Pers={pers_backbone_dim}, Classes={NUM_CLASSES}")

    # --- Create Datasets ---
    print("\n--- Creating Datasets ---");
    try:
        full_train_dataset = AudioVisualDataset(
            json_data=train_data, label_count=LABEL_COUNT,
            personalized_feature_file=PERSONALIZED_FEATURE_PATH,
            max_len=FEATURE_MAX_LEN, audio_path=AUDIO_FEATURE_DIR, video_path=VIDEO_FEATURE_DIR
        )
        val_dataset = AudioVisualDataset(
            json_data=val_data, label_count=LABEL_COUNT,
            personalized_feature_file=PERSONALIZED_FEATURE_PATH,
            max_len=FEATURE_MAX_LEN, audio_path=AUDIO_FEATURE_DIR, video_path=VIDEO_FEATURE_DIR
        )
        print(f"Dataset sizes: Train={len(full_train_dataset)}, Val={len(val_dataset)}")
    except Exception as e: print(f"Error creating datasets: {e}"); traceback.print_exc(); exit(1)
    if len(full_train_dataset) == 0 or len(val_dataset) == 0: print("Error: Datasets are empty after creation."); exit(1)

    # --- Calculate Class Weights and Sampler ---
    focal_loss_weights_tensor = None
    sampler = None
    if NUM_CLASSES > 1:
        print("\n--- Processing Class Distribution ---")
        train_labels = []
        label_extraction_failed = False
        try:
            print("   Extracting labels...")
            for i in range(len(full_train_dataset)):
                try:
                    sample = full_train_dataset[i]
                    if isinstance(sample, dict) and 'emo_label' in sample:
                        train_labels.append(int(sample['emo_label']))
                    else: print(f"Warn: Bad label format index {i}")
                except Exception as sample_e: print(f"Warn: Error access sample {i}: {sample_e}")
            print(f"   Extracted {len(train_labels)} labels.")
            if len(train_labels) != len(full_train_dataset): print(f"   Warn: Label count mismatch.")
            if not train_labels: label_extraction_failed = True
        except Exception as e: print(f"   Error extracting labels: {e}."); label_extraction_failed = True

        if not label_extraction_failed and train_labels:
            class_counts = Counter(train_labels)
            print(f"   Train counts: {dict(sorted(class_counts.items()))}")
            num_train_samples = len(train_labels)
            # --- Loss Weights ---
            if MANUAL_CLASS_WEIGHTS is not None:
                if len(MANUAL_CLASS_WEIGHTS) == NUM_CLASSES:
                    focal_loss_weights_tensor = torch.tensor(MANUAL_CLASS_WEIGHTS, dtype=torch.float).to(DEVICE)
                    print(f"   Using Manual Loss Weights: {focal_loss_weights_tensor.cpu().numpy()}")
                else: print(f"   Warn: Manual weights length mismatch. No weights used.")
            else: print("   Manual weights not specified. No weights used.")
            # --- Sampler ---
            if USE_WEIGHTED_SAMPLER and class_counts and num_train_samples > 0:
                print("   Creating WeightedRandomSampler...")
                class_sample_weights = {cls: num_train_samples / count if count > 0 else 0 for cls, count in class_counts.items()}
                sample_weights_list = [class_sample_weights.get(label, 0) for label in train_labels]
                if sample_weights_list and sum(sample_weights_list) > 0:
                    min_pos_weight = min((w for w in sample_weights_list if w > 0), default=1e-9)
                    safe_sample_weights = [max(w, min_pos_weight * 1e-6) for w in sample_weights_list]
                    sampler = WeightedRandomSampler(torch.DoubleTensor(safe_sample_weights), len(safe_sample_weights), replacement=True)
                    print(f"   Sampler created.")
                else: print("   Warn: Cannot create sampler weights."); sampler = None
            elif USE_WEIGHTED_SAMPLER: print("   Warn: Sampler enabled but cannot create."); sampler = None
            else: print("   Sampler disabled."); sampler = None
        else: print("   Skipping weights/sampler."); sampler = None

    # --- Model Instantiation ---
    print("\n--- Instantiating Model (MoE Adapter Fusion Version) ---")
    try:
        moe_model = MoEAdapterFusionModel( # Instantiate the MoE version
            audio_backbone_dim=audio_backbone_dim,
            video_backbone_dim=video_backbone_dim,
            pers_backbone_dim=pers_backbone_dim,
            adapter_hidden_dim=ADAPTER_HIDDEN_DIM,
            adapter_output_dim=ADAPTER_OUTPUT_DIM,
            num_experts=NUM_EXPERTS,                 # MoE specific
            expert_hidden_dim=EXPERT_HIDDEN_DIM,     # MoE specific
            fusion_output_dim=FUSION_OUTPUT_DIM,     # MoE specific
            num_classes=NUM_CLASSES,
            dropout_rate=DROPOUT_RATE,
        ).to(DEVICE)
        print("Model instantiated successfully.")
        print("Trainable parameters are adapters, MoE layer, and final classifier.")
    except Exception as e: print(f"ERROR: Model instantiation error: {e}"); traceback.print_exc(); exit(1)

    # --- Model Summary ---
    if torchinfo:
        print("\n--- Model Architecture & Parameters ---")
        # Note: Input shapes are for the *forward* method of the model
        example_audio_shape = (BATCH_SIZE, audio_backbone_dim)
        example_video_shape = (BATCH_SIZE, video_backbone_dim)
        example_pers_shape = (BATCH_SIZE, pers_backbone_dim)
        try:
            # Use dummy tensors on the correct device for torchinfo summary
            dummy_audio = torch.randn(example_audio_shape, device=DEVICE)
            dummy_video = torch.randn(example_video_shape, device=DEVICE)
            dummy_pers = torch.randn(example_pers_shape, device=DEVICE)

            torchinfo.summary(moe_model, # Use the moe_model instance
                              input_data=[dummy_audio, dummy_video, dummy_pers],
                              col_names=["input_size", "output_size", "num_params", "trainable"],
                              depth=6, # Increased depth slightly to see MoE details
                              device=DEVICE,
                              verbose=0)
        except Exception as e_summary:
            print(f"Could not generate torchinfo summary: {e_summary}")
            # Fallback manual count
            total_params = sum(p.numel() for p in moe_model.parameters())
            trainable_params = sum(p.numel() for p in moe_model.parameters() if p.requires_grad)
            print(f"Total Parameters (Manual Count): {total_params:,}")
            print(f"Trainable Parameters (Manual Count): {trainable_params:,}")

    # --- Create DataLoaders ---
    print(f"\n--- Creating DataLoaders (Batch Size: {BATCH_SIZE}, Sampler: {sampler is not None}) ---")
    num_workers = 0 # Adjust based on system/OS
    val_batch_size = BATCH_SIZE * 2
    try:
        final_train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=num_workers, shuffle=False if sampler else True, pin_memory=True if DEVICE == 'cuda' else False, drop_last=False)
        final_val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if DEVICE == 'cuda' else False)
        if len(final_train_loader) == 0 or len(final_val_loader) == 0: print("Error: DataLoaders are empty."); exit(1)
        print(f"Train DataLoader length: {len(final_train_loader)}, Val DataLoader length: {len(final_val_loader)}")
    except Exception as e: print(f"Error creating DataLoaders: {e}"); traceback.print_exc(); exit(1)

    # --- Optimizer ---
    print(f"\n--- Setting up Optimizer (AdamW) ---")
    trainable_params = moe_model.parameters() # Optimize parameters of the MoE model
    print(f"Optimizing parameters with LR={LEARNING_RATE}, Weight Decay={WEIGHT_DECAY}")
    final_optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Criterion (Loss Function) ---
    print(f"\n--- Setting up Loss Function (FocalLoss) ---")
    print(f"Using FocalLoss with Gamma={FOCAL_LOSS_GAMMA}, Weights={focal_loss_weights_tensor}")
    final_criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA, weight=focal_loss_weights_tensor).to(DEVICE)

    # --- Learning Rate Scheduler ---
    scheduler = None; lr_scheduler_active = False
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
                 print("Warning: Warmup steps >= Total steps. Adjusting warmup.")
                 NUM_WARMUP_STEPS = max(1, int(TOTAL_TRAINING_STEPS * 0.05))
                 print(f"Using adjusted Warmup steps: {NUM_WARMUP_STEPS}")

            scheduler = get_cosine_schedule_with_warmup(
                final_optimizer, num_warmup_steps=NUM_WARMUP_STEPS, num_training_steps=TOTAL_TRAINING_STEPS
            )
            print("LR Scheduler created.")
            lr_scheduler_active = True
        except Exception as e: print(f"Error setting up LR scheduler: {e}.")
    else: print("Training loader empty. Cannot setup LR scheduler.")

    # --- Training Loop Setup ---
    best_final_macro_f1 = -1.0; best_epoch = -1; epochs_without_improvement = 0; training_log = []

    # --- Final Training Loop ---
    print(f"\n--- Starting Final Training Loop ({NUM_EPOCHS_FINAL} epochs) ---")
    start_train_time = time.time()

    for epoch in range(NUM_EPOCHS_FINAL):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS_FINAL}")
        current_lr = final_optimizer.param_groups[0]['lr']
        print(f"   Learning Rate: {current_lr:.3e}")

        # --- Train ---
        print("   Starting training...")
        # Pass the moe_model to the training function
        train_loss, train_acc = train_epoch(moe_model, final_train_loader, final_optimizer, final_criterion, DEVICE, clip_grad_norm=CLIP_GRAD_NORM, scheduler=scheduler if lr_scheduler_active else None)

        # --- Validate ---
        print("   Starting validation...")
        # Pass the moe_model to the evaluation function
        val_loss, val_acc, val_report = evaluate(moe_model, final_val_loader, final_criterion, DEVICE, NUM_CLASSES)

        current_macro_f1 = 0.0; f1_extracted = False
        if val_report and 'macro avg' in val_report and isinstance(val_report['macro avg'], dict) and 'f1-score' in val_report['macro avg']:
             f1_val = val_report['macro avg']['f1-score']
             if f1_val is not None and not np.isnan(f1_val):
                 current_macro_f1 = float(f1_val); f1_extracted = True
             else: print("   Warn: Val Macro F1 is NaN/None.")
        else: print("   Warn: Macro F1 not found/valid in val report.")

        # --- Print epoch results ---
        print(f"   Results: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, Macro F1={current_macro_f1:.4f}")

        # Log results
        epoch_log = {'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'val_macro_f1': current_macro_f1, 'lr': current_lr }
        training_log.append(epoch_log)

        # --- Check for Best Model & Saving ---
        if f1_extracted and current_macro_f1 > best_final_macro_f1:
            delta = current_macro_f1 - best_final_macro_f1 if best_final_macro_f1 > -1.0 else current_macro_f1
            print(f"   -> Macro F1 improved by {delta:.4f} (from {max(0, best_final_macro_f1):.4f} to {current_macro_f1:.4f}). Saving model...")
            best_final_macro_f1 = current_macro_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            try:
                torch.save(moe_model.state_dict(), MODEL_SAVE_PATH) # Save the MoE model's state dict
                print(f"   -> Saved best model to {MODEL_SAVE_PATH}")
                best_val_report_path = MODEL_SAVE_PATH.replace('.pth', '_best_report.json')
                with open(best_val_report_path, 'w') as f: json.dump(val_report, f, indent=2)
                print(f"   -> Saved best validation report to {best_val_report_path}")
            except Exception as e_save: print(f"   -> Error saving model/report: {e_save}")
        else:
            epochs_without_improvement += 1
            print(f"   -> Macro F1 did not improve for {epochs_without_improvement} epoch(s). Best F1: {max(0, best_final_macro_f1):.4f} at epoch {best_epoch if best_epoch != -1 else 'N/A'}.")

        # --- Early Stopping Check ---
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without validation Macro F1 improvement.")
            break

        epoch_duration = time.time() - epoch_start_time
        print(f"   Epoch completed in {epoch_duration:.2f}s")

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
            # Instantiate the MoEAdapterFusionModel again for evaluation
            eval_model = MoEAdapterFusionModel(
                 audio_backbone_dim=audio_backbone_dim, video_backbone_dim=video_backbone_dim,
                 pers_backbone_dim=pers_backbone_dim, adapter_hidden_dim=ADAPTER_HIDDEN_DIM,
                 adapter_output_dim=ADAPTER_OUTPUT_DIM, num_experts=NUM_EXPERTS,
                 expert_hidden_dim=EXPERT_HIDDEN_DIM, fusion_output_dim=FUSION_OUTPUT_DIM,
                 num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, # Use same dropout for eval consistency if needed, though dropout layers are inactive in .eval()
            ).to(DEVICE)
            eval_model.load_state_dict(torch.load(model_eval_path, map_location=DEVICE))
            eval_model.eval() # Set to evaluation mode
            print("Model loaded successfully.")

            # Re-create criterion for evaluation loss calculation
            eval_criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA, weight=focal_loss_weights_tensor).to(DEVICE)

            # Perform evaluation
            print("Running final evaluation...")
            final_loss, final_acc, final_report = evaluate(eval_model, final_val_loader, eval_criterion, DEVICE, NUM_CLASSES)

            print(f"\nFinal Evaluation Results (using MoE model from epoch {best_epoch}):")
            print(f"   Loss: {final_loss:.4f}")
            print(f"   Accuracy: {final_acc:.4f}")
            if final_report and 'macro avg' in final_report:
                print(f"   Macro F1: {final_report['macro avg'].get('f1-score', 0.0):.4f}")
                print("   Classification Report:")
                print(json.dumps(final_report, indent=2))
            else:
                print("   Classification report could not be generated or was empty.")

        except Exception as e:
            print(f"Error evaluating best model from {model_eval_path}: {e}")
            traceback.print_exc()
    else:
        if best_epoch == -1: print("No best epoch recorded. Skipping final evaluation.")
        else: print(f"Best model path ({model_eval_path}) not found.")

    print(f"\n--- Script End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")