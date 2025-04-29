# Import the transformer model
from Models.late_fusion_transformer import TransformerLateFusion # Correct import assumed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import os
import json
import optuna # For hyperparameter tuning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random # For seeding
import torch.nn.functional as F # Needed for the new FocalLoss
from collections import Counter # Needed for weight calculation
from typing import Optional, List # Added for type hints

# Assuming these imports are correct based on your project structure
from DataClasses.config import Config
# from DataLoaders.audioVisualLoader import create_audio_visual_loader # Not used directly in script
from DataSets.audioVisualDataset import AudioVisualDataset
# Assuming your new FocalLoss class definition is in this file or imported correctly
from Utils.focal_loss import FocalLoss
from Utils.test_val_split import train_val_split1, train_val_split2

try:
    import torchinfo
except ImportError:
    print("torchinfo not found. Install using: pip install torchinfo")
    torchinfo = None

# ==============================================================================
# == Helper Function for Dynamic Weight Calculation ==
# ==============================================================================

def calculate_class_weights(labels_np: np.ndarray, num_classes: int) -> Optional[List[float]]:
    """
    Calculates class weights based on inverse frequency.

    Args:
        labels_np (np.ndarray): NumPy array containing all training labels.
        num_classes (int): The total number of classes.

    Returns:
        Optional[List[float]]: List of calculated weights (one per class) or None if calculation fails.
    """
    if labels_np is None or len(labels_np) == 0:
        print("Error: Cannot calculate weights, no labels provided.")
        return None

    try:
        class_counts = Counter(labels_np)
        total_samples = len(labels_np)
        if len(class_counts) != num_classes:
             print(f"Warning: Found {len(class_counts)} classes, but num_classes is {num_classes}. Weights might be inaccurate.")

        weights = []
        for i in range(num_classes):
            count = class_counts.get(i, 0)
            if count == 0:
                print(f"Warning: Class {i} has 0 samples in the training set. Assigning default weight 1.0.")
                weight = 1.0
            else:
                weight = total_samples / (num_classes * count)
            weights.append(weight)

        print(f"Calculated dynamic class weights: {[f'{w:.2f}' for w in weights]}") # Format output
        return weights

    except Exception as e:
        print(f"Error calculating class weights: {e}")
        return None

# ==============================================================================
# == Training and Evaluation Functions ==
# ==============================================================================

# train_epoch remains unchanged
def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_samples = 0
    if dataloader is None or len(dataloader.dataset) == 0: return 0.0, 0.0
    for batch_idx, batch in enumerate(dataloader):
        if not isinstance(batch, dict) or 'A_feat' not in batch: continue
        try:
            audio_feat = batch['A_feat'].to(device, non_blocking=True)
            video_feat = batch['V_feat'].to(device, non_blocking=True)
            pers_feat = batch['personalized_feat'].to(device, non_blocking=True)
            labels = batch['emo_label'].to(device, non_blocking=True)
            batch_size = labels.size(0)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(audio_feat, video_feat, pers_feat)
            loss = criterion(outputs, labels)
            if torch.isnan(loss): print(f"NaN loss train batch {batch_idx}"); continue
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_samples += batch_size
        except Exception as e: print(f"Error train batch {batch_idx}: {e}"); continue
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds) if num_samples > 0 else 0
    return avg_loss, accuracy

# evaluate remains unchanged
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_samples = 0
    if dataloader is None or len(dataloader.dataset) == 0: return 0.0, 0.0, {}, 0.0, 0.0, [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if not isinstance(batch, dict) or 'A_feat' not in batch: continue
            try:
                audio_feat = batch['A_feat'].to(device, non_blocking=True)
                video_feat = batch['V_feat'].to(device, non_blocking=True)
                pers_feat = batch['personalized_feat'].to(device, non_blocking=True)
                labels = batch['emo_label'].to(device, non_blocking=True)
                batch_size = labels.size(0)
                outputs = model(audio_feat, video_feat, pers_feat)
                loss = criterion(outputs, labels)
                if torch.isnan(loss): print(f"NaN loss eval batch {batch_idx}"); continue
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                num_samples += batch_size
            except Exception as e: print(f"Error eval batch {batch_idx}: {e}"); continue
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 0.0; f1_weighted = 0.0; f1_macro = 0.0; report_dict = {}
    if num_samples > 0 and len(all_labels) > 0 and len(all_preds) > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        report_dict = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
    return avg_loss, accuracy, report_dict, f1_weighted, f1_macro, all_labels, all_preds


# ==============================================================================
# == Optuna Objective Function (Optimizing MACRO F1) ==
# ==============================================================================


valid_transformer_configs = [
     # embed_dim, nhead
     (64, 2), (64, 4),
     (128, 2), (128, 4), (128, 8)
     # Add more valid pairs if needed, ensuring embed_dim % nhead == 0
]

# *** MODIFIED: Added computed_weights argument ***
def objective(trial, full_train_dataset, config, computed_weights): # Added computed_weights arg
    """Optuna objective function optimizing for MACRO F1 score using cross-validation."""

    # --- Suggest Hyperparameters ---
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)

    # *** FIX 1: Suggest embed_dim and nhead jointly ***
    valid_transformer_configs = [
         # embed_dim, nhead
         (64, 2), (64, 4),
         (128, 2), (128, 4), (128, 8)
         # Add more valid pairs if needed, ensuring embed_dim % nhead == 0
    ]
    config_index = trial.suggest_int("transformer_config_idx", 0, len(valid_transformer_configs) - 1)
    transformer_embed_dim, transformer_nhead = valid_transformer_configs[config_index]
    # --- End FIX 1 ---

    transformer_num_layers = trial.suggest_categorical("transformer_num_layers", [1, 2])
    # Widen dropout search slightly
    transformer_dropout = trial.suggest_float("transformer_dropout", 0.1, 0.5, step=0.1)
    mlp_hidden_dim = trial.suggest_categorical("mlp_hidden_dim", [64, 128, 256])
    mlp_dropout = trial.suggest_float("mlp_dropout", 0.2, 0.7, step=0.1) # Widen MLP dropout too
    batch_size = config.batch_size
    gamma = trial.suggest_float("gamma", 0.5, 5.0, step=0.5)

    # --- Cross-Validation Setup ---
    # (Label extraction logic - unchanged)
    n_splits = config.cv_folds; labels_np = [];
    try:
        label_key = {2: 'bin_category', 3: 'tri_category', 5: 'pen_category'}.get(config.labelcount);
        if label_key and hasattr(full_train_dataset, 'data') and isinstance(full_train_dataset.data, list):
            valid_items = [item for item in full_train_dataset.data if isinstance(item, dict) and label_key in item];
            valid_labels = [int(item[label_key]) for item in valid_items if str(item[label_key]).isdigit()];
            labels_np = np.array(valid_labels);
            if len(labels_np) != len(full_train_dataset.data): print(f"Warn: Extracted {len(labels_np)}/{len(full_train_dataset.data)} labels.")
        else: raise ValueError("Cannot extract labels directly.")
    except Exception as e:
        print(f"Direct label extract failed ({e}). Falling back.");
        try: 
            temp_bs=1; temp_loader = DataLoader(full_train_dataset, batch_size=temp_bs); labels_list = [b['emo_label'].numpy() for b in temp_loader if isinstance(b, dict) and 'emo_label' in b];
            if labels_list: labels_np = np.concatenate(labels_list)
            else: raise ValueError("DataLoader iter failed.")
        except Exception as e_iter: print(f"Label iter failed: {e_iter}"); return 0.0
    if len(labels_np) == 0: print("Error: No labels extracted for CV split."); return 0.0
    unique_labels, counts = np.unique(labels_np, return_counts=True); print(f"  Label distribution for CV: {dict(zip(unique_labels, counts))}");
    if len(counts) < 2: print(f"Error: Only {len(counts)} unique labels. Cannot Stratify."); return 0.0
    min_samples = np.min(counts); actual_n_splits = max(2, min(n_splits, min_samples));
    if actual_n_splits < 2: print(f"Error: Smallest class ({min_samples}) < 2. Cannot CV."); return 0.0
    if actual_n_splits < n_splits: print(f"Warn: Reducing CV folds to {actual_n_splits}.")
    # --- End CV Setup ---

    skf = StratifiedKFold(n_splits=actual_n_splits, shuffle=True, random_state=config.seed)
    fold_f1_macro_scores = []
    print(f"\nTrial {trial.number}: LR={lr:.6f}, WD={weight_decay:.6f}, Gamma={gamma:.2f}, MLP_H={mlp_hidden_dim}, MLP_DR={mlp_dropout:.2f}")
    # *** Updated print for transformer config ***
    print(f"  Transformer: ConfigIdx={config_index} (Embed={transformer_embed_dim}, Heads={transformer_nhead}), Layers={transformer_num_layers}, DR={transformer_dropout:.2f}")

    audio_dim, video_dim, pers_dim, num_classes = config.audio_dim, config.video_dim, config.pers_dim, config.num_classes
    max_len = config.feature_max_len
    if not all([audio_dim, video_dim, pers_dim, num_classes, max_len is not None]): print("Error: Invalid dims or max_len not found in config."); return 0.0

    valid_indices = np.arange(len(labels_np))

    # Convert computed_weights (list or None) to tensor ONCE before fold loop
    class_weights_tensor = None
    if computed_weights and isinstance(computed_weights, list):
         if len(computed_weights) == num_classes:
             try:
                 class_weights_tensor = torch.tensor(computed_weights, dtype=torch.float32).to(config.device)
             except Exception as e_tensor: print(f"Error converting weights to tensor in objective: {e_tensor}"); class_weights_tensor = None;
         else: class_weights_tensor = None; # Length mismatch
    # --- End weight tensor creation ---

    # *** FIX 2: Initialize global step counter for Optuna reporting ***
    global_step_counter = 0

    for fold, (train_split_idx, val_split_idx) in enumerate(skf.split(valid_indices, labels_np)):
        print(f"  Fold {fold+1}/{actual_n_splits}...")
        # (Dataset/Loader creation - unchanged)
        cv_train_dataset = Subset(full_train_dataset, valid_indices[train_split_idx])
        cv_val_dataset = Subset(full_train_dataset, valid_indices[val_split_idx])
        if len(cv_train_dataset) == 0 or len(cv_val_dataset) == 0: print(f"Warn: Fold {fold+1} empty subset."); continue
        cv_train_loader = DataLoader(cv_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
        cv_val_loader = DataLoader(cv_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)


        try:
            # Use suggested transformer params
            model = TransformerLateFusion(
                audio_dim=audio_dim, video_dim=video_dim, pers_dim=pers_dim, num_classes=num_classes,
                transformer_embed_dim=transformer_embed_dim, transformer_nhead=transformer_nhead, # Use values from suggested config
                transformer_num_layers=transformer_num_layers, transformer_dropout=transformer_dropout,
                mlp_hidden_dim=mlp_hidden_dim, mlp_dropout=mlp_dropout,
                max_seq_len=max_len
            ).to(config.device)
        except ValueError as e: print(f"Model init error: {e}"); continue

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = FocalLoss(gamma=gamma, class_weights=class_weights_tensor, reduction='mean').to(config.device)

        best_fold_f1_macro = 0.0
        epochs_no_improve = 0
        patience = 5

        for epoch in range(config.num_epochs_tuning):
            try:
                train_loss, train_acc = train_epoch(model, cv_train_loader, optimizer, criterion, config.device)
                if train_acc is None: print(f"    Train epoch {epoch+1} failed."); break

                val_loss, val_acc, _, _, current_f1_macro, _, _ = evaluate(model, cv_val_loader, criterion, config.device)
                if current_f1_macro is None or math.isnan(current_f1_macro): print(f"    Eval epoch {epoch+1} failed or returned invalid Macro F1."); break

                print(f"    Epoch {epoch+1}/{config.num_epochs_tuning}: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}, ValF1m={current_f1_macro:.4f}")

                if current_f1_macro > best_fold_f1_macro:
                    best_fold_f1_macro = current_f1_macro
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # *** FIX 2: Use global_step_counter for reporting step ***
                trial.report(current_f1_macro, global_step_counter)
                global_step_counter += 1 # Increment step counter

                if trial.should_prune(): print(f"    Trial pruned at step {global_step_counter} (Val F1m={current_f1_macro:.4f})."); raise optuna.TrialPruned();
                if epochs_no_improve >= patience: print(f"    Early stopping fold {fold+1} at epoch {epoch+1}."); break;
            except optuna.TrialPruned: raise
            except Exception as e_epoch: print(f"Error Fold {fold+1} Epoch {epoch+1}: {e_epoch}"); import traceback; traceback.print_exc(); break;

        fold_f1_macro_scores.append(best_fold_f1_macro)
        print(f"  Fold {fold+1} Best Val MACRO F1: {best_fold_f1_macro:.4f}")

    average_f1_macro = np.mean(fold_f1_macro_scores) if fold_f1_macro_scores else 0.0
    print(f"Trial {trial.number} Avg CV MACRO F1: {average_f1_macro:.4f}")
    if not fold_f1_macro_scores: print(f"Warn: Trial {trial.number} no folds complete."); return 0.0

    return average_f1_macro


if __name__ == '__main__':

    # --- Load Configuration ---
    try: config = Config.from_json('config.json')
    except Exception as e: print(f"Error loading config.json: {e}"); exit()

    # --- Define Paths & Perform Checks ---
    # (Assumes paths and checks are here)
    DATA_ROOT_PATH = config.data_root_path; DEV_JSON_PATH = os.path.join(DATA_ROOT_PATH, 'Training', 'labels', 'Training_Validation_files.json'); PERSONALIZED_FEATURE_PATH = os.path.join(DATA_ROOT_PATH, 'Training', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy');
    try: AUDIO_FEATURE_DIR = os.path.join(DATA_ROOT_PATH, 'Training', f"{config.window_split_time}s", 'Audio', f"{config.audio_feature_method}"); VIDEO_FEATURE_DIR = os.path.join(DATA_ROOT_PATH, 'Training', f"{config.window_split_time}s", 'Visual', f"{config.video_feature_method}");
    except AttributeError as e: print(f"ERROR: Missing path attr in config: {e}"); exit()
    required_paths = {"DEV_JSON": DEV_JSON_PATH,"PERS_FEAT": PERSONALIZED_FEATURE_PATH,"AUDIO_DIR": AUDIO_FEATURE_DIR,"VIDEO_DIR": VIDEO_FEATURE_DIR}; paths_ok=True;
    for name, path in required_paths.items(): 
        is_dir=name.endswith("_DIR"); exists=os.path.isdir(path) if is_dir else os.path.exists(path); 
        if not exists: print(f"ERROR: {name} {'dir ' if is_dir else ''}not found: {path}"); paths_ok=False;
    if not paths_ok: exit()

    # --- Setup Device and Seed ---
    # (Assumes device setup is here)
    torch.manual_seed(config.seed); np.random.seed(config.seed); random.seed(config.seed);
    if config.device == 'mps' and torch.backends.mps.is_available():
        if not torch.backends.mps.is_built(): config.device = 'cpu'; print("MPS not built. Using CPU.")
        else: print("Using MPS."); config.device = 'mps'
    elif config.device == 'cuda' and torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed); print(f"Using CUDA: {torch.cuda.get_device_name(0)}"); config.device = 'cuda';
    else: 
        if config.device != 'cpu': print(f"Warn: Device '{config.device}' unavailable. Using CPU."); config.device = 'cpu'; print("Using CPU.");


    # --- Split Data ---
    # (Assumes splitting logic is here)
    print("Splitting data..."); train_data, val_data = [], [];
    try:
        if config.track_option=='Track1': train_data, val_data, _, _ = train_val_split1(DEV_JSON_PATH, val_ratio=0.1, random_seed=config.seed)
        elif config.track_option=='Track2': train_data, val_data, _, _ = train_val_split2(DEV_JSON_PATH, val_percentage=0.1, seed=config.seed)
        else: print(f"Error: Invalid track_option '{config.track_option}'."); exit()
    except Exception as e: print(f"Error splitting data: {e}"); exit()
    if not train_data or not val_data: print("Error: Data splitting failed."); exit()
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}")

    # --- Determine Feature Dimensions ---
    # (Assumes dimension finding logic is here)
    print("Determining feature dimensions...");
    def get_npy_shape(directory):
        for fname in os.listdir(directory):
            if fname.endswith('.npy'):
                try: return np.load(os.path.join(directory, fname), mmap_mode='r').shape[-1]
                except Exception as e: print(f"Warn: Could not read shape from {fname}: {e}")
        return None
    audio_dim_found = get_npy_shape(AUDIO_FEATURE_DIR); video_dim_found = get_npy_shape(VIDEO_FEATURE_DIR);
    if audio_dim_found is None or video_dim_found is None: print(f"ERROR: Could not determine feature dimensions."); exit()
    print(f"  Dims Found: Audio={audio_dim_found}, Video={video_dim_found}")
    config.audio_dim = audio_dim_found; config.video_dim = video_dim_found; config.pers_dim = 1024; config.num_classes = config.labelcount;
    if not hasattr(config, 'feature_max_len'): print("ERROR: 'feature_max_len' missing in config."); exit()
    if not hasattr(config, 'gamma'): print("WARNING: 'gamma' not found in config.json.")
    if not hasattr(config, 'class_weights'): print("WARNING: 'class_weights' not found in config.json.")
    # *** ADDED: Check for dynamic weight flag ***
    if not hasattr(config, 'calculate_weights_dynamically'): config.calculate_weights_dynamically = False; print("WARNING: 'calculate_weights_dynamically' not found in config.json. Defaulting to False.")
    print(f"Config Dims: A={config.audio_dim}, V={config.video_dim}, P={config.pers_dim}, Cls={config.num_classes}, MaxLen={config.feature_max_len}")


    # --- Create Datasets ---
    # (Assumes dataset creation logic is here)
    print("Creating Datasets...");
    try: full_train_dataset = AudioVisualDataset(json_data=train_data, label_count=config.labelcount, personalized_feature_file=PERSONALIZED_FEATURE_PATH, max_len=config.feature_max_len, audio_path=AUDIO_FEATURE_DIR, video_path=VIDEO_FEATURE_DIR); val_dataset = AudioVisualDataset(json_data=val_data, label_count=config.labelcount, personalized_feature_file=PERSONALIZED_FEATURE_PATH, max_len=config.feature_max_len, audio_path=AUDIO_FEATURE_DIR, video_path=VIDEO_FEATURE_DIR);
    except Exception as e: print(f"Error creating datasets: {e}"); exit()
    if len(full_train_dataset) == 0 or len(val_dataset) == 0: print("Error: Datasets empty."); exit()
    print(f"Datasets created: Train size={len(full_train_dataset)}, Val size={len(val_dataset)}")


    # --- *** Calculate or Load Class Weights *** ---
    print("Determining class weights...")
    computed_weights = None
    labels_np = []
    # Extract labels from the full training dataset ONCE
    try:
        label_key = {2: 'bin_category', 3: 'tri_category', 5: 'pen_category'}.get(config.labelcount)
        if label_key and hasattr(full_train_dataset, 'data') and isinstance(full_train_dataset.data, list):
            valid_items = [item for item in full_train_dataset.data if isinstance(item, dict) and label_key in item]
            valid_labels = [int(item[label_key]) for item in valid_items if str(item[label_key]).isdigit()]
            labels_np = np.array(valid_labels)
        else: raise ValueError("Cannot extract labels directly.")
    except Exception as e:
        print(f"Direct label extract failed ({e}). Falling back.");
        try: 
            temp_bs=1; temp_loader = DataLoader(full_train_dataset, batch_size=temp_bs); labels_list = [b['emo_label'].numpy() for b in temp_loader if isinstance(b, dict) and 'emo_label' in b];
            if labels_list: labels_np = np.concatenate(labels_list)
            else: raise ValueError("DataLoader iter failed.")
        except Exception as e_iter: print(f"Label iter failed: {e_iter}"); exit()
    if len(labels_np) == 0: print("Error: No labels extracted for weight calculation."); exit()

    # Decide whether to calculate dynamically or use config
    if config.calculate_weights_dynamically:
        print("Calculating weights dynamically...")
        computed_weights = calculate_class_weights(labels_np, config.num_classes) # Use helper function
        if computed_weights is None:
             print("Warning: Dynamic weight calculation failed. Proceeding without class weights.")
    elif hasattr(config, 'class_weights') and config.class_weights and isinstance(config.class_weights, list):
        if len(config.class_weights) == config.num_classes:
            print(f"Using class weights from config: {config.class_weights}")
            computed_weights = config.class_weights # Use list from config
        else:
             print(f"Warning: Length of class_weights in config ({len(config.class_weights)}) != num_classes ({config.num_classes}). Not using class weights.")
             computed_weights = None
    else:
        print("No class weights specified in config and dynamic calculation disabled. Proceeding without class weights.")
        computed_weights = None
    # --- *** End Weight Calculation *** ---


    # --- Optuna Hyperparameter Tuning ---
    print(f"\n--- Starting Hyperparameter Tuning ({config.optuna_trials} trials, Optimizing MACRO F1-Score) ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
    try:
        # *** Pass computed_weights (list or None) to the objective function ***
        study.optimize(lambda trial: objective(trial, full_train_dataset, config, computed_weights),
                       n_trials=config.optuna_trials,
                       timeout=getattr(config, 'optuna_timeout', None)) # Use getattr for safety
    except Exception as e: print(f"Optuna optimization failed: {e}"); import traceback; traceback.print_exc();
    print("\n--- Optuna Study Complete ---")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials: print("Error: No Optuna trials completed successfully."); exit()
    best_trial = study.best_trial; best_params = best_trial.params;
    print(f"Best trial #{best_trial.number}: Achieved MACRO F1 = {best_trial.value:.4f}")
    print("Best Hyperparameters Found:")
    for k, v in best_params.items():
        if isinstance(v, float): print(f"  {k}: {v:.6f}")
        else: print(f"  {k}: {v}")


    # --- Final Model Training ---
    print("\n--- Training Final Model with Best Hyperparameters ---")
    try:
        # *** FIX 1 (Consistency): Get embed_dim and nhead from the suggested config index ***
        config_index = best_params['transformer_config_idx']
        final_transformer_embed_dim, final_transformer_nhead = valid_transformer_configs[config_index] # Use same list as defined in objective

        final_model = TransformerLateFusion(
            audio_dim=config.audio_dim, video_dim=config.video_dim, pers_dim=config.pers_dim,
            num_classes=config.num_classes,
            # Use unpacked values:
            transformer_embed_dim=final_transformer_embed_dim,
            transformer_nhead=final_transformer_nhead,
            # Use other best params:
            transformer_num_layers=best_params['transformer_num_layers'],
            transformer_dropout=best_params['transformer_dropout'],
            mlp_hidden_dim=best_params['mlp_hidden_dim'],
            mlp_dropout=best_params['mlp_dropout'],
            max_seq_len=config.feature_max_len
        ).to(config.device)
    except KeyError as e:
        print(f"Error: Missing hyperparameter '{e}' in Optuna's best_params. Check objective function suggestions.")
        exit()
    except Exception as e: print(f"Final model init error: {e}"); exit()

    # --- Model Summary ---
    # (Unchanged)
    if torchinfo:
        print("\n--- Final Model Architecture & Parameters ---"); example_shapes = [(config.batch_size, config.feature_max_len, config.audio_dim),(config.batch_size, config.feature_max_len, config.video_dim),(config.batch_size, config.pers_dim)];
        try:
            # Need to instantiate with correct params for summary too
             summary_model = TransformerLateFusion(
                 audio_dim=config.audio_dim, video_dim=config.video_dim, pers_dim=config.pers_dim, num_classes=config.num_classes,
                 transformer_embed_dim=final_transformer_embed_dim, transformer_nhead=final_transformer_nhead, # Use final values
                 transformer_num_layers=best_params['transformer_num_layers'], transformer_dropout=best_params['transformer_dropout'],
                 mlp_hidden_dim=best_params['mlp_hidden_dim'], mlp_dropout=best_params['mlp_dropout'], max_seq_len=config.feature_max_len
             ).to('cpu') # Summary on CPU
             torchinfo.summary(summary_model, input_size=example_shapes, col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=5, device='cpu', verbose=0);
             del summary_model # Clean up
        except Exception as e_summary: print(f"Could not generate torchinfo summary: {e_summary}"); print(f"Total Trainable Params: {sum(p.numel() for p in final_model.parameters() if p.requires_grad):,}");
    else: print(f"Total Trainable Params: {sum(p.numel() for p in final_model.parameters() if p.requires_grad):,}");


    final_train_loader = DataLoader(full_train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    final_val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    if len(final_train_loader) == 0 or len(final_val_loader) == 0: print("Error: Final loaders empty."); exit()

    final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

    # *** ADDED: Learning Rate Scheduler ***
    # Reduce LR if validation Macro F1 stagnates
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='max', factor=0.2, patience=5)

    # Instantiate Final FocalLoss using best gamma and COMPUTED weights
    best_gamma = best_params.get('gamma', getattr(config, 'gamma', 2.0))
    final_class_weights_tensor = None
    if computed_weights:
         try: final_class_weights_tensor = torch.tensor(computed_weights, dtype=torch.float32).to(config.device)
         except Exception as e_tensor: print(f"Error converting computed weights to tensor: {e_tensor}. Proceeding without weights."); final_class_weights_tensor = None;
    final_criterion = FocalLoss(gamma=best_gamma, class_weights=final_class_weights_tensor, reduction='mean').to(config.device)
    print(f"Final training using Focal Loss (gamma={best_gamma:.2f}, weights={'DYNAMICALLY COMPUTED' if config.calculate_weights_dynamically and final_class_weights_tensor is not None else ('FROM CONFIG' if not config.calculate_weights_dynamically and final_class_weights_tensor is not None else 'NONE')})")

    best_final_val_f1_macro = 0.0; best_epoch = -1; epochs_no_improve_final = 0; patience_final = 10; # Increased patience slightly

    print(f"Ensuring final model is on device {config.device} before training loop...")
    final_model.to(config.device)

    print("--- Starting Final Training Loop (Saving best model based on Validation Macro F1) ---")
    for epoch in range(config.num_epochs_final):
        try:
            # *** Pass scheduler to train_epoch ***
            train_loss, train_acc = train_epoch(final_model, final_train_loader, final_optimizer, final_criterion, config.device, scheduler=None) # Don't step scheduler during training epoch for ReduceLROnPlateau

            val_loss, val_acc, _, _, val_f1_macro, _, _ = evaluate(final_model, final_val_loader, final_criterion, config.device)
            if val_f1_macro is None or math.isnan(val_f1_macro): print(f"Epoch {epoch+1}: Invalid validation Macro F1."); continue

            print(f"Epoch {epoch+1}/{config.num_epochs_final}: TrainLoss={train_loss:.4f}, Acc={train_acc:.4f} | ValLoss={val_loss:.4f}, Acc={val_acc:.4f}, F1m={val_f1_macro:.4f} | LR={final_optimizer.param_groups[0]['lr']:.1e}")

            # *** Step the scheduler based on validation Macro F1 ***
            scheduler.step(val_f1_macro)

            if val_f1_macro > best_final_val_f1_macro:
                best_final_val_f1_macro = val_f1_macro; best_epoch = epoch + 1; epochs_no_improve_final = 0;
                try: 
                    save_dir = os.path.dirname(config.model_save_path);
                    if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir)
                    torch.save(final_model.state_dict(), config.model_save_path); print(f"  -> Saved best model to {config.model_save_path} (Epoch {best_epoch}, Val F1m: {best_final_val_f1_macro:.4f})");
                except Exception as e_save: print(f"Error saving model: {e_save}")
            else:
                epochs_no_improve_final += 1

            # Early stopping check
            # Also stop if LR becomes too small
            current_lr = final_optimizer.param_groups[0]['lr']
            if epochs_no_improve_final >= patience_final or current_lr < 1e-7:
                 print(f"Early stopping final training at epoch {epoch+1}. Patience: {epochs_no_improve_final}/{patience_final}, LR: {current_lr:.1e}")
                 break
        except Exception as e_final_epoch: print(f"Error final epoch {epoch+1}: {e_final_epoch}"); import traceback; traceback.print_exc(); break

    print("\n--- Final Training Complete ---")
    if best_epoch != -1: print(f"Best validation MACRO F1-score ({best_final_val_f1_macro:.4f}) achieved at epoch {best_epoch}")
    else: print("No best model saved during final training.")


    # --- Final Evaluation ---
    print("\n--- Evaluating Best Saved Model (based on Macro F1) on Validation Set ---")
    if best_epoch != -1 and os.path.exists(config.model_save_path):
        try:
            # *** FIX 1 (Consistency): Get embed_dim and nhead from best params for eval model ***
            eval_config_index = best_params['transformer_config_idx']
            eval_transformer_embed_dim, eval_transformer_nhead = valid_transformer_configs[eval_config_index]

            eval_model = TransformerLateFusion(
                audio_dim=config.audio_dim, video_dim=config.video_dim, pers_dim=config.pers_dim, num_classes=config.num_classes,
                # Use consistent best params
                transformer_embed_dim=eval_transformer_embed_dim,
                transformer_nhead=eval_transformer_nhead,
                transformer_num_layers=best_params['transformer_num_layers'],
                transformer_dropout=best_params['transformer_dropout'],
                mlp_hidden_dim=best_params['mlp_hidden_dim'],
                mlp_dropout=best_params['mlp_dropout'],
                max_seq_len=config.feature_max_len
            ).to(config.device)
            eval_model.load_state_dict(torch.load(config.model_save_path, map_location=config.device)); eval_model.eval();

            # Instantiate Eval FocalLoss using best gamma and COMPUTED weights
            # final_class_weights_tensor should still be available
            eval_criterion = FocalLoss(gamma=best_gamma, class_weights=final_class_weights_tensor, reduction='mean').to(config.device)

            # (Evaluation call and reporting - unchanged)
            final_loss, final_acc, final_report_dict, final_f1_weighted, final_f1_macro, final_true_labels, final_preds = evaluate(eval_model, final_val_loader, eval_criterion, config.device)
            print("\n--- Final Validation Results (Best Model) ---"); print(f"Validation Loss: {final_loss:.4f}"); print(f"Validation Accuracy: {final_acc:.4f}"); print(f"Validation Weighted F1-Score: {final_f1_weighted:.4f}"); print(f"Validation MACRO F1-Score: {final_f1_macro:.4f}");
            print("\nClassification Report:"); target_names=[f"Class {i}" for i in range(config.num_classes)];
            try: print(classification_report(final_true_labels, final_preds, zero_division=0, target_names=target_names))
            except Exception as e_report: print(f"Could not generate classification report: {e_report}")

            # --- Confusion Matrix ---
            # (Unchanged)
            if final_true_labels and final_preds:
                try:
                    cm = confusion_matrix(final_true_labels, final_preds); print("\nFinal Confusion Matrix:"); print(cm);
                    plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names); plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Confusion Matrix - Transformer Late Fusion (Best Macro F1 Model)');
                    cm_save_path = 'confusion_matrix_transformer_late_fusion_macro.png'; cm_save_dir = os.path.dirname(cm_save_path);
                    if cm_save_dir and not os.path.exists(cm_save_dir): os.makedirs(cm_save_dir)
                    plt.savefig(cm_save_path); print(f"\nConfusion Matrix plot saved as {cm_save_path}"); plt.close();
                except Exception as e_plot: print(f"\nCould not plot/save confusion matrix: {e_plot}")
            else: print("\nCould not generate confusion matrix (no labels/preds).")
            # --- End Confusion Matrix ---

        except KeyError as e:
             print(f"Error loading best model: Missing hyperparameter '{e}' in Optuna's best_params saved during tuning.")
        except Exception as e: print(f"Error evaluating best model: {e}"); import traceback; traceback.print_exc();
    else: print("Best model not saved/found. Skipping final evaluation.")

    print("\n--- Script Finished ---")