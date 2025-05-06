# run_early_fusion_mlp_cbp.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import os
import json
import optuna 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import random 
import traceback 
from collections import Counter 

from dataclasses import dataclass, field
from typing import Optional, List
import torch.nn.functional as F

try:
    from DataClasses.config import Config
    from DataSets.audioVisualDataset import AudioVisualDataset
    from Models.early_fusion_mlp_cbp import EarlyFusionMLPWithCBP
    from Utils.test_val_split import train_val_split1, train_val_split2
except ImportError as e:
     print(f"Import Error: {e}. Please ensure the script is run from the project root directory.")
     exit(1)

try:
    import torchinfo
except ImportError:
    print("torchinfo not found. Install using: pip install torchinfo")
    torchinfo = None

class FocalLoss(nn.Module):
    """
    Focal Loss implementation that correctly handles class weighting.

    Args:
        class_weights (torch.Tensor, optional): A manual rescaling weight given to each class.
                                                If given, has to be a Tensor of size C (number of classes).
                                                Higher weights give more importance to under-represented classes.
                                                Defaults to None (no class weighting).
        gamma (float, optional): Focusing parameter. Higher values down-weight easy examples more.
                                 Defaults to 2.0.
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'. Defaults to 'mean'.
    """
    def __init__(self, class_weights=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # Register weights buffer (moves with the model to CPU/GPU)
        # Ensure it's a float tensor if provided
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float)
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', None)

        self.reduction = reduction

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Model predictions, raw logits expected. Shape (N, C).
            targets (torch.Tensor): True labels, integers 0 <= targets < C. Shape (N,).
        """
        # Calculate base cross entropy loss *without reduction*
        # Pass the class weights directly to F.cross_entropy
        # Ensure weights are on the same device as preds
        current_class_weights = None
        if self.class_weights is not None:
             current_class_weights = self.class_weights.to(preds.device) # Move weights to correct device

        # Calculate Cross Entropy loss per sample, applying class weights here
        # reduction='none' is crucial to get per-sample loss before focal modulation
        ce_loss = F.cross_entropy(preds, targets,
                                  weight=current_class_weights,
                                  reduction='none')

        # Calculate pt = exp(-ce_loss). This is the probability of the true class
        pt = torch.exp(-ce_loss)

        # Calculate the focal loss component: (1 - pt)^gamma * ce_loss
        # The 'alpha' balancing from the paper is effectively handled by the 'weight'
        # parameter passed to F.cross_entropy above.
        focal_term = (1 - pt) ** self.gamma
        focal_loss = focal_term * ce_loss

        # Apply the final reduction
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Choose 'none', 'mean', or 'sum'.")

# --- Helper Function for Dynamic Weight Calculation ---
def calculate_class_weights(labels_np: np.ndarray, num_classes: int) -> Optional[List[float]]:
    if labels_np is None or len(labels_np) == 0:
        print("Error: Cannot calculate weights, no labels provided.")
        return None
    try:
        class_counts = Counter(labels_np)
        total_samples = len(labels_np)
        weights = []
        print(f"  Class counts for weight calculation: {dict(class_counts)}")
        for i in range(num_classes):
            count = class_counts.get(i, 0)
            if count == 0:
                print(f"Warning: Class {i} has 0 samples in the training set. Assigning default weight 1.0.")
                weight = 1.0
            else:
                weight = total_samples / (num_classes * count) # Inverse frequency
            weights.append(weight)
        print(f"  Calculated dynamic class weights: {[f'{w:.4f}' for w in weights]}")
        return weights
    except Exception as e:
        print(f"Error calculating class weights: {e}")
        return None

# --- Training and Evaluation Functions ---
def train_epoch(model, dataloader, optimizer, criterion, device):
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
            if torch.isnan(loss) or torch.isinf(loss): print(f"NaN/Inf loss train batch {batch_idx}. Skipping."); continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_samples += batch_size
        except Exception as e:
            print(f"Error train batch {batch_idx}: {e}"); traceback.print_exc(); continue

    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds) if num_samples > 0 else 0.0 
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_samples = 0
    if dataloader is None or len(dataloader.dataset) == 0: return 0.0, 0.0, {}, 0.0, [], []

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
                if torch.isnan(loss) or torch.isinf(loss): print(f"NaN/Inf loss eval batch {batch_idx}. Skipping."); continue

                total_loss += loss.item() * batch_size
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                num_samples += batch_size
            except Exception as e:
                print(f"Error eval batch {batch_idx}: {e}"); traceback.print_exc(); continue

    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    accuracy = 0.0; f1_weighted = 0.0; report_dict = {}
    if num_samples > 0 and len(all_labels) > 0 and len(all_preds) == len(all_labels):
        accuracy = accuracy_score(all_labels, all_preds)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        try: report_dict = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
        except ValueError: print("Warning: evaluate - Could not generate classification report."); report_dict = {}
    return avg_loss, accuracy, report_dict, f1_weighted, all_labels, all_preds


# --- Optuna Objective Function ---
def objective(trial, full_train_dataset, config):
    cbp_output_dim = trial.suggest_categorical("cbp_output_dim", [1024, 2048, 4096, 8192]) # Added 8192
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.7, step=0.1)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = config.batch_size
    gamma = int(config.gamma)

    n_splits = config.cv_folds
    labels_np = []
    # --- Label Extraction ---
    try:
        label_key = {2: 'bin_category', 3: 'tri_category', 5: 'pen_category'}.get(config.labelcount)
        if label_key and hasattr(full_train_dataset, 'data') and isinstance(full_train_dataset.data, list):
            valid_items = [item for item in full_train_dataset.data if isinstance(item, dict) and label_key in item]
            valid_labels = [int(item[label_key]) for item in valid_items if str(item[label_key]).isdigit()]
            labels_np = np.array(valid_labels)
        else: raise ValueError("Cannot extract labels directly.")
        if len(labels_np) != len(full_train_dataset.data): print(f"Warn: Extracted {len(labels_np)}/{len(full_train_dataset.data)} labels.")
    except Exception as e:
        print(f"Label extract failed ({e}). Falling back.");
        try:
            temp_loader = DataLoader(full_train_dataset, batch_size=max(1, config.batch_size))
            labels_list = [b['emo_label'].numpy() for b in temp_loader if 'emo_label' in b]
            if labels_list: labels_np = np.concatenate(labels_list)
            else: raise ValueError("Label extraction via DataLoader failed.")
        except Exception as e_iter: print(f"Label iter failed: {e_iter}"); return 0.0
    if len(labels_np) == 0: print("Error: No labels extracted for CV."); return 0.0
    # --- End Label Extraction ---

    # --- Weight Calculation ---
    tune_class_weights = None
    if getattr(config, 'calculate_weights_dynamically', False):
        print("Calculating dynamic weights for Optuna objective...")
        tune_class_weights = calculate_class_weights(labels_np, config.num_classes)
    elif hasattr(config, 'class_weights') and config.class_weights and isinstance(config.class_weights, list):
        if len(config.class_weights) == config.num_classes: tune_class_weights = config.class_weights; print(f"Using config weights for Optuna: {tune_class_weights}")
        else: print(f"Warn: Config weights length mismatch. Not using weights.")
    else: print("Weights disabled/unspecified for tuning.")
    tune_weights_tensor = None
    if tune_class_weights:
        try: tune_weights_tensor = torch.tensor(tune_class_weights, dtype=torch.float32).to(config.device)
        except Exception as e: print(f"Warn: Coud not convert tuning weights tensor: {e}.")
    # --- End Weight Calculation ---

    # --- CV Setup ---
    unique_labels, counts = np.unique(labels_np, return_counts=True)
    if len(counts) < 2: print(f"Error: Only {len(counts)} classes. Cannot Stratify."); return 0.0
    min_samples = np.min(counts); actual_n_splits = max(2, min(n_splits, min_samples));
    if actual_n_splits < 2: print(f"Warn: Smallest class ({min_samples}) < 2. Cannot CV."); return 0.0
    if actual_n_splits < n_splits: print(f"Warn: Reducing CV folds to {actual_n_splits}.")
    try: skf = StratifiedKFold(n_splits=actual_n_splits, shuffle=True, random_state=config.seed); list(skf.split(np.zeros(len(labels_np)), labels_np));
    except ValueError as e_skf: print(f"Error setting up SKF: {e_skf}"); return 0.0
    # --- End CV Setup ---

    fold_f1_scores = []
    print(f"\nTrial {trial.number}: cbp_out={cbp_output_dim}, hidden={hidden_dim}, dr={dropout_rate:.2f}, lr={lr:.6f}, wd={weight_decay:.6f}")

    audio_dim, video_dim, pers_dim, num_classes = config.audio_dim, config.video_dim, config.pers_dim, config.num_classes
    if not all([audio_dim, video_dim, pers_dim, num_classes]): print("Error: Invalid dims."); return 0.0

    split_indices = np.arange(len(labels_np))
    global_step_counter = 0

    # --- Fold Loop ---
    fold_exceptions = 0
    for fold, (train_split_idx, val_split_idx) in enumerate(skf.split(split_indices, labels_np)):
        fold_completed_successfully = False
        try:
            print(f"  Fold {fold+1}/{actual_n_splits}...")
            train_orig_idx = split_indices[train_split_idx]; val_orig_idx = split_indices[val_split_idx];
            if np.max(train_orig_idx) >= len(full_train_dataset) or np.max(val_orig_idx) >= len(full_train_dataset): print(f"Error: Fold indices out of bounds. Skipping."); continue
            cv_train_dataset = Subset(full_train_dataset, train_orig_idx); cv_val_dataset = Subset(full_train_dataset, val_orig_idx);
            if len(cv_train_dataset) == 0 or len(cv_val_dataset) == 0: print(f"Warn: Fold empty subset. Skipping."); continue
            fold_batch_size = max(1, batch_size); cv_train_loader = DataLoader(cv_train_dataset, batch_size=fold_batch_size, shuffle=True, num_workers=0, pin_memory=False); cv_val_loader = DataLoader(cv_val_dataset, batch_size=fold_batch_size, shuffle=False, num_workers=0, pin_memory=False);

            try:
                model = EarlyFusionMLPWithCBP(
                    audio_dim=audio_dim, video_dim=video_dim, pers_dim=pers_dim,
                    cbp_output_dim=cbp_output_dim, hidden_dim=hidden_dim,
                    num_classes=num_classes, dropout_rate=dropout_rate
                ).to(config.device)
            except ValueError as e: print(f"Model init error: {e}"); continue

            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = FocalLoss(gamma=gamma, class_weights=tune_weights_tensor).to(config.device)
            best_fold_f1 = 0.0

            # Epoch Loop
            for epoch in range(config.num_epochs_tuning):
                train_loss, train_acc = train_epoch(model, cv_train_loader, optimizer, criterion, config.device)
                if train_acc is None or np.isnan(train_loss): print(f"    Train epoch {epoch+1} failed."); break
                val_loss, val_acc, _, val_f1, _, _ = evaluate(model, cv_val_loader, criterion, config.device)
                if val_f1 is None or np.isnan(val_loss): print(f"    Eval epoch {epoch+1} failed."); break

                best_fold_f1 = max(best_fold_f1, val_f1)
                trial.report(val_f1, global_step_counter)
                global_step_counter += 1
                if trial.should_prune(): raise optuna.TrialPruned()
            else: # Only runs if epoch loop completes without break
                 fold_f1_scores.append(best_fold_f1)
                 print(f"  Fold {fold+1} Best Val F1w: {best_fold_f1:.4f}")
                 fold_completed_successfully = True

        except optuna.TrialPruned: print(f"  Trial pruned during fold {fold+1}."); raise
        except Exception as e_fold: print(f"ERROR during Fold {fold+1}: {e_fold}"); traceback.print_exc(); fold_exceptions += 1;

    # --- End Fold Loop ---

    average_f1 = np.mean(fold_f1_scores) if fold_f1_scores else 0.0
    print(f"Trial {trial.number} completed {len(fold_f1_scores)}/{actual_n_splits} folds. Avg CV F1-score: {average_f1:.4f}")
    if not fold_f1_scores: print(f"Warn: Trial {trial.number} completed no folds successfully."); return 0.0
    return average_f1


# --- Main Execution Block ---
if __name__ == '__main__':

    # --- Config Loading ---
    try: config = Config.from_json('config.json')
    except Exception as e: print(f"FATAL ERROR loading config.json: {e}"); exit(1)
    config.model_save_path = './best_early_fusion_cbp.pth'
    print(f"Model will be saved to: {config.model_save_path}")

    # --- Path Setup & Validation ---
    try: DATA_ROOT_PATH = config.data_root_path; DEV_JSON_PATH = os.path.join(DATA_ROOT_PATH, 'Training', 'labels', 'Training_Validation_files.json'); PERSONALIZED_FEATURE_PATH = os.path.join(DATA_ROOT_PATH, 'Training', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy'); AUDIO_FEATURE_DIR = os.path.join(DATA_ROOT_PATH, 'Training', f"{config.window_split_time}s", 'Audio', f"{config.audio_feature_method}"); VIDEO_FEATURE_DIR = os.path.join(DATA_ROOT_PATH, 'Training', f"{config.window_split_time}s", 'Visual', f"{config.video_feature_method}");
    except AttributeError as e: print(f"FATAL ERROR: Missing path attr in config: {e}"); exit(1)
    required_paths = {"DEV_JSON": DEV_JSON_PATH,"PERS_FEAT": PERSONALIZED_FEATURE_PATH,"AUDIO_DIR": AUDIO_FEATURE_DIR,"VIDEO_DIR": VIDEO_FEATURE_DIR}; paths_ok=True; print("\nValidating paths...");
    for name, path in required_paths.items():
        is_dir = name.endswith("_DIR")
        exists = os.path.isdir(path) if is_dir else os.path.exists(path)
        print(f"  Checking {name}: {path} ... {'Found' if exists else 'NOT FOUND!'}")
        if not exists: paths_ok = False
    if not paths_ok: print("\nFATAL ERROR: Required data paths not found."); exit(1)

    # --- Device and Seed ---
    try: 
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        if config.device == 'mps' and torch.backends.mps.is_available():
            device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")
            print(f"Using MPS: {torch.backends.mps.is_built()}")
        elif config.device == 'cuda' and torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.manual_seed_all(config.seed)
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU.")
        config.device = str(device)
    except Exception as e: print(f"Error setting device/seed: {e}. Using CPU."); config.device = "cpu"; device = torch.device("cpu")

    # --- Data Splitting ---
    print("Splitting data..."); train_data, val_data = [], [];
    try:
        if config.track_option=='Track1': train_data, val_data, _, _ = train_val_split1(DEV_JSON_PATH, val_ratio=0.1, random_seed=config.seed)
        elif config.track_option=='Track2': train_data, val_data, _, _ = train_val_split2(DEV_JSON_PATH, val_percentage=0.1, seed=config.seed)
        else: raise ValueError(f"Invalid track_option '{config.track_option}'.")
    except Exception as e: print(f"FATAL ERROR splitting data: {e}"); exit(1)
    if not train_data or not val_data: print("FATAL ERROR: Data splitting failed."); exit(1)
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}")

    # --- Dimension & Config Checks ---
    print("Determining feature dimensions & checking config...");
    def get_npy_shape(directory):
        for fname in os.listdir(directory):
            if fname.endswith('.npy'):
                try: return np.load(os.path.join(directory, fname), mmap_mode='r').shape[-1]
                except Exception as e: print(f"Warn: Dim check failed for {fname}: {e}"); return None
        return None
    try:
        audio_dim_found = get_npy_shape(AUDIO_FEATURE_DIR)
        video_dim_found = get_npy_shape(VIDEO_FEATURE_DIR)
        if audio_dim_found is None or video_dim_found is None: 
            raise ValueError("Could not determine dims.")
        config.audio_dim = audio_dim_found
        config.video_dim = video_dim_found
        config.pers_dim = 1024
        config.num_classes = config.labelcount
        defaults = {'feature_max_len': 26, 'gamma': 2.0, 'alpha': 0.25, 'num_epochs_tuning': 15, 'num_epochs_final': 50, 'optuna_trials': 50, 'calculate_weights_dynamically': False, 'class_weights': None}
        for attr, default_val in defaults.items():
            if not hasattr(config, attr):
                setattr(config, attr, default_val)
                print(f"Warning: Config missing '{attr}'. Setting default: {default_val}")
    except Exception as e: print(f"FATAL ERROR during dimension/config check: {e}"); exit(1)
    print(f"Config Dims: A={config.audio_dim}, V={config.video_dim}, P={config.pers_dim}, Cls={config.num_classes}, MaxLen={config.feature_max_len}")

    # --- Dataset Creation ---
    print("Creating Datasets...");
    try: full_train_dataset = AudioVisualDataset(json_data=train_data, label_count=config.labelcount, personalized_feature_file=PERSONALIZED_FEATURE_PATH, max_len=config.feature_max_len, audio_path=AUDIO_FEATURE_DIR, video_path=VIDEO_FEATURE_DIR); val_dataset = AudioVisualDataset(json_data=val_data, label_count=config.labelcount, personalized_feature_file=PERSONALIZED_FEATURE_PATH, max_len=config.feature_max_len, audio_path=AUDIO_FEATURE_DIR, video_path=VIDEO_FEATURE_DIR);
    except Exception as e: print(f"FATAL ERROR creating datasets: {e}"); traceback.print_exc(); exit(1)
    if len(full_train_dataset) == 0 or len(val_dataset) == 0: print("FATAL ERROR: Datasets empty."); exit(1)

    # --- Optuna Tuning ---
    print(f"\n--- Starting Hyperparameter Tuning ({config.optuna_trials} trials, Optimizing Weighted F1) ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    try: study.optimize(lambda trial: objective(trial, full_train_dataset, config),
                       n_trials=config.optuna_trials, timeout=getattr(config, 'optuna_timeout', None))
    except Exception as e: print(f"ERROR during Optuna optimization: {e}"); traceback.print_exc();
    print("\n--- Optuna Study Complete ---")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials: print("ERROR: No Optuna trials completed successfully. Cannot proceed."); exit(1)
    best_trial = study.best_trial; best_params = best_trial.params;
    print(f"Best trial #{best_trial.number}: Weighted F1={best_trial.value:.4f}");
    for k, v in best_params.items(): print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- Final Training Setup ---
    print("\n--- Training Final CBP MLP Model ---")
    try:
        final_model = EarlyFusionMLPWithCBP(
            audio_dim=config.audio_dim, video_dim=config.video_dim, pers_dim=config.pers_dim,
            cbp_output_dim=best_params['cbp_output_dim'], hidden_dim=best_params['hidden_dim'],
            num_classes=config.num_classes, dropout_rate=best_params['dropout_rate']
        ).to(device)
    except KeyError as e: print(f"FATAL ERROR: Missing hyperparameter '{e}'."); exit(1)
    except Exception as e: print(f"FATAL ERROR: Final model init failed: {e}"); exit(1)

    # (Device verification)
    model_device = next(final_model.parameters()).device; print(f"Final model on device: {model_device}");
    if str(model_device) != config.device:
        print(f"ERROR: Device mismatch!")
        final_model.to(device)
        model_device = next(final_model.parameters()).device
        print(f"Re-moved. Now on: {model_device}")
        if str(model_device) != config.device: exit(f"Failed move.")

    # (Model Summary)
    if torchinfo:
        print("\n--- Final Model Architecture & Parameters ---")
        example_audio_shape=(config.batch_size, config.audio_dim); example_video_shape=(config.batch_size, config.video_dim); example_pers_shape=(config.batch_size, config.pers_dim);
        try:
             summary_model = EarlyFusionMLPWithCBP(
                 audio_dim=config.audio_dim, video_dim=config.video_dim, pers_dim=config.pers_dim,
                 cbp_output_dim=best_params['cbp_output_dim'], hidden_dim=best_params['hidden_dim'],
                 num_classes=config.num_classes, dropout_rate=best_params['dropout_rate'] ).to('cpu')
             dummy_input_data = {'A_feat': torch.randn(example_audio_shape, device='cpu'),'V_feat': torch.randn(example_video_shape, device='cpu'),'P_feat': torch.randn(example_pers_shape, device='cpu')}
             torchinfo.summary(summary_model, input_data=dummy_input_data, col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=3, device='cpu', verbose=0)
             del summary_model
        except Exception as e_summary: print(f"Summary failed: {e_summary}"); print(f"Total Params: {sum(p.numel() for p in final_model.parameters() if p.requires_grad):,}");
    else: print(f"Total Params: {sum(p.numel() for p in final_model.parameters() if p.requires_grad):,}");

    # --- Calculate Final Weights ---
    final_labels_np = []
    if getattr(config, 'calculate_weights_dynamically', False):
        try:
            label_key = {2: 'bin_category', 3: 'tri_category', 5: 'pen_category'}.get(config.labelcount)
            if label_key and hasattr(full_train_dataset, 'data') and isinstance(full_train_dataset.data, list):
                valid_items = [item for item in full_train_dataset.data if isinstance(item, dict) and label_key in item]
                valid_labels = [int(item[label_key]) for item in valid_items if str(item[label_key]).isdigit()]
                final_labels_np = np.array(valid_labels)
            else: raise ValueError("Cannot extract labels for final weights.")
        except Exception as e: print(f"Final weight label extraction failed ({e}).")

    final_class_weights = None
    if getattr(config, 'calculate_weights_dynamically', False) and len(final_labels_np) > 0:
        print("Calculating dynamic weights for final training...")
        final_class_weights = calculate_class_weights(final_labels_np, config.num_classes)
    elif hasattr(config, 'class_weights') and config.class_weights and isinstance(config.class_weights, list):
        if len(config.class_weights) == config.num_classes: print(f"Using config weights: {config.class_weights}"); final_class_weights = config.class_weights
        else: print(f"Warn: Mismatch config weights/num_classes.")
    else: print("Weights disabled/unspecified for final training.")

    final_weights_tensor = None
    if final_class_weights:
        try: final_weights_tensor = torch.tensor(final_class_weights, dtype=torch.float32).to(device)
        except Exception as e: print(f"Warn: Could not convert final weights tensor: {e}.")
    # --- End Final Weights ---

    # --- Final Loaders, Optimizer, Criterion ---
    try:
        final_train_loader = DataLoader(full_train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=False)
        final_val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=False)
        if len(final_train_loader) == 0 or len(final_val_loader) == 0: raise ValueError("Final loaders empty.")
    except Exception as e: print(f"FATAL ERROR creating final DataLoaders: {e}"); exit(1)

    final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    final_criterion = FocalLoss(gamma=int(config.gamma), class_weights=final_weights_tensor).to(device)
    print(f"Final criterion using gamma={int(config.gamma)}, weights={'YES' if final_weights_tensor is not None else 'NO'}")
    best_final_val_f1 = 0.0; best_epoch = -1; epochs_no_improve = 0; patience = 10;

    print(f"Ensuring final model is on device {config.device}..."); final_model.to(device);

    # --- Final Training Loop ---
    print("\n--- Starting Final Training Loop (Saving based on Validation Weighted F1) ---")
    try:
        for epoch in range(config.num_epochs_final):
            train_loss, train_acc = train_epoch(final_model, final_train_loader, final_optimizer, final_criterion, device)
            val_loss, val_acc, val_report_dict, val_f1, _, _ = evaluate(final_model, final_val_loader, final_criterion, device)

            if val_f1 is None or np.isnan(val_f1): print(f"Epoch {epoch+1}: Warn - Invalid val F1."); val_f1 = -1.0;
            else: print(f"Epoch {epoch+1}/{config.num_epochs_final}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1w={val_f1:.4f}")

            if val_f1 > best_final_val_f1:
                best_final_val_f1 = val_f1; best_epoch = epoch + 1; epochs_no_improve = 0;
                try:
                    save_dir = os.path.dirname(config.model_save_path)
                    if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir)
                    print(f"Created save dir: {save_dir}")
                    torch.save(final_model.state_dict(), config.model_save_path)
                    print(f"  -> Saved best model (Epoch {best_epoch}, Val F1w: {best_final_val_f1:.4f}) to {config.model_save_path}")
                except Exception as e_save: print(f"ERROR saving model: {e_save}")
            else: epochs_no_improve += 1

            if epochs_no_improve >= patience: print(f"Early stopping epoch {epoch+1}."); break
    except Exception as e_final_epoch: print(f"ERROR during final training: {e_final_epoch}"); traceback.print_exc()

    print("\n--- Final Training Complete ---")
    if best_epoch != -1: print(f"Best validation F1-score ({best_final_val_f1:.4f}) at epoch {best_epoch}")
    else: print("No best model saved.")

    # --- Final Evaluation ---
    print("\n--- Evaluating Best Saved Model (Based on Validation F1w) ---")
    if best_epoch != -1 and os.path.exists(config.model_save_path):
        try:
            eval_model = EarlyFusionMLPWithCBP(
                audio_dim=config.audio_dim, video_dim=config.video_dim, pers_dim=config.pers_dim,
                cbp_output_dim=best_params['cbp_output_dim'], hidden_dim=best_params['hidden_dim'],
                num_classes=config.num_classes, dropout_rate=best_params['dropout_rate']
            ).to(device)
            eval_model.load_state_dict(torch.load(config.model_save_path, map_location=device))
            eval_model.eval()

            eval_criterion = FocalLoss(gamma=int(config.gamma), class_weights=final_weights_tensor).to(device)
            _, final_acc, final_report_dict, final_f1, final_true_labels, final_preds = evaluate(eval_model, final_val_loader, eval_criterion, device)

            print(f"Final Best Model Validation Accuracy: {final_acc:.4f}")
            print(f"Final Best Model Validation Weighted F1-Score: {final_f1:.4f}")
            print("Classification Report:")
            if final_true_labels and final_preds: report_str = classification_report(final_true_labels, final_preds, zero_division=0); print(report_str)
            else: print(" (No labels/preds for report)")

            # (Confusion Matrix)
            if final_true_labels and final_preds:
                cm = confusion_matrix(final_true_labels, final_preds)
                print("\nFinal Best Model Confusion Matrix:")
                print(cm)
                try:
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(config.num_classes), yticklabels=range(config.num_classes))
                    plt.xlabel('Predicted Label')
                    plt.ylabel('True Label')
                    plt.title('Confusion Matrix - Early Fusion MLP with CBP')
                    cm_save_path = 'confusion_matrix_early_fusion_cbp.png'
                    cm_save_dir = os.path.dirname(cm_save_path)
                    if cm_save_dir and not os.path.exists(cm_save_dir): os.makedirs(cm_save_dir)
                    plt.savefig(cm_save_path)
                    print(f"\nCM plot saved as {cm_save_path}")
                    plt.close()
                except Exception as e_cm: print(f"ERROR generating confusion matrix: {e_cm}"); traceback.print_exc()
            else: print("\nCould not generate confusion matrix.")

        except Exception as e_eval: print(f"ERROR evaluating best model: {e_eval}"); traceback.print_exc()
    else: print("Best model not saved/found. Skipping final evaluation.")

    print("\n--- Script Finished ---")