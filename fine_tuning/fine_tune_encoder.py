"""
Fine-tune the encoder for per-reaction classification (5 classes by default).

Each route comes with a list of reaction-level scores. In addition to the original
DeepSets sum-aggregated output (encoded = torch.sum(x, dim=0)), the model now
predicts every reaction's score; these per-step predictions can be aggregated to
the lowest score and the average score for route-level evaluation.

Loss used during fine-tuning: Focal loss on per-reaction class logits.
Evaluation metrics: per-reaction accuracy plus route-level lowest/average accuracy.
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

# Make the script runnable from project root or from fine_tuning/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import canonicalize_route, canonicalize_smiles, canonicalize_reactions, save_picture, add_indices
from distance_regression.data_processing import data_process
from fine_tuning.models import DeepSetEncoder, DeepSetEncoderWithLoRAAndCls


# ========== Loss ==========

class FocalLoss(nn.Module):
    """Multi-class focal loss. Input is logits; softmax is applied internally.

    FL = - alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer('alpha', alpha if alpha is not None else None)
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [N, C] or [L, C];  targets: [N] or [L] (Long)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        targets = targets.view(-1)
        log_p_t = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        p_t = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        focal_weight = (1 - p_t).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)
            if alpha_t.dim() == 0:
                focal_weight = alpha_t * focal_weight
            else:
                alpha_class = alpha_t.gather(0, targets)
                focal_weight = alpha_class * focal_weight

        loss = -focal_weight * log_p_t
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


# ========== Dataset ==========

class SmallClsDataset(Dataset):
    """Expects a DataFrame with at least the following columns:
      - 'feasibility_input' : list[list[float]], variable-length set fed to DeepSetEncoder
      - 'inputs'            : list[float], extra route-level features
      - 'step_scores'       : list[int], per-reaction class label sequence
      - 'distance'          : float, used for stratified splitting
    """

    def __init__(self, df, device):
        self.df = df.reset_index(drop=True)
        self.device = device

        # Per-reaction labels
        ycol = 'step_scores'
        if ycol not in self.df.columns:
            raise ValueError("df must contain column 'step_scores'.")
        self._y_raw = self.df[ycol].tolist()
        self.label_map = None
        if isinstance(self._y_raw[0], str):
            uniq = sorted(set(self._y_raw))
            self.label_map = {s: i for i, s in enumerate(uniq)}
            self.y = torch.tensor([self.label_map[s] for s in self._y_raw], dtype=torch.long)
        else:
            # Each route gets a per-reaction label sequence; cast to int64 to avoid float->long warnings
            self.y = [torch.tensor(np.array(y, dtype=np.int64), dtype=torch.long) - 1 for y in self._y_raw]

        # Required columns
        for col in ('feasibility_input', 'inputs', 'distance'):
            if col not in self.df.columns:
                raise ValueError(f"df must contain column '{col}'.")

        self.feats_list = self.df['feasibility_input'].tolist()
        self.inputs = self.df['inputs'].tolist()
        self.distances = self.df['distance'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        list_inputs = torch.tensor(np.array(self.feats_list[idx]), dtype=torch.float32, device=self.device)
        x2 = torch.tensor(np.array(self.inputs[idx]), dtype=torch.float32, device=self.device)
        y = self.y[idx].to(self.device).long()
        distance_val = torch.tensor(float(self.distances[idx]), dtype=torch.float32, device=self.device)
        return list_inputs, x2, y, distance_val


def collate_keep_list(batch):
    """Custom collate that keeps the per-route variable-length structure.

    batch: list of (list_inputs[L_i, D], x2[D2], y_steps[L_i], distance)
    Returns:
      - list_inputs_list: length-B list, each [L_i, D]
      - x2:               [B, D2]
      - y_list:           length-B list, each [L_i] LongTensor
      - distance:         [B]
    """
    list_inputs_list, x2_list, y_list, distance_list = zip(*batch)
    x2 = torch.stack(x2_list, dim=0)
    distance = torch.stack(distance_list, dim=0)
    return list(list_inputs_list), x2, list(y_list), distance


# ========== Train & evaluate ==========

def evaluate_cls(model: DeepSetEncoderWithLoRAAndCls, loader: DataLoader):
    model.eval()
    correct = 0
    total = 0
    all_y = []
    all_pred = []
    pr = None
    sr = None
    mse = None  # no regression head; MSE left as None
    accs = None
    per_class = None
    precision = None
    recall = None
    with torch.no_grad():
        for list_inputs_batch, x2_batch, y_list, distance_batch in loader:
            for j in range(len(y_list)):
                encoded, logits_steps = model(list_inputs_batch[j])  # logits_steps: [L_j, C]
                pred_steps = logits_steps.argmax(dim=-1)
                y_steps = y_list[j].long()
                correct += (pred_steps == y_steps).sum().item()
                total += y_steps.numel()
                all_y.append(y_steps.detach().cpu().numpy())
                all_pred.append(pred_steps.detach().cpu().numpy())

    acc = correct / max(1, total)
    f1 = None
    cm = None
    try:
        from scipy.stats import pearsonr, spearmanr
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

        # Route score, 5-bin
        all_y_lowest = [np.min(i) for i in all_y]
        all_pred_lowest = [np.min(i) for i in all_pred]
        pr, _ = pearsonr(all_y_lowest, all_pred_lowest)
        sr, _ = spearmanr(all_y_lowest, all_pred_lowest)
        acc_5_bin_route = (pd.Series(all_y_lowest) == pd.Series(all_pred_lowest)).mean()

        # Route score, 3-bin: {0,1}->0, {2,3}->1, {4}->2
        all_y_3_bin = [pd.Series(i).replace({1: 0, 2: 1, 3: 1, 4: 2}) for i in all_y]
        all_pred_3_bin = [pd.Series(i).replace({1: 0, 2: 1, 3: 1, 4: 2}) for i in all_pred]
        pr_3_bin_route, _ = pearsonr([np.mean(i) for i in all_y_3_bin], [np.mean(i) for i in all_pred_3_bin])
        sr_3_bin_route, _ = spearmanr([np.min(i) for i in all_y_3_bin], [np.min(i) for i in all_pred_3_bin])
        acc_3_bin_route = (pd.Series([np.min(i) for i in all_y_3_bin]) == pd.Series([np.min(i) for i in all_pred_3_bin])).mean()
        f1 = f1_score([np.min(i) for i in all_y_3_bin], [np.min(i) for i in all_pred_3_bin], average='macro')
        cm = confusion_matrix([np.min(i) for i in all_y_3_bin], [np.min(i) for i in all_pred_3_bin])

        all_y_lowest = [np.min(i) for i in all_y_3_bin]
        all_pred_lowest = [np.min(i) for i in all_pred_3_bin]

        precision = precision_score(all_y_lowest, all_pred_lowest, average='macro', zero_division=0)
        recall = recall_score(all_y_lowest, all_pred_lowest, average='macro', zero_division=0)
        f1_lowest = f1_score(all_y_lowest, all_pred_lowest, average='macro')
        precision_pc = precision_score(all_y_lowest, all_pred_lowest, average=None, zero_division=0)
        recall_pc = recall_score(all_y_lowest, all_pred_lowest, average=None, zero_division=0)
        f1_pc = f1_score(all_y_lowest, all_pred_lowest, average=None)
        per_class = {
            "precision": precision_pc.tolist(),
            "recall": recall_pc.tolist(),
            "f1": f1_pc.tolist(),
            "f1_lowest": f1_lowest,
        }

        # Per-reaction 3-bin accuracies under three different mergings
        all_y = np.concatenate(all_y)
        all_pred = np.concatenate(all_pred)
        acc1 = (pd.Series(all_y).replace({1: 0, 2: 1, 3: 1, 4: 2}) == pd.Series(all_pred).replace({1: 0, 2: 1, 3: 1, 4: 2})).mean()  # 01,23,4
        acc2 = (pd.Series(all_y).replace({2: 1, 3: 2, 4: 2}) == pd.Series(all_pred).replace({2: 1, 3: 2, 4: 2})).mean()              # 0,12,34
        acc3 = (pd.Series(all_y).replace({1: 0, 2: 1, 3: 2, 4: 2}) == pd.Series(all_pred).replace({1: 0, 2: 1, 3: 2, 4: 2})).mean()  # 01,2,34

        accs = [acc1, acc2, acc3, acc_3_bin_route, pr_3_bin_route, sr_3_bin_route, acc_5_bin_route]

    except Exception:
        pass
    return acc, f1, cm, pr, sr, mse, accs, per_class, precision, recall


def train_lora_smalldata(
        df_train,
        df_test,
        *,
        device: torch.device,
        model_dir: str,
        encoding_size: int,
        other_feature_size_base: int,
        lr: float = 3e-4,
        num_epochs: int = 30,
        batch_size: int = 16,
        lora_r: int = 8,
        lora_alpha: int = 16,
        num_classes: int = 3,
        seed: int = 0,
        save_dir: Optional[str] = None,
        save_model: bool = False,
):
    """LoRA fine-tuning on small data (per-reaction classification).

    Steps:
      1. Load pretrained encoder weights.
      2. Build LoRA + classification-head model and copy the base weights.
      3. Freeze the base; train only LoRA (A, B) + cls_head.
      4. Train for `num_epochs` epochs and report metrics on the test loader each epoch.

    Note: `other_feature_size_base` is the dimension of `inputs` (without `length`).
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---------- Data ----------
    train_set = SmallClsDataset(df_train, device)
    test_set = SmallClsDataset(df_test, device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False,
                              collate_fn=collate_keep_list)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False,
                             collate_fn=collate_keep_list)

    # ---------- Model and weights ----------
    in_dim = train_set[0][0].shape[-1]
    pretrained_encoder = DeepSetEncoder(
        input_size=in_dim,
        encoding_size=encoding_size,
    ).to(device).float()

    enc_path = os.path.join(model_dir, "encoder_sdf_prediction.pt")
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Pretrained encoder not found: {enc_path}")
    pretrained_encoder.load_state_dict(torch.load(enc_path, map_location=device))

    model = DeepSetEncoderWithLoRAAndCls(
        input_size=in_dim,
        encoding_size=encoding_size,
        num_classes=num_classes,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    ).to(device).float()
    model.load_from_pretrained(pretrained_encoder)

    # ---------- Optimizer and loss ----------
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=1e-4)
    focal = FocalLoss(gamma=2.0, alpha=0.75, reduction='sum')

    save_root = save_dir if save_dir is not None else "lora_5_classes_ckpt"
    save_root = os.path.join(save_root, f"seed{seed}")

    if save_model:
        os.makedirs(save_root, exist_ok=True)
        hyperparams = {
            "seed": seed,
            "encoding_size": encoding_size,
            "other_feature_size_base": other_feature_size_base,
            "learning_rate": lr,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "num_classes": num_classes,
            "device": str(device),
            "model_dir": model_dir,
            "save_dir": save_root,
        }
        with open(os.path.join(save_root, "hyperparams.json"), "w") as f:
            json.dump(hyperparams, f, indent=2)

    # ---------- Training ----------
    final_metrics = None
    for epoch in range(1, num_epochs + 1):
        model.train()

        total, correct, total_loss, total_steps = 0, 0, 0.0, 0

        for list_inputs_batch, x2_batch, y_list, _ in train_loader:
            batch_loss = 0.0
            steps_in_batch = 0
            for j in range(len(y_list)):
                encoded, logits_steps = model(list_inputs_batch[j])
                y_steps = y_list[j]
                loss = focal(logits_steps, y_steps)
                batch_loss += loss
                pred_steps = logits_steps.argmax(dim=-1)
                correct += (pred_steps == y_steps).sum().item()
                total += y_steps.numel()
                steps_in_batch += y_steps.numel()
            if steps_in_batch == 0:
                batch_loss = torch.tensor(0.0, device=device)

            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            total_loss += batch_loss.item()
            total_steps += steps_in_batch

        train_acc = correct / max(1, total)
        train_loss = total_loss / max(1, total_steps)

        # Validation
        val_acc, val_f1, val_cm, val_pr, val_sr, val_mse, val_accs, val_per_class, val_precision, val_recall = \
            evaluate_cls(model, test_loader)

        msg = f"[Epoch {epoch:03d}] TrainLoss={train_loss:.4f} | TrainAcc={train_acc:.3f} | ValAcc={val_acc:.3f}"
        if val_f1 is not None:
            msg += f" | ValF1(macro)={val_f1:.3f}"
        if val_pr is not None:
            msg += f" | val_pr={val_pr:.3f}"
        if val_sr is not None:
            msg += f" | val_sr={val_sr:.3f}"
        if val_accs is not None:
            msg += f" | ValAccs={val_accs}"
        print(msg)
        print(val_cm)

        final_metrics = {
            "epoch": epoch,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_cm": val_cm,
            "val_pr": val_pr,
            "val_sr": val_sr,
            "val_mse": val_mse,
            "val_accs": val_accs,
            "val_per_class": val_per_class,
            "val_precision": val_precision,
            "val_recall": val_recall,
        }

    # Save final model after the last epoch
    if save_model:
        torch.save(model.state_dict(), os.path.join(save_root, "final_model_lora_encoder_cls.pt"))
        try:
            metrics_to_save = dict(final_metrics) if final_metrics is not None else {}
            if metrics_to_save.get("val_cm") is not None:
                metrics_to_save["val_cm"] = metrics_to_save["val_cm"].tolist()
            metrics_to_save["seed"] = seed
            with open(os.path.join(save_root, "final_metrics.json"), "w") as f:
                json.dump(metrics_to_save, f, indent=2)
        except Exception:
            pass

    return {
        **(final_metrics or {}),
        "save_dir": save_root,
    }


def read_data(file_name):
    file_list = []
    distances_dict = {}
    with open(file_name, "r") as fp:
        distance_dict = json.load(fp)
        distances_dict.update(distance_dict)
        mols = list(distances_dict.keys())

        for mol in mols:
            for j in range(len(distances_dict[mol])):
                file_list.append(distances_dict[mol][j][0])
                distances_dict[mol][j].pop(0)
                tree = distances_dict[mol][j][6]  # [molecule name][route index][route]
                distances_dict[mol][j][6] = canonicalize_route(tree)
    return distances_dict, file_list


def make_stratify_bins(series, q=5):
    """Bin a continuous distance column into q quantile bins for stratified splitting.

    Adds a small jitter when duplicate quantile edges would cause qcut to fail.
    """
    x = series.values.astype(float)
    try:
        return pd.qcut(x, q=q, labels=False, duplicates='drop')
    except Exception:
        noise = (np.random.rand(len(x)) - 0.5) * 1e-8
        return pd.qcut(x + noise, q=q, labels=False, duplicates='drop')


def run_kfold_training(
        df,
        *,
        device,
        model_dir: str,
        encoding_size: int,
        other_feature_size_base: int,
        num_classes: int = 3,
        lr: float = 6e-4,
        num_epochs: int = 250,
        batch_size: int = 64,
        lora_r: int = 8,
        lora_alpha: int = 16,
        n_splits: int = 5,
        seed_data: int = 81,
        save_root: str = "checkpoints_lora_small_kfold",
        save_model: bool = False,
):
    """Stratified k-fold cross-validation (stratified by quantile bins of `distance`).

    Each fold calls `train_lora_smalldata` once and reports the final-epoch metrics.
    """
    rng = np.random.RandomState(seed_data)
    strat_bins = make_stratify_bins(df['distance'], q=5)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_data)

    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, strat_bins), start=1):
        df_train = df.iloc[tr_idx].reset_index(drop=True)
        df_test = df.iloc[te_idx].reset_index(drop=True)
        print(f"\n========== Fold {fold}/{n_splits} ==========")
        seed = int(rng.randint(0, 1000))
        out = train_lora_smalldata(
            df_train=df_train,
            df_test=df_test,
            device=device,
            model_dir=model_dir,
            encoding_size=encoding_size,
            other_feature_size_base=other_feature_size_base,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            num_classes=num_classes,
            seed=seed,
            save_dir=os.path.join(save_root, f"fold_{fold}"),
            save_model=save_model,
        )
        print(f'Fold {fold}', out)
        fold_metrics.append(out)

    # Aggregate
    def _mean_std(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    accs = [m.get("val_acc") for m in fold_metrics]
    f1s = [m.get("val_f1") for m in fold_metrics]
    srs = [m.get("val_sr") for m in fold_metrics]
    prs = [m.get("val_pr") for m in fold_metrics]
    mses = [m.get("val_mse") for m in fold_metrics]
    accs_list = [m.get("val_accs") for m in fold_metrics]
    per_class_list = [m.get("val_per_class") for m in fold_metrics]

    # Confusion matrices
    cm_list = [m.get("val_cm") for m in fold_metrics if m is not None and m.get("val_cm") is not None]
    if len(cm_list) > 0:
        try:
            cm_stack = np.stack([np.array(cm) for cm in cm_list], axis=0)
            cm_mean = cm_stack.mean(axis=0)
            cm_std = cm_stack.std(axis=0)
        except Exception:
            cm_mean, cm_std = None, None
    else:
        cm_mean, cm_std = None, None

    # f1_lowest across folds
    f1_lowest_list = []
    for pc in per_class_list:
        if pc is not None and isinstance(pc, dict) and ("f1_lowest" in pc):
            f1_lowest_list.append(pc.get("f1_lowest"))
    f1_lowest_mean, f1_lowest_std = _mean_std(f1_lowest_list)
    acc_mean, acc_std = _mean_std(accs)
    f1_mean, f1_std = _mean_std(f1s)
    srs_mean, srs_std = _mean_std(srs)
    prs_mean, prs_std = _mean_std(prs)
    mses_mean, mses_std = _mean_std(mses)

    print("\n===== K-Fold Summary =====")
    print(f"ValAcc: {acc_mean:.3f} ± {acc_std:.3f}" if acc_mean is not None else "ValAcc: N/A")
    print(f"ValF1 (macro): {f1_mean:.3f} ± {f1_std:.3f}" if f1_mean is not None else "ValF1: N/A")
    print(f"Valsr (macro): {srs_mean:.3f} ± {srs_std:.3f}" if srs_mean is not None else "Valsr: N/A")
    print(f"Valpr (macro): {prs_mean:.3f} ± {prs_std:.3f}" if prs_mean is not None else "Valpr: N/A")

    # ValAccs curve (mean ± std over folds)
    try:
        accs_mean_arr = np.array(accs_list).mean(axis=0)
        accs_std_arr = np.array(accs_list).std(axis=0)
        print(f"ValAccs: {np.array2string(accs_mean_arr, precision=3)} ± {np.array2string(accs_std_arr, precision=3)}")
    except Exception:
        pass

    if f1_lowest_mean is not None:
        print(f"F1_lowest: {f1_lowest_mean:.3f} ± {f1_lowest_std:.3f}")

    if cm_mean is not None:
        print("Confusion Matrix (mean):")
        print(np.array2string(cm_mean, precision=3))
        print("Confusion Matrix (std):")
        print(np.array2string(cm_std, precision=3))

    # Per-class precision/recall/f1 mean ± std
    pc_prec_list = []
    pc_rec_list = []
    pc_f1_list = []
    for pc in per_class_list:
        if pc is None:
            continue
        if "precision" in pc and pc["precision"] is not None:
            pc_prec_list.append(np.array(pc["precision"], dtype=float))
        if "recall" in pc and pc["recall"] is not None:
            pc_rec_list.append(np.array(pc["recall"], dtype=float))
        if "f1" in pc and pc["f1"] is not None:
            pc_f1_list.append(np.array(pc["f1"], dtype=float))
    per_class_summary = None
    if len(pc_prec_list) > 0 and len(pc_rec_list) > 0 and len(pc_f1_list) > 0:
        try:
            prec_stack = np.stack(pc_prec_list, axis=0)
            rec_stack = np.stack(pc_rec_list, axis=0)
            f1_stack = np.stack(pc_f1_list, axis=0)
            prec_mean = prec_stack.mean(axis=0)
            prec_std = prec_stack.std(axis=0)
            rec_mean = rec_stack.mean(axis=0)
            rec_std = rec_stack.std(axis=0)
            f1pc_mean = f1_stack.mean(axis=0)
            f1pc_std = f1_stack.std(axis=0)
            per_class_summary = {
                "precision_mean": prec_mean.tolist(),
                "precision_std": prec_std.tolist(),
                "recall_mean": rec_mean.tolist(),
                "recall_std": rec_std.tolist(),
                "f1_mean": f1pc_mean.tolist(),
                "f1_std": f1pc_std.tolist(),
            }
            num_classes_pc = len(prec_mean)
            for c in range(num_classes_pc):
                print(f"Class {c} - Precision: {prec_mean[c]:.3f} ± {prec_std[c]:.3f}, "
                      f"Recall: {rec_mean[c]:.3f} ± {rec_std[c]:.3f}, "
                      f"F1: {f1pc_mean[c]:.3f} ± {f1pc_std[c]:.3f}")
        except Exception:
            per_class_summary = None

    return {
        "fold_metrics": fold_metrics,
        "acc_mean": acc_mean, "acc_std": acc_std,
        "f1_mean": f1_mean, "f1_std": f1_std,
        "srs_mean": srs_mean, "srs_std": srs_std,
        "prs_mean": prs_mean, "prs_std": prs_std,
        "mses_mean": mses_mean, "mses_std": mses_std,
        "accs_mean": np.array(accs_list).mean(axis=0),
        "accs_std": np.array(accs_list).std(axis=0),
        "accs_list": accs_list,
        "per_class_list": per_class_list,
        "cm_mean": cm_mean,
        "cm_std": cm_std,
        "f1_lowest_mean": f1_lowest_mean,
        "f1_lowest_std": f1_lowest_std,
        "per_class_summary": per_class_summary,
        "n_splits": n_splits,
    }


def run_full_training(
        df,
        *,
        device,
        model_dir: str,
        encoding_size: int,
        other_feature_size_base: int,
        num_classes: int = 3,
        lr: float = 6e-4,
        num_epochs: int = 250,
        batch_size: int = 64,
        lora_r: int = 8,
        lora_alpha: int = 16,
        seed: int = 42,
        save_root: str = "checkpoints_lora_full",
        save_model: bool = True,
):
    """Train on all data (no cross-validation).

    Train and "test" sets are both the full DataFrame; the test loader is only
    used to monitor the training curve.
    """
    print(f"\n========== Full Data Training (seed={seed}) ==========")
    print(f"Total samples: {len(df)}")

    out = train_lora_smalldata(
        df_train=df,
        df_test=df,
        device=device,
        model_dir=model_dir,
        encoding_size=encoding_size,
        other_feature_size_base=other_feature_size_base,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        num_classes=num_classes,
        seed=seed,
        save_dir=save_root,
        save_model=save_model,
    )

    print("\n===== Full Training Summary =====")
    print(f"TrainAcc (on full data): {out.get('val_acc'):.3f}" if out.get('val_acc') is not None else "TrainAcc: N/A")
    print(f"TrainF1 (macro): {out.get('val_f1'):.3f}" if out.get('val_f1') is not None else "TrainF1: N/A")
    print(f"TrainSR: {out.get('val_sr'):.3f}" if out.get('val_sr') is not None else "TrainSR: N/A")
    print(f"TrainPR: {out.get('val_pr'):.3f}" if out.get('val_pr') is not None else "TrainPR: N/A")
    if out.get('val_accs') is not None:
        print(f"TrainAccs: {out.get('val_accs')}")
    if out.get('val_cm') is not None:
        print("Confusion Matrix:")
        print(out.get('val_cm'))

    return {
        "train_acc": out.get("val_acc"),
        "train_f1": out.get("val_f1"),
        "train_pr": out.get("val_pr"),
        "train_sr": out.get("val_sr"),
        "train_cm": out.get("val_cm"),
        "train_mse": out.get("val_mse"),
        "train_accs": out.get("val_accs"),
        "train_per_class": out.get("val_per_class"),
        "train_precision": out.get("val_precision"),
        "train_recall": out.get("val_recall"),
        "save_dir": out.get("save_dir"),
        "seed": seed,
        "n_samples": len(df),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_data", type=int, default=24,
                        help="Data-splitting seed.")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--score_type", type=str, default='lowest',
                        choices=['average_round', 'average_floor', 'lowest'],
                        help="Scoring scheme: average_round / average_floor / lowest.")
    parser.add_argument('--save_path', type=str, default='step_scores',
                        help="Save path under results/.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lora_r", type=int, default=12)
    parser.add_argument("--lora_alpha", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--save_model", action='store_true', default=False)
    parser.add_argument("--mode", type=str, default='kfold', choices=['kfold', 'full'],
                        help="Training mode: kfold (cross-validation) or full (use all data).")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs.")
    args = parser.parse_args()

    if args.save_path is not None:
        save_dir = os.path.join(ROOT_DIR, "results", args.save_path, args.score_type)
    else:
        save_dir = os.path.join(ROOT_DIR, "results", "lora_5_classes", args.score_type)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_distance = pd.read_pickle(os.path.join(ROOT_DIR, 'data', 'data_5_classes_step_scores.pkl'))

    # Read route dictionary, canonicalize, and build features
    distance_dict, mol_list = read_data(os.path.join(ROOT_DIR, 'data', 'processed_routes_assessment_step_scores.json'))
    df = data_process(distance_dict, mode='sdf')
    df['distance'] = df_distance['distance']  # used for stratification
    df['lowest_score'] = df_distance['lowest_score'].astype(int) - 1
    df['average_score'] = df_distance['average_score'].round().astype(int) - 1  # round to nearest int
    df['step_scores'] = df_distance['step_scores']

    # Tag the save path with key hyperparameters to avoid overwrites
    hparam_tag = f"bs{args.batch_size}_r{args.lora_r}_a{args.lora_alpha}_lr{args.lr}"

    if args.mode == 'full':
        out = run_full_training(
            df=df,
            device=device,
            model_dir=os.path.join(ROOT_DIR, "model"),
            encoding_size=256,
            other_feature_size_base=4,
            num_classes=5,
            lr=args.lr,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            seed=args.seed_data,
            save_root=os.path.join(save_dir, "full_training", hparam_tag),
            save_model=args.save_model,
        )
        print(out)

        # Log full-training run
        try:
            log_dir = os.path.join(save_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{hparam_tag}_full_training.log")

            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Full Training Mode, seed={args.seed_data}\n")
                lf.write(f"hparams: batch_size={args.batch_size}, lora_r={args.lora_r}, "
                         f"lora_alpha={args.lora_alpha}, lr={args.lr}\n")
                lf.write(f"n_samples: {out.get('n_samples')}\n")
                lf.write(f"TrainAcc: {out.get('train_acc'):.3f}\n" if out.get('train_acc') is not None else "TrainAcc: N/A\n")
                lf.write(f"TrainF1: {out.get('train_f1'):.3f}\n" if out.get('train_f1') is not None else "TrainF1: N/A\n")
                lf.write(f"TrainSR: {out.get('train_sr'):.3f}\n" if out.get('train_sr') is not None else "TrainSR: N/A\n")
                lf.write(f"TrainPR: {out.get('train_pr'):.3f}\n" if out.get('train_pr') is not None else "TrainPR: N/A\n")
                lf.write(f"TrainAccs: {out.get('train_accs')}\n")
                lf.write(f"save_dir: {out.get('save_dir')}\n\n")

            csv_path = os.path.join(save_dir, "results_summary_full.csv")
            csv_row = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "mode": "full",
                "seed": args.seed_data,
                "n_samples": out.get('n_samples'),
                "train_acc": out.get('train_acc'),
                "train_f1": out.get('train_f1'),
                "train_sr": out.get('train_sr'),
                "train_pr": out.get('train_pr'),
                "train_accs": str(out.get('train_accs')),
                "train_cm": out.get('train_cm').tolist() if out.get('train_cm') is not None else None,
                "batch_size": args.batch_size,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lr": args.lr,
            }
            df_row = pd.DataFrame([csv_row])
            header_needed = not os.path.exists(csv_path)
            df_row.to_csv(csv_path, mode='a', header=header_needed, index=False)
        except Exception as e:
            print(f"[Warn] Failed to write full training log: {e}")

    else:
        out = run_kfold_training(
            df=df,
            device=device,
            model_dir=os.path.join(ROOT_DIR, "model"),
            encoding_size=256,
            other_feature_size_base=4,
            num_classes=5,
            lr=args.lr,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            n_splits=args.n_splits,
            seed_data=args.seed_data,
            save_root=os.path.join(save_dir, f"seed_data_{args.seed_data}", hparam_tag),
            save_model=args.save_model,
        )
        print(out)

        # Log all folds at the end of training
        try:
            log_dir = os.path.join(save_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{hparam_tag}_summary.log")
            fold_metrics = out.get("fold_metrics", [])
            acc_list = []
            sr_list = []
            pr_list = []
            mse_list = []
            accs_list = []
            for idx, m in enumerate(fold_metrics, start=1):
                if m is None:
                    continue
                acc_list.append(m.get("val_acc"))
                sr_list.append(m.get("val_sr"))
                pr_list.append(m.get("val_pr"))
                mse_list.append(m.get("val_mse"))
                accs_list.append(m.get("val_accs"))

            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                         f"seed_data={args.seed_data}, n_splits={out.get('n_splits')}\n")
                lf.write(f"hparams: batch_size={args.batch_size}, lora_r={args.lora_r}, "
                         f"lora_alpha={args.lora_alpha}, lr={args.lr}\n")
                lf.write((f"ValAcc: {out.get('acc_mean'):.3f} ± {out.get('acc_std'):.3f}\n")
                         if out.get('acc_mean') is not None else "ValAcc: N/A\n")
                lf.write((f"Valsr (macro): {out.get('srs_mean'):.3f} ± {out.get('srs_std'):.3f}\n")
                         if out.get('srs_mean') is not None else "Valsr: N/A\n")
                lf.write((f"Valpr (macro): {out.get('prs_mean'):.3f} ± {out.get('prs_std'):.3f}\n")
                         if out.get('prs_mean') is not None else "Valpr: N/A\n")
                lf.write((f"ValMSE: {out.get('mses_mean'):.3f} ± {out.get('mses_std'):.3f}\n")
                         if out.get('mses_mean') is not None else "ValMSE: N/A\n")
                lf.write((f"ValAccs: {out.get('accs_mean')} ± {out.get('accs_std')}\n")
                         if out.get('accs_mean') is not None else "ValAccs: N/A\n")
                lf.write(f"fold_acc: {acc_list}\n")
                lf.write(f"fold_sr: {sr_list}\n")
                lf.write(f"fold_pr: {pr_list}\n")
                lf.write(f"fold_mse: {mse_list}\n")
                lf.write(f"fold_accs: {accs_list}\n\n")

            csv_path = os.path.join(save_dir, "results_summary.csv")
            csv_row = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "seed_data": args.seed_data,
                "n_splits": out.get('n_splits'),
                "acc_mean": out.get('acc_mean'),
                "srs_mean": out.get('srs_mean'),
                "prs_mean": out.get('prs_mean'),
                "mses_mean": out.get('mses_mean'),
                "val_accs_mean": out.get('accs_mean'),
                "f1_lowest_mean": out.get('f1_lowest_mean'),
                "pc_precision_mean": (out.get('per_class_summary') or {}).get('precision_mean')
                                      if out.get('per_class_summary') is not None else None,
                "pc_recall_mean": (out.get('per_class_summary') or {}).get('recall_mean')
                                   if out.get('per_class_summary') is not None else None,
                "pc_f1_mean": (out.get('per_class_summary') or {}).get('f1_mean')
                               if out.get('per_class_summary') is not None else None,
                "cm_mean": out.get('cm_mean').tolist() if out.get('cm_mean') is not None else None,
                "batch_size": args.batch_size,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lr": args.lr,
            }

            df_row = pd.DataFrame([csv_row])
            header_needed = not os.path.exists(csv_path)
            df_row.to_csv(csv_path, mode='a', header=header_needed, index=False)
        except Exception as e:
            print(f"[Warn] Failed to write final log: {e}")
