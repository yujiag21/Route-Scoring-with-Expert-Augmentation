"""
Deploy script (main_fine_tuned.py)
Following the style of main.py. Runs both the fine-tuned LoRA model and the
original regression network on pre-processed route JSON, producing three outputs:
  1. Per-reaction 5-class score (1-5 scale)
  2. Route-level 3-class rating (driven by the lowest step score)
  3. Continuous route score from the original NeuralNetwork regression head

Model files (all under model/):
  - fine_tune_encoder_cls.pt      : LoRA fine-tuned encoder + classification head (5 classes)
  - encoder_sdf_prediction.pt     : Pretrained encoder (base weights for the regression branch)
  - main_network_sdf_prediction.pt: Original regression network
"""

import os
import sys
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import canonicalize_route, save_picture, add_indices
from distance_regression.data_processing import data_process, input_process
from fine_tuning.fine_tune_encoder_5_classes import (
    DeepSetEncoder,
    DeepSetEncoderWithLoRAAndCls,
    NeuralNetwork,
)

# ---------- Constants ----------

# 5-class labels (internal 0-4; displayed as 1-5)
LABEL_NAMES_5 = ["1", "2", "3", "4", "5"]

# 5-class (0-4) -> 3-bin: {0,1}->0 Bad, {2,3}->1 Plausible, {4}->2 Good
BIN3_MAP = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
BIN3_NAMES = {0: "Bad", 1: "Plausible", 2: "Good"}


# ---------- Data loading (kept identical to main.py) ----------

def read_data(file_name: str):
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
                tree = distances_dict[mol][j][6]
                distances_dict[mol][j][6] = canonicalize_route(tree)
    return distances_dict, file_list


# ---------- Model loading ----------

def infer_lora_config(ckpt: dict) -> dict:
    """
    Infer architecture hyperparameters from the LoRA checkpoint state_dict.

    Inferable from weights:
      - input_size      : fc1.base.weight shape -> [64, input_size]
      - encoding_size   : fc5.base.weight shape -> [encoding_size, 512]
      - lora_r          : fc1.A shape           -> [out_features, r]
      - num_classes     : cls_head.<last>.weight shape -> [num_classes, hidden]

    Not inferable (scalar hyperparam, not in state_dict):
      - lora_alpha      : returned as None; caller must supply (e.g. via hyperparams.json)
    """
    input_size = ckpt["fc1.base.weight"].shape[1]
    encoding_size = ckpt["fc5.base.weight"].shape[0]
    lora_r = ckpt["fc1.A"].shape[1]

    # cls_head is an nn.Sequential; the last Linear layer holds the class count.
    cls_head_weight_keys = sorted(
        [k for k in ckpt.keys() if k.startswith("cls_head.") and k.endswith(".weight")]
    )
    num_classes = ckpt[cls_head_weight_keys[-1]].shape[0]

    return {
        "input_size": input_size,
        "encoding_size": encoding_size,
        "lora_r": lora_r,
        "num_classes": num_classes,
    }


def load_lora_model(
    checkpoint_path: str,
    model_dir: str,
    device: torch.device,
    lora_alpha: int,
) -> tuple:
    """
    Load the LoRA fine-tuned model. All architecture hyperparameters except
    lora_alpha are inferred from the checkpoint state_dict.

    Returns (model, config_dict).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = infer_lora_config(ckpt)
    cfg["lora_alpha"] = lora_alpha

    pretrained_encoder = DeepSetEncoder(
        input_size=cfg["input_size"],
        encoding_size=cfg["encoding_size"],
    ).to(device).float()
    pretrained_encoder.load_state_dict(
        torch.load(os.path.join(model_dir, "encoder_sdf_prediction.pt"), map_location=device)
    )

    model = DeepSetEncoderWithLoRAAndCls(
        input_size=cfg["input_size"],
        encoding_size=cfg["encoding_size"],
        num_classes=cfg["num_classes"],
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
    ).to(device).float()
    model.load_from_pretrained(pretrained_encoder)
    model.load_state_dict(ckpt)
    model.eval()
    return model, cfg


def load_regression_models(
    model_dir: str,
    encoding_size: int,
    other_feature_size: int,
    device: torch.device,
    mode: str = "sdf",
):
    """Load the original encoder + main_network for the regression branch."""
    enc_state = torch.load(
        os.path.join(model_dir, f"encoder_{mode}_prediction.pt"), map_location=device
    )
    in_dim = enc_state["fc1.weight"].shape[1]

    encoder = DeepSetEncoder(
        input_size=in_dim,
        encoding_size=encoding_size,
    ).to(device).float()
    encoder.load_state_dict(enc_state)
    encoder.eval()

    main_network = NeuralNetwork(
        input_size=encoding_size + other_feature_size,
    ).to(device).float()
    main_network.load_state_dict(
        torch.load(os.path.join(model_dir, f"main_network_{mode}_prediction.pt"), map_location=device)
    )
    main_network.eval()
    return encoder, main_network


# ---------- Inference ----------

@torch.no_grad()
def run_inference(
    df: pd.DataFrame,
    lora_model: DeepSetEncoderWithLoRAAndCls,
    reg_encoder: DeepSetEncoder,
    main_network: NeuralNetwork,
    device: torch.device,
) -> pd.DataFrame:
    """
    Run inference on every route in df. Returns the DataFrame with these columns added:
      - step_scores_5cls       : list[int], per-reaction 5-class prediction (1-5 scale)
      - step_scores_3bin       : list[int], per-reaction 3-class prediction (0=Bad, 1=Plausible, 2=Good)
      - step_labels_5cls       : list[str], readable labels
      - route_rating_3bin      : int, route-level 3-class rating (lowest step decides)
      - route_rating_3bin_name : str, route-level 3-class rating name
      - route_score            : float, regression network's continuous route score
    """
    step_scores_5cls_col = []
    step_scores_3bin_col = []
    step_labels_col = []
    route_rating_3bin_col = []
    route_rating_name_col = []
    route_score_col = []

    for i in range(len(df)):
        row = df.iloc[i]

        feats = torch.tensor(
            np.array(row["feasibility_input"]), dtype=torch.float32, device=device
        )
        inputs_vec = torch.tensor(
            np.array(row["inputs"]), dtype=torch.float32, device=device
        )

        # ---- LoRA classification branch ----
        encoded_lora, logits_steps = lora_model(feats)  # logits_steps: [L, 5]
        pred_5cls = logits_steps.argmax(dim=-1).cpu().numpy()  # 0-4

        # Display on 1-5 scale
        pred_5cls_display = (pred_5cls + 1).tolist()
        pred_3bin = [BIN3_MAP[p] for p in pred_5cls.tolist()]
        route_3bin = int(min(pred_3bin))

        step_labels = [LABEL_NAMES_5[p] for p in pred_5cls.tolist()]

        step_scores_5cls_col.append(pred_5cls_display)
        step_scores_3bin_col.append(pred_3bin)
        step_labels_col.append(step_labels)
        route_rating_3bin_col.append(route_3bin)
        route_rating_name_col.append(BIN3_NAMES[route_3bin])

        # ---- Regression branch ----
        encoded_reg = reg_encoder(feats)
        route_score = main_network(encoded_reg.float(), inputs_vec.float()).cpu().item()
        route_score_col.append(route_score)

    df = df.copy()
    df["step_scores_5cls"] = step_scores_5cls_col
    df["step_scores_3bin"] = step_scores_3bin_col
    df["step_labels_5cls"] = step_labels_col
    df["route_rating_3bin"] = route_rating_3bin_col
    df["route_rating_3bin_name"] = route_rating_name_col
    df["route_score"] = route_score_col
    return df


# ---------- Main ----------

def main(args):
    warnings.simplefilter(action="ignore", category=FutureWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Resolve lora_alpha (the only non-inferable arch hyperparameter) ----
    # Priority: --lora_alpha CLI flag > hyperparams.json > default (16)
    lora_alpha = args.lora_alpha
    if lora_alpha is None:
        hp_path = os.path.join(args.model_dir, "hyperparams.json")
        if os.path.exists(hp_path):
            with open(hp_path) as f:
                hp = json.load(f)
            lora_alpha = hp.get("lora_alpha", 16)
            print(f"Loaded lora_alpha={lora_alpha} from {hp_path}")
        else:
            lora_alpha = 16
            print(f"hyperparams.json not found; using default lora_alpha={lora_alpha}")

    # ---- Load data (identical to main.py) ----
    input_path = os.path.join(ROOT_DIR, "data", args.input_file)
    print(f"\nLoading data from {input_path}...")
    distance_dict, mol_list = read_data(input_path)
    df = data_process(distance_dict, mode=args.mode)
    df = add_indices(df).copy()
    print(f"DataFrame shape: {df.shape}")

    # ---- Load models ----
    ckpt_path = os.path.join(args.model_dir, "fine_tune_encoder_cls.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"LoRA checkpoint not found: {ckpt_path}\n"
            "Pass --model_dir to point at a directory containing fine_tune_encoder_cls.pt."
        )
    print(f"\nLoading LoRA model from {ckpt_path}...")
    lora_model, cfg = load_lora_model(
        checkpoint_path=ckpt_path,
        model_dir=args.model_dir,
        device=device,
        lora_alpha=lora_alpha,
    )
    print(
        f"Inferred config: input_size={cfg['input_size']}, "
        f"encoding_size={cfg['encoding_size']}, num_classes={cfg['num_classes']}, "
        f"lora_r={cfg['lora_r']}, lora_alpha={cfg['lora_alpha']}"
    )

    print("Loading regression model (encoder + main_network)...")
    reg_encoder, main_network = load_regression_models(
        model_dir=args.model_dir,
        encoding_size=cfg["encoding_size"],
        other_feature_size=4,
        device=device,
        mode=args.mode,
    )

    # ---- Inference ----
    print("\nRunning inference...")
    df_out = run_inference(df, lora_model, reg_encoder, main_network, device)

    # ---- Build output DataFrame ----
    output_cols = [
        "molecule_index", "image_index", "SMILES",
        "cost", "stability", "feasibility",
        "reaction_list",
        "route_score",            # continuous regression score
        "route_rating_3bin",      # route-level 3-class (0/1/2)
        "route_rating_3bin_name", # Bad / Plausible / Good
        "step_scores_5cls",       # list: per-step 1-5
        "step_scores_3bin",       # list: per-step 0/1/2
        "step_labels_5cls",       # list: per-step text label
    ]
    available = [c for c in output_cols if c in df_out.columns]
    df_result = df_out[available]

    # ---- Save ----
    output_dir = os.path.join(args.output_path, args.mode)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "prediction_fine_tuned.csv")
    df_result.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")

    if args.save_excel:
        xlsx_path = os.path.join(output_dir, "prediction_fine_tuned.xlsx")
        df_result.to_excel(xlsx_path, index=False)
        print(f"Excel saved to {xlsx_path}")

    if args.save_picture:
        save_picture(df_out, output_dir)
        print(f"Route pictures saved to {output_dir}")

    # ---- Print summary ----
    print(f"\n{'='*65}")
    print(f"{'Mol':>4} {'Route':>5}  {'Route Score':>11}  {'Rating(3bin)':>12}  Steps(1-5)")
    print(f"{'='*65}")
    for _, row in df_result.iterrows():
        steps_str = str(row.get("step_scores_5cls", ""))
        print(
            f"{str(row.get('molecule_index', '')):>4} "
            f"{str(row.get('image_index', '')):>5}  "
            f"{row.get('route_score', 0.0):>11.4f}  "
            f"{str(row.get('route_rating_3bin_name', '')):>12}  "
            f"{steps_str}"
        )

    return df_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fine-tuned LoRA + regression model on pre-processed routes"
    )
    parser.add_argument("--input_file", type=str, default="route_10.json",
                        help="Input JSON file name (under data/)")
    parser.add_argument("--output_path", type=str, default="route_score_fine_tuned",
                        help="Root output directory")
    parser.add_argument("--model_dir", type=str, default=os.path.join(ROOT_DIR, "model"),
                        help="Directory containing model checkpoints (and optional hyperparams.json)")
    parser.add_argument("--mode", type=str, default="sdf",
                        help="Feature mode passed to data_process (sdf / default)")
    # Architecture hyperparameters (input_size / encoding_size / lora_r / num_classes)
    # are auto-inferred from the checkpoint state_dict. Only lora_alpha is a scalar
    # that cannot be recovered from weights; it is read from hyperparams.json by default.
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="Override lora_alpha (default: read from model_dir/hyperparams.json)")
    # Output toggles
    parser.add_argument("--save_picture", action="store_true", default=False,
                        help="Also render route pictures into output_dir")
    parser.add_argument("--save_excel", action="store_true", default=False,
                        help="Also save an .xlsx alongside the CSV")
    args = parser.parse_args()
    main(args)
