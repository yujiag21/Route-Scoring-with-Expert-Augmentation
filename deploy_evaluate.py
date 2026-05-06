"""
Deploy script: 使用训练好的 LoRA 模型对 mixed_ground_truth_molecules_trees.json 中的路线进行评估。

流程：
  1. 加载 mixed_ground_truth_molecules_trees.json（每个 key 对应一个分子的 10 条 route）
  2. 用 route_feature_processing.py 的逻辑（AiZynthFinder + WeightedScorer + RouteCostScorer）提取特征
  3. 调用 data_process 构建 feasibility_input / inputs
  4. 加载训练好的 LoRA + 分类头模型
  5. 推理并输出每条 route 每步的预测得分以及路线级汇总
"""

import os
import sys
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
import torch

# ---- 项目根目录（脚本位于项目根，因此只需取一层 dirname） ----
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from aizynthfinder.aizynthfinder import AiZynthFinder
from aizynthfinder.context.scoring.scorers import RouteCostScorer
from aizynthfinder.reactiontree import ReactionTree
from route_feature_processing import WeightedScorer
from utils import canonicalize_route
from distance_regression.data_processing import data_process
from fine_tuning.fine_tune_encoder_5_classes import (
    DeepSetEncoder,
    DeepSetEncoderWithLoRAAndCls,
    NeuralNetwork,
)


# ========== Hash 工具函数 ==========

def _short_hash(s: str, length: int = 12) -> str:
    """对字符串取 SHA-256 并截取前 length 位十六进制。"""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]


def compute_route_hash(route_tree: dict) -> str:
    """
    为一条 route 计算唯一 hash。
    基于 route 树的 JSON 序列化（sort_keys 确保稳定性）。
    """
    return _short_hash(json.dumps(route_tree, sort_keys=True, ensure_ascii=False))


def extract_reaction_hashes(route_tree: dict) -> list:
    """
    以与 process_tree (data_processing.py) 完全相同的 DFS(stack) 遍历顺序，
    提取每个 reaction 节点的 hash。
    hash 基于 mapped_reaction_smiles（若不存在则用 reaction smiles）。

    返回: [reaction_hash_0, reaction_hash_1, ...] 顺序与 feasibility_input 一致。
    """
    stack = [route_tree]
    hashes = []
    while stack:
        node = stack.pop()
        if "reaction" in node.get("type", ""):
            # 优先用 mapped_reaction_smiles，其次 smiles
            rxn_smi = node.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smi:
                rxn_smi = node.get("smiles", "")
            hashes.append(_short_hash(rxn_smi))
            # 子节点入栈（与 process_tree 一致）
            for child in node.get("children", []):
                stack.append(child)
        else:
            if "children" in node:
                for child in node["children"]:
                    stack.append(child)
    return hashes


def add_hashes_to_routes(mixed_routes: dict) -> dict:
    """
    为 mixed_routes 中每条 route 原地添加 route_hash 和每个 reaction 的 reaction_hash。
    同时返回一个 {(group_id, route_idx): {"route_hash": ..., "reaction_hashes": [...]}} 的查找表。
    """
    hash_lookup = {}
    for group_id, routes in mixed_routes.items():
        for route_idx, route in enumerate(routes):
            r_hash = compute_route_hash(route)
            rxn_hashes = extract_reaction_hashes(route)
            # 原地写入 route dict
            route["route_hash"] = r_hash
            # 给每个 reaction 节点也写入 reaction_hash（DFS 同序）
            stack = [route]
            rxn_i = 0
            while stack:
                node = stack.pop()
                if "reaction" in node.get("type", ""):
                    node["reaction_hash"] = rxn_hashes[rxn_i]
                    rxn_i += 1
                    for child in node.get("children", []):
                        stack.append(child)
                else:
                    if "children" in node:
                        for child in node["children"]:
                            stack.append(child)
            hash_lookup[(str(group_id), route_idx)] = {
                "route_hash": r_hash,
                "reaction_hashes": rxn_hashes,
            }
    return hash_lookup


# ========== Step 1: 加载并处理原始 route 树 ==========

def load_mixed_routes(json_path: str) -> dict:
    """
    加载 mixed_ground_truth_molecules_trees.json。
    返回: {group_id(str): [route_tree_dict, ...]}
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} groups from {json_path}")
    return data


def process_routes_features(
    mixed_routes: dict,
    finder: AiZynthFinder,
    weighted_scorer: WeightedScorer,
    cost_scorer: RouteCostScorer,
    hash_lookup: dict,
) -> dict:
    """
    对每条 route 用 route_feature_processing 的逻辑提取特征，
    构建 distances_dict（与 data_process 兼容的格式）。

    返回:
        distances_dict: {smiles: [[cost, vol, complexity, feas, rxn_list, 0, route_tree], ...]}
        group_mapping: [(group_id, route_idx, compound, route_hash, reaction_hashes), ...]
            顺序与 entry 插入顺序一致
    """
    distances_dict: dict = {}
    group_mapping = []  # 记录每条 route 的来源信息（按插入顺序）
    global_idx = 0

    for group_id in sorted(mixed_routes.keys(), key=lambda x: int(x)):
        routes = mixed_routes[group_id]
        for route_idx, route in enumerate(routes):
            compound = route["smiles"]
            try:
                finder.target_smiles = compound
                route_tree = ReactionTree.from_dict(route)
                cost = cost_scorer(route_tree)
                features = weighted_scorer(route_tree)  # [volume, complexity, feasibility, reaction_list]
            except Exception as e:
                print(f"[Warn] Group {group_id}, route {route_idx}: feature extraction failed — {e}")
                continue

            route_canonicalized = canonicalize_route(route)
            entry = [cost] + features + [0, route_canonicalized]
            # entry: [cost, volume, complexity, feasibility, reaction_list, 0, route_tree]

            # 从 hash_lookup 中获取预计算的 hash
            h_info = hash_lookup[(str(group_id), route_idx)]
            route_hash = h_info["route_hash"]
            reaction_hashes = h_info["reaction_hashes"]

            distances_dict.setdefault(compound, []).append(entry)
            group_mapping.append((group_id, route_idx, compound, route_hash, reaction_hashes))
            global_idx += 1

    print(f"Processed {global_idx} routes in total.")
    return distances_dict, group_mapping


# ========== Step 2: 加载模型 ==========

def load_model(
    model_dir: str,
    checkpoint_path: str,
    encoding_size: int,
    num_classes: int,
    lora_r: int,
    lora_alpha: int,
    device: torch.device,
):
    """
    加载预训练 encoder 权重 + LoRA 微调权重。
    """
    # 先用一个 dummy input_size；实际在推理时会根据数据确定
    # 我们先读 checkpoint 的 state_dict 来推断 input_size
    ckpt = torch.load(checkpoint_path, map_location=device)
    in_dim = ckpt["fc1.base.weight"].shape[1]  # base Linear weight: [out, in]

    # 构建 pretrained encoder (仅用于 shape 参考)
    pretrained_encoder = DeepSetEncoder(
        input_size=in_dim,
        encoding_size=encoding_size,
    ).to(device).float()
    enc_path = os.path.join(model_dir, "encoder_sdf_prediction.pt")
    pretrained_encoder.load_state_dict(torch.load(enc_path, map_location=device))

    # 构建 LoRA 模型
    model = DeepSetEncoderWithLoRAAndCls(
        input_size=in_dim,
        encoding_size=encoding_size,
        num_classes=num_classes,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    ).to(device).float()

    # 先拷贝 base 权重
    model.load_from_pretrained(pretrained_encoder)
    # 再加载 LoRA + cls_head 权重
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def load_route_score_model(
    model_dir: str,
    encoding_size: int,
    other_feature_size: int,
    device: torch.device,
    mode: str = "sdf",
):
    """
    加载原始 encoder + main_network（用于预测 route distance score），
    参考 main.py 的加载逻辑。
    """
    enc_path = os.path.join(model_dir, f"encoder_{mode}_prediction.pt")
    main_path = os.path.join(model_dir, f"main_network_{mode}_prediction.pt")
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"未找到 encoder 权重: {enc_path}")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"未找到 main_network 权重: {main_path}")

    # 从权重推断 input_size
    enc_state = torch.load(enc_path, map_location=device)
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
    main_network.load_state_dict(torch.load(main_path, map_location=device))
    main_network.eval()

    print(f"Route score model loaded from {model_dir} (encoder + main_network, mode={mode})")
    return encoder, main_network


# ========== Step 3: 推理 ==========

LABEL_NAMES_5 = ["1 (Very Poor)", "2 (Poor)", "3 (Moderate)", "4 (Good)", "5 (Excellent)"]
BIN3_MAP = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}  # 5-class → 3-bin: 01→Good, 23→Moderate, 4→Poor
BIN3_NAMES = ["Bad (0)", "Plausible (1)", "Good(2)"]


@torch.no_grad()
def predict(model, df, device, encoder=None, main_network=None):
    """
    对 df 中每条 route 进行推理。
    - model (LoRA): 预测每步的 5 类得分
    - encoder + main_network (原始模型): 预测 route distance score
    返回:
      results: list of dict，每个 dict 包含 route 的预测信息
      pred_steps_5cls / pred_steps_3bin 以 {reaction_hash: value} 形式输出
    """
    results = []
    for i in range(len(df)):
        row = df.iloc[i]
        feats = torch.tensor(
            np.array(row["feasibility_input"]),
            dtype=torch.float32,
            device=device,
        )
        reaction_hashes = row["reaction_hashes"]  # list of reaction hash strings

        # ---- LoRA 模型: 逐步分类 ----
        encoded_lora, logits_steps = model(feats)  # logits_steps: [L, C]
        pred_steps = logits_steps.argmax(dim=-1).cpu().numpy()  # [L], 0-4
        probs_steps = torch.softmax(logits_steps, dim=-1).cpu().numpy()

        # 3-bin mapping
        pred_steps_3bin = np.array([BIN3_MAP[p] for p in pred_steps])
        route_score_3bin = int(pred_steps_3bin.min())  # lowest
        route_score_5cls = int(pred_steps.min())  # lowest in 5 classes

        # 构建 {reaction_hash: result} 的 dict
        pred_steps_5cls_dict = {
            rxn_h: int(val) for rxn_h, val in zip(reaction_hashes, pred_steps.tolist())
        }
        pred_steps_3bin_dict = {
            rxn_h: int(val) for rxn_h, val in zip(reaction_hashes, pred_steps_3bin.tolist())
        }
        step_probs_dict = {
            rxn_h: probs.tolist() for rxn_h, probs in zip(reaction_hashes, probs_steps)
        }

        # ---- 原始模型: route distance score ----
        route_score = None
        if encoder is not None and main_network is not None:
            encoded = encoder(feats)  # [encoding_size]
            inputs_vec = torch.tensor(
                np.array(row["inputs"]),
                dtype=torch.float32,
                device=device,
            )
            prediction = main_network(encoded.float(), inputs_vec.float())
            route_score = prediction.cpu().item()

        results.append({
            "smiles": row.get("SMILES", ""),
            "route_hash": row["route_hash"],
            "pred_steps_5cls": pred_steps_5cls_dict,
            "pred_steps_3bin": pred_steps_3bin_dict,
            "route_lowest_5cls": route_score_5cls,
            "route_lowest_3bin": route_score_3bin,
            "route_lowest_3bin_name": BIN3_NAMES[route_score_3bin],
            "route_score": route_score,
            "step_probs": step_probs_dict,
        })
    return results


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description="Deploy: evaluate routes with trained LoRA model")
    parser.add_argument(
        "--input_json",
        type=str,
        default=os.path.join(ROOT_DIR, "data", "mixed_ground_truth_molecules_trees.json"),
        help="Path to mixed_ground_truth_molecules_trees.json",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(
            ROOT_DIR,
            "results", "step_scores", "lowest", "full_training",
            "bs64_r12_a20_lr0.006", "seed24",
        ),
        help="Directory containing best_model_lora_encoder_cls.pt and hyperparams.json",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "model"),
        help="Directory containing pretrained encoder_sdf_prediction.pt",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=os.path.join(ROOT_DIR, "results", "deploy_evaluation_results.json"),
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--finder_config",
        type=str,
        default=os.path.join(ROOT_DIR, "finder.yml"),
        help="Path to AiZynthFinder config file",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- 读取超参 ----
    hp_path = os.path.join(args.checkpoint_dir, "hyperparams.json")
    with open(hp_path, "r") as f:
        hparams = json.load(f)
    encoding_size = hparams["encoding_size"]
    num_classes = hparams["num_classes"]
    lora_r = hparams["lora_r"]
    lora_alpha = hparams["lora_alpha"]
    print(f"Hyperparams: encoding_size={encoding_size}, num_classes={num_classes}, lora_r={lora_r}, lora_alpha={lora_alpha}")

    # ---- 1. 加载原始 route 数据 ----
    print("\n[Step 1] Loading route data...")
    mixed_routes = load_mixed_routes(args.input_json)

    # ---- 1b. 计算 route_hash / reaction_hash 并写入 route 树 ----
    print("\n[Step 1b] Computing route & reaction hashes...")
    hash_lookup = add_hashes_to_routes(mixed_routes)

    # 保存添加了 hash 的数据文件
    hashed_data_path = os.path.splitext(args.input_json)[0] + "_with_hashes.json"
    with open(hashed_data_path, "w") as f:
        json.dump(mixed_routes, f, indent=2, ensure_ascii=False)
    print(f"Data with hashes saved to {hashed_data_path}")

    # ---- 2. 用 route_feature_processing 提取特征 ----
    print("\n[Step 2] Extracting features with AiZynthFinder...")
    finder = AiZynthFinder(configfile=args.finder_config)
    finder.stock.select("emolecules")
    finder.expansion_policy.select("uspto-nm")
    finder.filter_policy.select("uspto")
    weighted_scorer = WeightedScorer(finder.config)
    cost_scorer = RouteCostScorer(finder.config)

    distances_dict, group_mapping = process_routes_features(
        mixed_routes, finder, weighted_scorer, cost_scorer, hash_lookup
    )

    # ---- 3. 调用 data_process 处理为 DataFrame ----
    print("\n[Step 3] Processing features into DataFrame...")
    df = data_process(distances_dict, mode="sdf")
    print(f"DataFrame shape after data_process: {df.shape}")

    # data_process 按照 distances_dict 的插入顺序构建行，
    # 过滤空行后保留原始整数 index，因此可以通过 df.index 回查 group_mapping
    group_ids = []
    route_idxs = []
    route_hashes = []
    reaction_hashes_list = []
    for idx in df.index:
        gid, ridx, _, r_hash, rxn_hashes = group_mapping[idx]
        group_ids.append(gid)
        route_idxs.append(ridx)
        route_hashes.append(r_hash)
        reaction_hashes_list.append(rxn_hashes)
    df = df.reset_index(drop=True)
    df["group_id"] = group_ids
    df["route_idx"] = route_idxs
    df["route_hash"] = route_hashes
    df["reaction_hashes"] = reaction_hashes_list

    # ---- 4. 加载模型 ----
    print("\n[Step 4a] Loading trained LoRA model...")
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model_lora_encoder_cls.pt")
    model = load_model(
        model_dir=args.model_dir,
        checkpoint_path=ckpt_path,
        encoding_size=encoding_size,
        num_classes=num_classes,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        device=device,
    )

    print("\n[Step 4b] Loading route score model (encoder + main_network)...")
    other_feature_size = 4  # [cost, price, stability, feasibility]
    encoder, main_network = load_route_score_model(
        model_dir=args.model_dir,
        encoding_size=encoding_size,
        other_feature_size=other_feature_size,
        device=device,
        mode="sdf",
    )

    # ---- 5. 推理 ----
    print("\n[Step 5] Running inference...")
    pred_results = predict(model, df, device, encoder=encoder, main_network=main_network)

    # ---- 6. 按 group 汇总并输出 ----
    print("\n[Step 6] Aggregating results by group...")
    grouped_results = {}
    for i, res in enumerate(pred_results):
        group_id = str(df.iloc[i]["group_id"])
        compound = res["smiles"]

        if group_id not in grouped_results:
            grouped_results[group_id] = {
                "compound": compound,
                "routes": [],
            }
        grouped_results[group_id]["routes"].append({
            "route_hash": res["route_hash"],
            "route_score": res["route_score"],
            "pred_steps_5cls": res["pred_steps_5cls"],
            "pred_steps_3bin": res["pred_steps_3bin"],
            "route_lowest_5cls": res["route_lowest_5cls"],
            "route_lowest_3bin": res["route_lowest_3bin"],
            "route_lowest_3bin_name": res["route_lowest_3bin_name"],
        })

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"Evaluation Results Summary ({len(grouped_results)} groups)")
    print(f"{'='*60}")
    for gid in sorted(grouped_results.keys(), key=lambda x: int(x)):
        grp = grouped_results[gid]
        n_routes = len(grp["routes"])
        scores_3bin = [r["route_lowest_3bin"] for r in grp["routes"]]
        scores_5cls = [r["route_lowest_5cls"] for r in grp["routes"]]
        route_scores = [round(r["route_score"], 3) if r["route_score"] is not None else None for r in grp["routes"]]
        route_hashes = [r["route_hash"][:8] for r in grp["routes"]]
        print(
            f"Group {gid:>3s} | {grp['compound'][:50]:50s} | "
            f"#routes={n_routes:2d} | "
            f"route_scores={route_scores} | "
            f"3bin={scores_3bin} | "
            f"5cls={[s+1 for s in scores_5cls]}"  # +1 to show 1-5 scale
        )

    # 保存结果
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(grouped_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
