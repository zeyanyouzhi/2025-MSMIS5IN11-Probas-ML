import os
import json
import math
import pickle
import random
from typing import Dict, List

import numpy as np

BASE_DIR = r"D:\GNN_BAD"
BASE_NAMES = ["droit", "croise"]
SPLIT_WEIGHTS = {"train": 90, "val": 20, "test": 10}  # 最终会按 9:2:1 归一化
OUTPUT_DIR = os.path.join(BASE_DIR, "combined_ctr_gcn")
RANDOM_SEED = 42


def load_metadata(base_name: str):
    meta_path = os.path.join(BASE_DIR, f"{base_name}_skeleton", f"{base_name}_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"未找到 {meta_path}，请先为 {base_name} 运行 extract_skeleton.py")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_batches(base_name: str):
    skeleton_dir = os.path.join(BASE_DIR, f"{base_name}_skeleton")
    batch_files = []
    for name in os.listdir(skeleton_dir):
        if name.startswith(f"{base_name}_") and name.endswith(".npy"):
            batch_files.append(os.path.join(skeleton_dir, name))
    batch_files.sort()
    if not batch_files:
        raise RuntimeError(f"在 {skeleton_dir} 未找到任何 {base_name}_*.npy 文件")

    arrays = []
    for path in batch_files:
        arr = np.load(path)
        if arr.ndim != 4:
            raise ValueError(f"{path} 期望 shape (N, T, V, C)，实际 {arr.shape}")
        arrays.append(arr)
        print(f"[INFO] 载入 {path} -> {arr.shape}")
    return arrays


def convert_to_ctr_format(arrays, meta, base_name):
    data = np.concatenate(arrays, axis=0)  # (N, T, V, C)
    data = np.transpose(data, (0, 3, 1, 2))  # (N, C, T, V)
    data = np.expand_dims(data, axis=-1)    # (N, C, T, V, 1)
    if data.shape[0] != len(meta["t0"]):
        raise RuntimeError(f"{base_name} 数据样本数 {data.shape[0]} 与元数据 t0 数 {len(meta['t0'])} 不符")

    sample_names = [f"{base_name}_{idx:05d}_t{t0}" for idx, t0 in enumerate(meta["t0"])]
    labels = [base_name] * len(sample_names)
    return data, sample_names, labels


def allocate_counts(total_per_label: int) -> Dict[str, int]:
    weights = SPLIT_WEIGHTS
    total_weight = sum(weights.values())
    raw = {k: total_per_label * (w / total_weight) for k, w in weights.items()}
    counts = {k: math.floor(v) for k, v in raw.items()}
    remain = total_per_label - sum(counts.values())
    if remain > 0:
        order = sorted(weights, key=lambda k: raw[k] - counts[k], reverse=True)
        for k in order:
            if remain == 0:
                break
            counts[k] += 1
            remain -= 1
    return counts


def split_indices(indices: List[int], counts: Dict[str, int]):
    pos = 0
    split_map = {}
    for split in SPLIT_WEIGHTS.keys():
        cnt = counts[split]
        split_map[split] = indices[pos:pos + cnt]
        pos += cnt
    return split_map


def main():
    rng = random.Random(RANDOM_SEED)
    datasets = []
    for base in BASE_NAMES:
        meta = load_metadata(base)
        arrays = load_batches(base)
        data, sample_names, labels = convert_to_ctr_format(arrays, meta, base)
        datasets.append({
            "name": base,
            "data": data,
            "sample_names": sample_names,
            "labels": labels,
        })

    per_label_available = min(len(ds["sample_names"]) for ds in datasets)
    if per_label_available == 0:
        raise RuntimeError("存在标签数据量为 0，无法构建 50/50 的数据集")
    print(f"[INFO] 每个标签可用 {per_label_available} 条，将保持各 split 为 50/50。")

    per_label_counts = allocate_counts(per_label_available)
    print(f"[INFO] 每标签的划分: {per_label_counts}")

    split_assignments = {split: {} for split in SPLIT_WEIGHTS.keys()}
    for ds in datasets:
        idxs = list(range(len(ds["sample_names"])))
        rng.shuffle(idxs)
        usable = idxs[:per_label_available]
        splits = split_indices(usable, per_label_counts)
        for split, indices in splits.items():
            split_assignments[split][ds["name"]] = indices

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split in SPLIT_WEIGHTS.keys():
        data_parts = []
        sample_names = []
        labels = []
        for ds in datasets:
            indices = split_assignments[split].get(ds["name"], [])
            if not indices:
                continue
            data_parts.append(ds["data"][indices])
            sample_names.extend([ds["sample_names"][i] for i in indices])
            labels.extend([ds["labels"][i] for i in indices])

        if not data_parts:
            print(f"[WARN] {split} split 为空，跳过输出。")
            continue

        split_data = np.concatenate(data_parts, axis=0)
        out_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(out_dir, exist_ok=True)
        data_path = os.path.join(out_dir, f"{split}_data.npy")
        label_path = os.path.join(out_dir, f"{split}_label.pkl")

        np.save(data_path, split_data)
        with open(label_path, "wb") as f:
            pickle.dump((sample_names, labels), f)

        label_ratio = {name: labels.count(name) for name in set(labels)}
        print(f"[INFO] {split}: 保存 {split_data.shape} -> {data_path}，样本数 {len(sample_names)}, 标签分布 {label_ratio}")

    print("[INFO] 完成 train/val/test 构建，可在 CTR-GCN 中分别指向对应目录。")


if __name__ == "__main__":
    main()
