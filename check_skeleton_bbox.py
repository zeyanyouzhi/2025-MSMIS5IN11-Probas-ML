import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = r"D:\GNN_BAD"
NPY_PATH = os.path.join(BASE_DIR, "droit_skeleton", "droit_000.npy")
OUTPUT_DIR = os.path.join(BASE_DIR, "droit_bbox_check")


def compute_bbox_metrics(sequence):
    centers = []
    areas = []
    for joints in sequence:
        valid_mask = ~(np.isclose(joints[:, 0], 0.0) & np.isclose(joints[:, 1], 0.0))
        if not np.any(valid_mask):
            centers.append((np.nan, np.nan))
            areas.append(0.0)
            continue

        x_valid = joints[valid_mask, 0]
        y_valid = joints[valid_mask, 1]
        x_min, x_max = float(np.min(x_valid)), float(np.max(x_valid))
        y_min, y_max = float(np.min(y_valid)), float(np.max(y_valid))

        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        area = max((x_max - x_min) * (y_max - y_min), 0.0)

        centers.append((center_x, center_y))
        areas.append(area)
    return np.array(centers), np.array(areas)


def main():
    if not os.path.exists(NPY_PATH):
        raise FileNotFoundError(f"未找到骨骼文件: {NPY_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = np.load(NPY_PATH)  # (N, T, 33, 2)
    num_samples, num_frames = data.shape[:2]
    print(f"[INFO] 读取 {NPY_PATH}, 样本数 {num_samples}, 每样本 {num_frames} 帧")

    for sample_idx, sequence in enumerate(data):
        centers, areas = compute_bbox_metrics(sequence)

        print(f"\nSample {sample_idx}:")
        for frame_idx in range(num_frames):
            cx, cy = centers[frame_idx]
            area = areas[frame_idx]
            print(f"  frame {frame_idx:02d}: center=({cx:.4f}, {cy:.4f}), area={area:.6f}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(range(num_frames), areas, marker="o")
        axes[0].set_title(f"Sample {sample_idx} bbox area")
        axes[0].set_xlabel("frame")
        axes[0].set_ylabel("area")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(centers[:, 0], centers[:, 1], "-o")
        axes[1].set_title(f"Sample {sample_idx} center path")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"bbox_sample_{sample_idx:03d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] 已输出 {out_path}")


if __name__ == "__main__":
    main()
