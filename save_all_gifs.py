import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# ==========================================
# 骨架连接规则
# ==========================================
SKELETON_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),  #躯干
    (11, 13), (13, 15),                     #左臂
    (12, 14), (14, 16),                     #右臂
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),  #左腿
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),  #右腿
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)  #头部
]

BASE_DIR = r"D:\GNN_BAD"
SKELETON_FILE = os.path.join(BASE_DIR, "droit_skeleton", "droit_all.npy")
OUTPUT_DIR = os.path.join(BASE_DIR, "droit_all")

def batch_generate_gifs():
    try:
        if not os.path.exists(SKELETON_FILE):
            print(f"[ERROR] 找不到文件 {SKELETON_FILE}")
            return

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        data = np.load(SKELETON_FILE)
        num_groups, num_frames = data.shape[:2]
        print(f"[INFO] 发现 {num_groups} 组数据，每组 {num_frames} 帧，开始批量生成 GIF...")

        for group_idx in range(num_groups):
            print(f"   正在处理第 {group_idx} 组 ({group_idx + 1}/{num_groups})...")
            frames_data = data[group_idx]

            fig, ax = plt.subplots(figsize=(6, 6))
            all_x = frames_data[:, :, 0]
            all_y = frames_data[:, :, 1]
            margin = 0.05
            ax.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
            ax.set_ylim(np.max(all_y) + margin, np.min(all_y) - margin)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Group {group_idx}")

            lines = []
            for _ in SKELETON_CONNECTIONS:
                line, = ax.plot([], [], 'b-', lw=2)
                lines.append(line)
            points, = ax.plot([], [], 'ro', ms=4)

            def update(frame_idx):
                current_frame = frames_data[frame_idx]
                for line, (start, end) in zip(lines, SKELETON_CONNECTIONS):
                    line.set_data(
                        [current_frame[start, 0], current_frame[end, 0]],
                        [current_frame[start, 1], current_frame[end, 1]],
                    )
                points.set_data(current_frame[:, 0], current_frame[:, 1])
                return lines + [points]

            ani = animation.FuncAnimation(
                fig, update, frames=num_frames, interval=150, blit=True
            )

            output_path = os.path.join(OUTPUT_DIR, f"skeleton_group_{group_idx:03d}.gif")
            ani.save(output_path, writer=PillowWriter(fps=8))
            plt.close(fig)

        print(f"[INFO] 所有 GIF 已生成完毕，位于 {OUTPUT_DIR}")

    except Exception as e:
        print(f"[ERROR] 发生错误: {e}")

if __name__ == "__main__":
    batch_generate_gifs()
