import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ====== 你要改的三行 ======
BASE_DIR = r"D:\GNN_BAD"
BASE_NAME = "droit"
VIDEO_PATH = os.path.join(BASE_DIR, f"{BASE_NAME}.mp4")
CSV_PATH = os.path.join(BASE_DIR, f"{BASE_NAME}.csv")  # 第一列是帧号
OUTPUT_DIR = os.path.join(BASE_DIR, f"{BASE_NAME}_skeleton")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{BASE_NAME}_all.npy")
WINDOW = 35    # 固定 35 帧 (包含 t0)
YOLO_WEIGHT = os.path.join(BASE_DIR, "yolov8n.pt")
DETECTION_CONF = 0.35
ROI_MARGIN = 0.12
ROI_MIN_SIZE = 64
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) 读 CSV
df = pd.read_csv(CSV_PATH, encoding="utf-8", engine="python")
df.columns = df.columns.str.strip().str.replace("\ufeff", "")
impact_frames = df.iloc[:, 0].astype(int).tolist()
print("击球帧列表", impact_frames)

# 2) 视频信息
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("[ERROR] 无法打开视频，请检查 VIDEO_PATH 是否正确")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"总帧数 {total_frames}, FPS: {fps}, 分辨率 {w}x{h}")

# 3) 初始化 Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)


def load_yolo_detector():
    if YOLO is None:
        print("[WARN] 未安装 ultralytics，无法启用 YOLO 锁人，将回退到整帧 Pose。")
        return None
    weight_path = YOLO_WEIGHT if os.path.exists(YOLO_WEIGHT) else "yolov8n.pt"
    try:
        model = YOLO(weight_path)
        print(f"[INFO] 已加载 YOLO 模型: {weight_path}")
        return model
    except Exception as err:
        print(f"[WARN] YOLO 模型加载失败 ({err})，将回退到整帧 Pose。")
        return None


def select_person_roi(frame, detector):
    frame_h, frame_w = frame.shape[:2]
    if detector is None:
        return frame, (0, 0, frame_w, frame_h)

    results = detector(frame, conf=DETECTION_CONF, classes=[0], verbose=False)
    best_box = None
    best_area = 0.0
    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls_id in zip(xyxy, confs, classes):
            if cls_id != 0 or conf < DETECTION_CONF:
                continue
            x1, y1, x2, y2 = box
            x1 = max(0.0, min(float(x1), frame_w))
            x2 = max(0.0, min(float(x2), frame_w))
            y1 = max(0.0, min(float(y1), frame_h))
            y2 = max(0.0, min(float(y2), frame_h))
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)

    if best_box is None:
        return frame, (0, 0, frame_w, frame_h)

    x1, y1, x2, y2 = best_box
    bw = x2 - x1
    bh = y2 - y1
    if bw < ROI_MIN_SIZE or bh < ROI_MIN_SIZE:
        return frame, (0, 0, frame_w, frame_h)

    margin_x = bw * ROI_MARGIN
    margin_y = bh * ROI_MARGIN
    x1 = max(0, int(round(x1 - margin_x)))
    y1 = max(0, int(round(y1 - margin_y)))
    x2 = min(frame_w, int(round(x2 + margin_x)))
    y2 = min(frame_h, int(round(y2 + margin_y)))
    if x2 <= x1 or y2 <= y1:
        return frame, (0, 0, frame_w, frame_h)

    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return frame, (0, 0, frame_w, frame_h)
    return cropped, (x1, y1, x2, y2)


def pose_from_frame(frame, detector, frame_w, frame_h):
    roi_img, (x1, y1, x2, y2) = select_person_roi(frame, detector)
    rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if not result.pose_landmarks:
        return np.zeros((33, 2), dtype=np.float32)

    roi_w = max(x2 - x1, 1)
    roi_h = max(y2 - y1, 1)
    coords = []
    for p in result.pose_landmarks.landmark:
        abs_x = p.x * roi_w + x1
        abs_y = p.y * roi_h + y1
        coords.append([abs_x / frame_w, abs_y / frame_h])
    return np.array(coords, dtype=np.float32)


yolo_detector = load_yolo_detector()

# 4) 预计算所有需要的帧
frame_specs = []  # (start, end, t0)
needed_frames = set()
for idx, t0 in enumerate(impact_frames):
    if t0 < 0 or t0 >= total_frames:
        print(f"[WARN] 跳过非法帧号 {t0}")
        continue
    start = max(0, t0 - WINDOW + 1)
    end = t0
    frame_specs.append((start, end, t0))
    needed_frames.update(range(start, end + 1))

if not frame_specs:
    pose.close()
    cap.release()
    raise RuntimeError("[ERROR] CSV 中没有可用的帧号")

needed_list = sorted(needed_frames)
needed_set = set(needed_list)
max_needed = needed_list[-1]
print(f"[INFO] 需要解析 {len(needed_list)} 帧 (最大帧号 {max_needed})")

frame_joints = {}
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
current_idx = 0
missing_read = False
while current_idx <= max_needed:
    ret, frame = cap.read()
    if not ret:
        print(f"[WARN] 在帧 {current_idx} 读取失败，后续帧将补 0")
        missing_read = True
        break
    if current_idx in needed_set:
        frame_joints[current_idx] = pose_from_frame(frame, yolo_detector, w, h)
    current_idx += 1

if missing_read:
    missing = [f for f in needed_list if f not in frame_joints]
    print(f"[WARN] 共缺失 {len(missing)} 帧，将使用 0 填充")

def fetch_frame_joints(frame_idx):
    joints = frame_joints.get(frame_idx)
    if joints is not None:
        return joints
    return np.zeros((33, 2), dtype=np.float32)

samples = []
valid_t0 = []
for start, end, t0 in frame_specs:
    seq = [fetch_frame_joints(f) for f in range(start, end + 1)]
    seq = np.stack(seq, axis=0)
    if seq.shape[0] < WINDOW:
        pad = np.repeat(seq[:1], WINDOW - seq.shape[0], axis=0)
        seq = np.concatenate([pad, seq], axis=0)
    samples.append(seq)
    valid_t0.append(t0)
    print(f"[OK] t0={t0}, 序列长度 {seq.shape[0]} 帧")

pose.close()
cap.release()

if not samples:
    raise RuntimeError("[ERROR] 没有成功生成任何样本")

X = np.stack(samples, axis=0)
np.save(OUTPUT_FILE, X)
print(f"[INFO] 已保存 {X.shape} 到 {OUTPUT_FILE}")
print("对应的击球帧列表:", valid_t0)
