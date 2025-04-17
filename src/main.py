import os
import cv2
import time
import numpy as np
import joblib
from pathlib import Path
from collections import deque

from deep_sort_tracking_id import run_tracking
import mediapipe as mp

# -------------------------------
# Settings
# -------------------------------
# ▶️ '1.mp4'            → 테스트 영상
# ▶️ '0'                → 내장 웹캠 사용
# ▶️ '1' 또는 '2'       → 외장 웹캠 (USB 연결 카메라 번호)
# ▶️ 'rtsp://...'       → RTSP 기반 IP 카메라 스트림
# ▶️ 'http://...'       → HTTP 영상 스트림
SOURCE = '1.mp4'
WEIGHTS_PATH = 'yolov7-tiny.pt'
CROP_ROOT = './outputs/crops'
KEYPOINT_ROOT = './outputs/keypoints'
FALL_MODEL_PATH = '../model/best_weight/fall_detection.pkl'
TYPE_MODEL_PATH = '../model/best_weight/fall_type.pkl'
WINDOW_SIZE = 600

# -------------------------------
# Step 1: YOLOv7 + DeepSORT Tracking & Cropping
# -------------------------------
run_tracking(
    weights=WEIGHTS_PATH,
    source=SOURCE,
    save_dir=CROP_ROOT,
    conf_thres=0.5,
    iou_thres=0.25,
    view_img=False
)

# -------------------------------
# Fall Predictor Class
# -------------------------------
class FallPredictor:
    def __init__(self, fall_model_path, type_model_path, window_size=600):
        self.fall_model = joblib.load(fall_model_path)
        self.type_model = joblib.load(type_model_path)
        self.buffer = deque(maxlen=window_size)

    def update(self, keypoints):
        self.buffer.append(keypoints)
        if len(self.buffer) < self.buffer.maxlen:
            print(f"[MAIN] 현재 시퀀스 길이: {len(self.buffer)} (예측 대기 중)")
            return None
        return self.predict()

    def predict(self):
        input_array = np.array(self.buffer).reshape(1, -1)
        fall_pred = self.fall_model.predict(input_array)[0]
        if fall_pred == 0:
            return {"fall": 0, "type": "N"}
        type_pred = self.type_model.predict(input_array)[0]
        type_label = {0: "FY", 1: "SY", 2: "BY"}.get(type_pred, "Unknown")
        return {"fall": 1, "type": type_label}

    def reset(self):
        self.buffer.clear()

# -------------------------------
# Extract Keypoints from Results
# -------------------------------
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# -------------------------------
# Step 2: Real-time Monitoring
# -------------------------------
video_id = 'webcam' if SOURCE == '0' else Path(SOURCE).stem
crop_dir = os.path.join(CROP_ROOT, video_id)
predictor = FallPredictor(FALL_MODEL_PATH, TYPE_MODEL_PATH, WINDOW_SIZE)
processed = set()

print("✅ Starting keypoint extraction and fall detection...")
print(f"[DEBUG] crop_dir: {crop_dir}")
print(f"[DEBUG] crop_dir exists: {os.path.exists(crop_dir)}")

with mp.solutions.holistic.Holistic(static_image_mode=True, min_detection_confidence=0.3) as holistic:
    while True:
        try:
            if not os.path.exists(crop_dir):
                print("[WAIT] crop_dir not found yet. Waiting...")
                time.sleep(0.1)
                continue

            for id_folder in os.listdir(crop_dir):
                id_dir = os.path.join(crop_dir, id_folder)
                if not os.path.isdir(id_dir):
                    continue

                frame_files = sorted(os.listdir(id_dir))
                for frame_file in frame_files:
                    frame_path = os.path.join(id_dir, frame_file)
                    if frame_path in processed:
                        continue
                    processed.add(frame_path)

                    print(f"[MAIN] Try extracting: {frame_path}")
                    image = cv2.imread(frame_path)
                    if image is None:
                        continue

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)

                    if results.pose_landmarks is None:
                        print(f"[!] No pose detected (image size: {image.shape})")
                        keypoints = np.zeros(1662)
                    else:
                        keypoints = extract_keypoints(results)
                        print(f"[MAIN] Keypoints: {keypoints.shape}")

                    # 🔸 Save keypoints
                    save_path = os.path.join(KEYPOINT_ROOT, video_id, id_folder)
                    os.makedirs(save_path, exist_ok=True)
                    np.save(os.path.join(save_path, frame_file.replace('.jpg', '.npy')), keypoints)

                    # 🔸 Prediction
                    result = predictor.update(keypoints)
                    if result:
                        print(f"[결과] ID: {id_folder} → 낙상 여부: {result['fall']} / 유형: {result['type']}")
                        predictor.reset()

            time.sleep(0.01)

        except KeyboardInterrupt:
            print("🛑 실시간 감지 종료.")
            break
