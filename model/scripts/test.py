import os
import numpy as np
import cv2
import joblib
from sklearn.metrics import classification_report
import mediapipe as mp

# -------------------------
# Settings
# -------------------------
VIDEO_PATH = "../sample_video.mp4" # 영상 위치에 맞게 설정
FALL_MODEL_PATH = "../model/best_weight/fall_detection.pkl"
FALL_TYPE_MODEL_PATH = "../model/best_weight/fall_type.pkl"
RESULT_SAVE_DIR = "../model/eval_results/manual_test"
RESULT_TEXT_FILE = os.path.join(RESULT_SAVE_DIR, "manual_test_result.txt")

SEQUENCE_LENGTH = 600 

# -------------------------
# 결과 디렉토리 생성
# -------------------------
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

# -------------------------
# MediaPipe 초기화
# -------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# -------------------------
# 영상 처리
# -------------------------
print(f"▶️ 영상 처리 시작: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []

while len(frames) < SEQUENCE_LENGTH:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (480, 270))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        pose_data = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
        frames.append(np.array(pose_data).flatten())

cap.release()
pose.close()

if len(frames) < SEQUENCE_LENGTH:
    print(f"❌ 프레임 부족: {len(frames)}프레임 (최소 {SEQUENCE_LENGTH} 필요)")
    exit()

X = np.array(frames).reshape(1, SEQUENCE_LENGTH * len(frames[0]))

# -------------------------
# 낙상 여부 예측
# -------------------------
clf_fall = joblib.load(FALL_MODEL_PATH)
y_pred_fall = clf_fall.predict(X)[0]
fall_label = "FALL" if y_pred_fall == 1 else "NonFall"

print(f"\n📊 낙상 여부 판단 결과: {fall_label}")

# -------------------------
# 낙상 유형 예측 (FALL일 경우)
# -------------------------
fall_type_label = "N/A"
if fall_label == "FALL":
    clf_type = joblib.load(FALL_TYPE_MODEL_PATH)
    y_pred_type = clf_type.predict(X)[0]
    fall_type_label = ["BY", "FY", "SY"][y_pred_type]  # 클래스 순서는 학습 모델에 따라 다를 수 있음

# -------------------------
# 결과 저장
# -------------------------
with open(RESULT_TEXT_FILE, "w", encoding="utf-8") as f:
    f.write(f"[Result for video: {VIDEO_PATH}]\n")
    f.write(f"Fall Detection: {fall_label}\n")
    f.write(f"Fall Type: {fall_type_label}\n")

print("\n✅ 결과 저장 완료:", RESULT_TEXT_FILE)
