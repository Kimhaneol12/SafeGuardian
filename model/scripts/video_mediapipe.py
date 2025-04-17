import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# =========================
# Settings
# =========================
DATA_TYPES = ["Training", "Validation"]
FPS = 60
FOURCC = cv2.VideoWriter_fourcc(*'DIVX')
RESIZE_WIDTH = 480
RESIZE_HEIGHT = 270

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                      for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] 
                      for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] 
                    for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] 
                    for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def process_video(scenario_dir, scenario, output_img_dir, output_npy_dir, output_vid_file):
    if os.path.exists(output_npy_dir):
        if any(f.endswith('.npy') for f in os.listdir(output_npy_dir)):
            print(f"Skipping {scenario} (already processed).")
            return

    cap = cv2.VideoCapture(os.path.join(scenario_dir, scenario + ".mp4"))
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_npy_dir, exist_ok=True)
    out = cv2.VideoWriter(output_vid_file, FOURCC, FPS, (RESIZE_WIDTH, RESIZE_HEIGHT))

    with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            black = np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, 3), np.uint8)
            try:
                resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
                cv2.imwrite(os.path.join(output_img_dir, f"{frame_index}.jpg"), resized)
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(black, results)
                keypoints = extract_keypoints(results)
                out.write(black)
                np.save(os.path.join(output_npy_dir, f"{frame_index}.npy"), keypoints)
                frame_index += 1
            except Exception as e:
                print(f"Error processing frame {frame_index} in {scenario}: {e}")
                break
        cap.release()

# =========================
# loop
# =========================
for DATA_TYPE in DATA_TYPES:
    if DATA_TYPE == "Training":
        ORG_PATH = "../../../fall_data/data/data/Training/raw_data/TS/video"
    elif DATA_TYPE == "Validation":
        ORG_PATH = "../../../fall_data/data/data/Validation/raw_data/VS/video"

    OUTPUT_ROOT = f"./new_mediapipe_{DATA_TYPE}"
    IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
    NPY_DIR = os.path.join(OUTPUT_ROOT, "npy")
    VID_DIR = os.path.join(OUTPUT_ROOT, "vid")

    for directory in [OUTPUT_ROOT, IMAGES_DIR, NPY_DIR, VID_DIR]:
        os.makedirs(directory, exist_ok=True)

    fall_root = os.path.join(ORG_PATH, "Y")
    nonfall_root = os.path.join(ORG_PATH, "N", "N")

    # FY, SY, BY
    if os.path.exists(fall_root):
        for fall_type in ['FY', 'SY', 'BY']:
            fall_type_dir = os.path.join(fall_root, fall_type)
            if not os.path.exists(fall_type_dir):
                continue
            for scenario in os.listdir(fall_type_dir):
                scenario_dir = os.path.join(fall_type_dir, scenario)
                video_file = os.path.join(scenario_dir, scenario + ".mp4")
                if not os.path.exists(video_file):
                    print(f"Video file not found: {video_file}")
                    continue
                out_img_dir = os.path.join(IMAGES_DIR, "img_" + scenario)
                out_npy_dir = os.path.join(NPY_DIR, "npy_" + scenario)
                out_vid_file = os.path.join(VID_DIR, "key_" + scenario + ".mp4")
                process_video(scenario_dir, scenario, out_img_dir, out_npy_dir, out_vid_file)

    # NonFall
    if os.path.exists(nonfall_root):
        for scenario in os.listdir(nonfall_root):
            scenario_dir = os.path.join(nonfall_root, scenario)
            video_file = os.path.join(scenario_dir, scenario + ".mp4")
            if not os.path.exists(video_file):
                print(f"Video file not found: {video_file}")
                continue
            out_img_dir = os.path.join(IMAGES_DIR, "img_" + scenario)
            out_npy_dir = os.path.join(NPY_DIR, "npy_" + scenario)
            out_vid_file = os.path.join(VID_DIR, "key_" + scenario + ".mp4")
            process_video(scenario_dir, scenario, out_img_dir, out_npy_dir, out_vid_file)

    print(f"✅ Finished MediaPipe processing for {DATA_TYPE}")

cv2.destroyAllWindows()
for _ in range(4):
    cv2.waitKey(1)
