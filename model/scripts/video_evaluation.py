import os
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -------------------------------
# Settings
# -------------------------------
DATA_TYPE = "Validation"
SEQUENCE_LENGTH = 600

BASE_NPY_FOLDER = f"./new_mediapipe_{DATA_TYPE}/npy"
FALL_MODEL_PATH = "../../best_weight/fall_detection.pkl"
FALL_TYPE_MODEL_PATH = "../../best_weight/fall_type.pkl"

RESULT_DIR = f"./results"
os.makedirs(RESULT_DIR, exist_ok=True)

TEXT_OUTPUT_PATH = os.path.join(RESULT_DIR, f"evaluation_result_{DATA_TYPE}.txt")
CM_OUTPUT_PATH_BINARY = os.path.join(RESULT_DIR, f"confusion_matrix_binary_{DATA_TYPE}.png")
CM_OUTPUT_PATH_FALLTYPE = os.path.join(RESULT_DIR, f"confusion_matrix_falltype_{DATA_TYPE}.png")

# -------------------------------
# npy data load
# -------------------------------
def input_generator_npy(base_folder, sequence_length=600):
    sequences = []
    labels = []
    for scenario_folder in tqdm(os.listdir(base_folder), desc="Loading Validation npy data"):
        scenario_path = os.path.join(base_folder, scenario_folder)
        if not os.path.isdir(scenario_path):
            continue
        npy_files = sorted([f for f in os.listdir(scenario_path) if f.endswith('.npy')],
                           key=lambda x: int(os.path.splitext(x)[0]))
        if len(npy_files) < sequence_length:
            continue
        window = []
        for i in range(sequence_length):
            file_path = os.path.join(scenario_path, npy_files[i])
            try:
                data = np.load(file_path)
                window.append(data)
            except:
                break
        if len(window) == sequence_length:
            sequences.append(window)
            folder_upper = scenario_folder.upper()
            if "_FY" in folder_upper:
                labels.append("FY")
            elif "_SY" in folder_upper:
                labels.append("SY")
            elif "_BY" in folder_upper:
                labels.append("BY")
            else:
                labels.append("N")
    X = np.array(sequences)
    return X, labels

# -------------------------------
# 평가 결과 저장 함수
# -------------------------------
def save_classification_results(y_true, y_pred, class_names, output_text_path, cm_image_path, title):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    accuracy = accuracy_score(y_true, y_pred)

    with open(output_text_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== {title} =====\n")
        f.write(f"Accuracy: {accuracy:.3f}\n\n")
        f.write(report + "\n")

    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names,
                                                   xticks_rotation=45, cmap='Blues', colorbar=False)
    disp.ax_.set_title(title)
    plt.tight_layout()
    plt.savefig(cm_image_path)
    plt.close()

# -------------------------------
# 실행
# -------------------------------
if __name__ == '__main__':
    X_val, y_true_full = input_generator_npy(BASE_NPY_FOLDER, SEQUENCE_LENGTH)
    print(f"Validation data shape: {X_val.shape}")

    num_samples, seq_length, features = X_val.shape
    X_flat = X_val.reshape(num_samples, seq_length * features)

    # 낙상 여부 판단
    le_fall = LabelEncoder()
    y_binary_true = ['FALL' if l in ['FY', 'SY', 'BY'] else 'NonFall' for l in y_true_full]
    y_binary_encoded = le_fall.fit_transform(y_binary_true)
    print("Binary Classes:", le_fall.classes_)

    clf_fall = joblib.load(FALL_MODEL_PATH)
    y_binary_pred = clf_fall.predict(X_flat)

    save_classification_results(y_binary_encoded, y_binary_pred, le_fall.classes_,
                                TEXT_OUTPUT_PATH, CM_OUTPUT_PATH_BINARY, title="Fall vs NonFall Classification")

    # 낙상인 것만 따로 평가
    fall_indices = [i for i, pred in enumerate(y_binary_pred) if pred == le_fall.transform(['FALL'])[0]]
    X_fall_only = X_flat[fall_indices]
    y_true_falltype = [y_true_full[i] for i in fall_indices if y_true_full[i] in ['FY', 'SY', 'BY']]

    if len(X_fall_only) == 0 or len(y_true_falltype) == 0:
        print("No FALL data to evaluate FALL TYPE classification.")
    else:
        le_type = LabelEncoder()
        y_type_encoded = le_type.fit_transform(y_true_falltype)
        print("Fall Type Classes:", le_type.classes_)

        clf_type = joblib.load(FALL_TYPE_MODEL_PATH)
        y_type_pred = clf_type.predict(X_fall_only)

        save_classification_results(y_type_encoded, y_type_pred, le_type.classes_,
                                    TEXT_OUTPUT_PATH, CM_OUTPUT_PATH_FALLTYPE, title="Fall Type Classification (FY, SY, BY)")
