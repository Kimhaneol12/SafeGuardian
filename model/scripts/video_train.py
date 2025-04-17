import os
import numpy as np
import pickle
import joblib
import time
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

# -------------------------------
# Settings
# -------------------------------
DATA_TYPE = "Training"
SEQUENCE_LENGTH = 600

BASE_NPY_FOLDER = f"./new_mediapipe_{DATA_TYPE}/npy"
CHECKPOINT_PATH = f"./new_mediapipe_{DATA_TYPE}/training_npy_checkpoint.pkl"
MODEL_SAVE_PATH = "../../best_weight/fall_type.pkl"

# -------------------------------
# Load npy files and labels
# -------------------------------
def input_generator_npy(base_folder, sequence_length=600, checkpoint_path=None):
    processed = set()
    sequences, labels = [], []

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as f:
                processed, sequences, labels = pickle.load(f)
            processed = set(processed)
            print(f"Loaded checkpoint: {len(processed)} scenarios processed.")
        except Exception as e:
            print("Error loading checkpoint:", e)

    for i, scenario_folder in enumerate(tqdm(os.listdir(base_folder), desc="Loading Training npy data")):
        if scenario_folder in processed:
            continue
        scenario_path = os.path.join(base_folder, scenario_folder)
        if not os.path.isdir(scenario_path):
            continue

        try:
            npy_files = sorted([f for f in os.listdir(scenario_path) if f.endswith('.npy')],
                               key=lambda x: int(os.path.splitext(x)[0]))
            if len(npy_files) < sequence_length:
                continue
            window = []
            for j in range(sequence_length):
                file_path = os.path.join(scenario_path, npy_files[j])
                try:
                    data = np.load(file_path)
                    window.append(data)
                except:
                    break
            if len(window) == sequence_length:
                sequences.append(window)
                folder_upper = scenario_folder.upper()
                if "FY" in folder_upper:
                    labels.append("FY")
                elif "SY" in folder_upper:
                    labels.append("SY")
                elif "BY" in folder_upper:
                    labels.append("BY")
                elif "N" in folder_upper:
                    labels.append("NonFall")
                else:
                    labels.append("Unknown")
                processed.add(scenario_folder)
        except Exception as e:
            print("Error processing folder", scenario_folder, e)

        if checkpoint_path and (i + 1) % 50 == 0:
            print(f"Saving checkpoint at scenario index {i+1}")
            with open(checkpoint_path, "wb") as f:
                pickle.dump((list(processed), sequences, labels), f)

    if checkpoint_path:
        with open(checkpoint_path, "wb") as f:
            pickle.dump((list(processed), sequences, labels), f)

    return np.array(sequences), labels

# -------------------------------
# Apply SMOTE
# -------------------------------
def apply_SMOTE(X_train, y_train, sequence_length):
    num_sequences, seq_length, features = X_train.shape
    X_reshaped = X_train.reshape(num_sequences * seq_length, features).astype(np.float32)
    y_expanded = np.repeat(y_train, seq_length)

    oversample = SMOTE()
    X_res, y_res = oversample.fit_resample(X_reshaped, y_expanded)
    print("After SMOTE, sample distribution:", Counter(y_res))

    new_num = int(X_res.shape[0] / sequence_length)
    X_smote = X_res.reshape(new_num, sequence_length, features)

    y_res_reshaped = y_res.reshape(new_num, sequence_length)
    y_smote, X_final = [], []
    for i in range(new_num):
        unique_labels = set(y_res_reshaped[i])
        if len(unique_labels) == 1:
            y_smote.append(list(unique_labels)[0])
            X_final.append(X_smote[i])
        else:
            print("Inconsistent labels in sequence", i)
    return np.array(X_final), np.array(y_smote)

# -------------------------------
# Train RandomForest Model
# -------------------------------
if __name__ == '__main__':
    X, labels = input_generator_npy(BASE_NPY_FOLDER, sequence_length=SEQUENCE_LENGTH, checkpoint_path=CHECKPOINT_PATH)
    print("Training data shape:", X.shape)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    print("Classes:", le.classes_)

    X_smote, y_smote = apply_SMOTE(X, y, SEQUENCE_LENGTH)

    num_samples, seq_length, features = X_smote.shape
    X_flat = X_smote.reshape(num_samples, seq_length * features)

    clf = RandomForestClassifier(n_estimators=100, random_state=1337)
    clf.fit(X_flat, y_smote)
    y_pred = clf.predict(X_flat)

    print("Training accuracy: {:.3f}".format(clf.score(X_flat, y_smote)))
    print(classification_report(y_smote, y_pred, target_names=le.classes_))

    joblib.dump(clf, MODEL_SAVE_PATH)
