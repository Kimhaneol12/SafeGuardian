from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dense, LSTM, Dropout, Conv1D, BatchNormalization, ReLU, Bidirectional
from tensorflow.keras import Sequential
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.keras.utils import to_categorical
from tensorflow import keras 
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from collections import Counter
import tensorflow as tf
import numpy as np
import pandas as pd
import copy, time, os, sys, joblib
from collections import Counter
import matplotlib.pyplot as plt
import datatable as dt
from tqdm import tqdm

from ahrs.filters import Tilt
from scipy.signal import find_peaks, welch
import scipy

def find_best_model(scores): #input => list of mean accuracy and std of models
    hi_acc_idx = np.where(scores == np.max(scores[:, 0], 0))[0]
    lo_std_idx = np.where(scores[hi_acc_idx, 1] == np.min(scores[hi_acc_idx, 1]))[0]
    indices = scores == scores[hi_acc_idx][lo_std_idx][0]
    best_model_idx = np.where(indices[:, 0] & indices[:, 1])[0][0]
    return best_model_idx

def load_data(fname, prefix=""):
    #df = pd.read_excel(os.path.join(prefix, fname))
    df = dt.fread(os.path.join(prefix, fname)).to_pandas()
    return df
    
def get_filenames(path, prefix=""):
    fnames = [os.path.join(os.path.join(prefix, path), f) for f in os.listdir(os.path.join(prefix, path)) if "DS" not in f and "ipynb_checkpoints" not in f and "." in f]
    print(f'Numer of available files: {len(fnames)}', flush=True)
    return fnames

def load_total_datasets(date_folder):

    ds_path = os.path.join('/Volumes/Developments/SynologyDrive/Datasets/NIA_Project/IMUs', 'dataset by ' + date_folder)
    fall_ds_path = "Y"; non_fall_ds_path = "N"
    backward_fall_ds_path = "BY"; forward_fall_ds_path = "FY"; lateral_fall_ds_path = "SY"
    
    nf_ds_path = os.path.join(ds_path, non_fall_ds_path)
    bf_ds_path = os.path.join(os.path.join(ds_path, fall_ds_path), backward_fall_ds_path)
    ff_ds_path = os.path.join(os.path.join(ds_path, fall_ds_path), forward_fall_ds_path)
    lf_ds_path = os.path.join(os.path.join(ds_path, fall_ds_path), lateral_fall_ds_path)
    column_names = pd.read_excel(os.path.join(os.path.join(ds_path, non_fall_ds_path), os.listdir(os.path.join(ds_path, non_fall_ds_path))[0])).columns.values
    
    nf_ds_fnames = get_filenames(os.path.join(ds_path, nf_ds_path))
    bf_ds_fnames = get_filenames(os.path.join(os.path.join(ds_path, fall_ds_path), backward_fall_ds_path))
    #bf_ds_fnames = [bf_ds_fname for bf_ds_fname in bf_ds_fnames if "00907" not in bf_ds_fname]
    ff_ds_fnames = get_filenames(os.path.join(os.path.join(ds_path, fall_ds_path), forward_fall_ds_path))
    lf_ds_fnames = get_filenames(os.path.join(os.path.join(ds_path, fall_ds_path), lateral_fall_ds_path))
    
    fnames = []
    fnames.extend(nf_ds_fnames)
    fnames.extend(bf_ds_fnames)
    fnames.extend(ff_ds_fnames)
    fnames.extend(lf_ds_fnames)
    fnames = [fname for fname in fnames if len(fname.split("_")[0]) <= 5]
    fnames = np.array(fnames)

    nf_datasets = load_datasets(nf_ds_fnames, nf_ds_path)
    bf_datasets = load_datasets(bf_ds_fnames, bf_ds_path)
    ff_datasets = load_datasets(ff_ds_fnames, ff_ds_path)
    lf_datasets = load_datasets(lf_ds_fnames, lf_ds_path)
    datasets_fall_nonfall = np.vstack((nf_datasets, bf_datasets, ff_datasets, lf_datasets))
    datasets_fall_direction_only = np.vstack((bf_datasets, ff_datasets, lf_datasets))
    datasets_fall_direction = np.vstack((nf_datasets, bf_datasets, ff_datasets, lf_datasets))
    datasets = [datasets_fall_nonfall, datasets_fall_direction_only, datasets_fall_direction]

    from collections import Counter
    nf_labels = ['non-fall' for i in range(len(nf_datasets))]#np.zeros(len(nf_datasets)).astype(int)
    bf_labels = ['bwd-fall' for i in range(len(bf_datasets))]#np.ones(len(bf_datasets)).astype(int) * 1
    ff_labels = ['fwd-fall' for i in range(len(ff_datasets))]#np.ones(len(ff_datasets)).astype(int) * 0
    lf_labels = ['lat-fall' for i in range(len(lf_datasets))]#np.ones(len(lf_datasets)).astype(int) * 2
    label_fall_direction_only = np.hstack((bf_labels, ff_labels, lf_labels))
    label_fall_nonfall = encode_fall_nonfall(np.hstack((nf_labels, bf_labels, ff_labels, lf_labels)))
    label_fall_direction = np.hstack((nf_labels, bf_labels, ff_labels, lf_labels))
    labels = [label_fall_nonfall, label_fall_direction_only, label_fall_direction]
    
    ord_label_fall_nonfall = encode_ordinal_label(label_fall_nonfall)
    ord_label_fall_direction_only = encode_ordinal_label(label_fall_direction_only)
    ord_label_fall_direction = encode_ordinal_label(label_fall_direction)
    ord_labels = [ord_label_fall_nonfall, ord_label_fall_direction_only, ord_label_fall_direction]
    
    return datasets, labels, ord_labels, fnames

def get_labels_for_n_classes(fnames, n_classes=4):
    non_fall = "N"; fwd_fall = "FY"; bwd_fall = "BY"; lat_fall = "SY"; fall = "Y"

    label_n_classes = np.empty(0)
    if n_classes == 2:
        for fname in fnames:
            fname = fname.split("\\")[-1]
            if non_fall in fname:
                label_n_classes = np.hstack((label_n_classes, "Fall"))
            else:
                label_n_classes = np.hstack((label_n_classes, "Non Fall"))
    elif n_classes == 3:
        for fname in fnames:
            if fwd_fall in fname:
                label_n_classes = np.hstack((label_n_classes, "FY"))
            elif bwd_fall in fname:
                label_n_classes = np.hstack((label_n_classes, "BY"))
            else:
                label_n_classes = np.hstack((label_n_classes, "SY"))
    else:
        pass
    return label_n_classes
        

def get_index_fall_only(fnames):
    non_fall = "N"; fwd_fall = "FY"; bwd_fall = "BY"; lat_fall = "SY"; fall = "Y"
    index = np.empty(0).astype(bool)
    for fname in fnames:
        fname = fname.split("\\")[-1]
        if non_fall in fname:
            index = np.hstack((index, False))
        else:
            index = np.hstack((index, True))
    return index

def get_label(fname):
    non_fall = "N"; forward_fall = "FY"; backward_fall = "BY"; lateral_fall = "SY"
    label = fname.split("\\")[-1]
    if non_fall in label:
        label = 'non-fall'
    elif forward_fall in label:
        label = 'fwd-fall'
    elif backward_fall in label:
        label = 'bwd-fall'
    else:
        label = 'lat-fall'
    return label

# load dataset
def load_datasets(fnames, prefix=""): # return dataset and labels
    n_axis = 3 * 3 * 12 # number of sensor axes
    length = 600 # length of array
    n_error = 0
    labels = np.empty(0)
    datasets = np.empty((0, length, n_axis))
    new_fnames = np.empty(0)
    for i, fname in enumerate(fnames):
        df = load_data(fname, prefix)
        data = df.values[:, 1:]

        # reshape array --> records X n_timesteps X axis 
        try:
            if len(datasets) == 0:
                columns = df.columns.values
                datasets = data.reshape(-1, length, n_axis)
            else:
                datasets = np.vstack((datasets, data.reshape(-1, length, n_axis)))
            labels = np.hstack((labels, get_label(fname)))
            new_fnames = np.hstack((new_fnames, fname)) 
        except:
            try: # dealing with data that contain more than 600 rows
                if data.shape[-1] == n_axis and data.shape[-2] > length:
                    data = data[-length:, :]
                    datasets = np.vstack((datasets, data.reshape(-1, length, n_axis)))
                    print(f'Recoverd file: {fname}')
                    labels = np.hstack((labels, get_label(fname)))
                    new_fnames = np.hstack((new_fnames, fname))
                else:
                    n_error += 1
                    print(f'Error occured for file name: {fname}, index: {i}, shape: {data.shape}')
                
            except: # throw error when number of axes of sensor lower than 108
                n_error += 1
                print(f'Error occured for file name: {fname}, index: {i}, shape: {data.shape}')
        print(f'Loading dataset: {i+1}/{len(fnames)}', end='\r', flush=True)
    print(f'Total number of {datasets.shape[0]} datasets loaded... Error occured for {n_error} files....', flush=True)
    
    return datasets, np.array(labels), new_fnames

def encode_fall_nonfall(label, ordEnc=True):
    new_label = []
    nf = 'non-fall'; f = 'fall'
    bf = 'bwd-fall'; ff = 'fwd-fall'; lf = 'lat-fall'
    for i, item in enumerate(label):
        if item == nf:
            new_label.append(nf)
        else:
            new_label.append(f)
    return np.array(new_label)

def encode_ordinal_label(label):
    new_label = []
    num_classes = len(np.unique(label))
    ord_nf = 0; ord_f = 1    
    ord_bf = 0; ord_ff = 1; ord_lf = 2
    nf = 'non-fall'; f = 'fall'
    ff = 'fwd-fall'; bf = 'bwd-fall'; lf = 'lat-fall'
    
    if num_classes == 2:
        
        for i, item in enumerate(label):
            
            if item == nf:
                new_label.append(ord_nf)
            else:
                new_label.append(ord_f)
        
    elif num_classes == 3:        
        
        for i, item in enumerate(label):
            if item == bf:
                new_label.append(ord_bf)
            elif item == ff:
                new_label.append(ord_ff)
            elif item == lf:
                new_label.append(ord_lf)
    else:
        ord_nf = 0; ord_bf += 1; ord_ff += 1; ord_lf += 1
        
        for i, item in enumerate(label):
            if item == nf:
                new_label.append(ord_nf)
            elif item == bf:
                new_label.append(ord_bf)
            elif item == ff:
                new_label.append(ord_ff)
            elif item == lf:
                new_label.append(ord_lf)
    return np.array(new_label)
    
def ensemble(models, X_test, y_test):
    if len(y_test.shape) > 1:
        preds = np.empty((0, len(np.unique(np.argmax(y_test, 1)))))
    else:    
        preds = np.empty((0, len(np.unique(y_test))))
    
    for i, model in enumerate(models):     #
        if model.__class__.__name__ != 'Functional' and model.__class__.__name__ != 'Sequential':
            if len(X_test.shape) == 2: # for feature sets derived from IMU sensors rows X columns
                pred = model.predict_proba(X_test)          
                
            else:
                pass
        
        else: # for deep learning
            if model.name == "LSTM-IMU" or model.name == "CNN-IMU":
                pred = model.predict(X_test)    

            else: # deep learning for images
                pass            
        
        preds = np.vstack((preds, pred))
    
    if len(X_test.shape) > 2:
        preds = preds.reshape(preds.shape[0]//X_test.shape[0], -1, len(np.unique(np.argmax(y_test, 1))))
    else:
        preds = preds.reshape(preds.shape[0]//X_test.shape[0], -1, len(np.unique(y_test)))
    avg_preds = np.mean(preds, 0)
    return preds, avg_preds
    
def print_accuracy(list_models, test_feature_set, test_label, isDL):

    if isDL:
        predictions = np.empty((0, len(np.unique(np.argmax(test_label,1)))))
    else:
        predictions = np.empty((0, len(np.unique(test_label))))
    for i, models in enumerate(list_models):
        for model in models:
            if isDL:
                prediction = model.predict(test_feature_set)
            else:
                prediction = model.predict_proba(test_feature_set)
            predictions = np.vstack((predictions, prediction))

    if isDL:
        predictions = predictions.reshape(
            predictions.shape[0]//test_feature_set.shape[0], -1, len(np.unique(np.argmax(test_label, 1))))
    else:
        predictions = predictions.reshape(
            predictions.shape[0]//test_feature_set.shape[0], -1, len(np.unique(test_label)))
        
    avg_predictions = np.mean(predictions, 0)

    if isDL:
        print(confusion_matrix(np.argmax(test_label, 1), np.argmax(avg_predictions, 1)))
        print(f'-- Ensemble accuracy: {accuracy_score(np.argmax(test_label, 1), np.argmax(avg_predictions, 1))*100:.2f} %')
    else:
        print(confusion_matrix(test_label, np.argmax(avg_predictions, 1)))
        print(f'-- Ensemble accuracy: {accuracy_score(test_label, np.argmax(avg_predictions, 1))*100:.2f} %')

    return predictions, avg_predictions
    
def time2vec_feature_set(t2v, arr):
    print("Time2Vec Conversion in progress...\t\t\t\t\t\t\t", end='\r', flush=True)
    for i, item in enumerate(arr):
        t2v_item = t2v(item)
        t2v_item = tf.reshape(t2v_item, [1, t2v_item.shape[0], t2v_item.shape[1]])
        if i == 0:
            t2v_feature_set = t2v_item
        else:
            t2v_feature_set = np.vstack([t2v_feature_set, t2v_item])
            
        print(f'{i}/{len(arr)} completed\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t', end='\r', flush=True)
    print("Time2Vec Conversion completed...\t\t\t\t\t\t\t\t\t", end='\r', flush=True)
    return t2v_feature_set

def run_experiments_lstm(trainX, trainY, valX, valY, isDatasetBalanced, repeats=10):
    # load data
    start = time.time()

    # balancing dataset
    #print(f'Before trainX.shape:{trainX.shape}, trainy.shape:{trainy.shape}, valy.shape:{valy.shape}, testy.shape:{testy.shape}, train: {Counter(trainy)}, val: {Counter(valy)} test: {Counter(testy)}')
    np_config.enable_numpy_behavior()
    sm = SMOTE(random_state=0)
    org_trainX = trainX.shape
    org_trainY = trainY.shape
    trainX = trainX.reshape([org_trainX[0], -1])
    print(f'Before SMOTE - {Counter(trainY)}')
    if isDatasetBalanced:
        trainX, trainY = sm.fit_resample(trainX, trainY)
    print(f'After SMOTE - {Counter(trainY)}')
    trainX = trainX.reshape([-1, org_trainX[1], org_trainX[2]])
    #print(f'After trainX.shape:{trainX.shape}, trainy.shape:{trainy.shape}, valy.shape:{valy.shape}, testy.shape:{testy.shape}, train: {Counter(trainy)}, val: {Counter(valy)} test: {Counter(testy)}')

    # labellig
    trainY =  to_categorical(trainY)
    valY = to_categorical(valY)

    trainX = tf.convert_to_tensor(trainX)
    trainY = tf.convert_to_tensor(trainY)
    valX = tf.convert_to_tensor(valX)
    valY = tf.convert_to_tensor(valY)
    #print(f'trainX.shape:{trainX.shape}, valX.shape:{valX.shape}, testX.shape:{testX.shape}')
    # repeat experiment
    scores = list()
    models = list()
    print('Training LSTM Model Started...')
    
    for r in range(repeats):
        t1 = time.time()
        model, score = evaluate_lstm_model(trainX, trainY, valX, valY)
        t2 = time.time()
        score = score * 100.0
        print(f'Accuracy -> #{r+1}: {score:.2f} %, Time Taken for Training a Model: {t2-t1:.2f} s')
        scores.append(score)
        models.append(model)
        # summarize results
    
    summarize_results(scores)
    finish = time.time()
    print(f'Training an LSTM Model Completed --> Training Time for Repeat - {repeats} Times: {finish-start:.2f} s')
    return models, scores
    
def classify(model_name, model, feature_set, label, trainDSIdx, isScaled, isDL, random_seed):
    X, y = feature_set[trainDSIdx], label[trainDSIdx]
    tempIdx = np.arange(0, len(X))
    trainIdx, valIdx = train_test_split(tempIdx, test_size=val_size, random_state=random_seed)
    X_train, y_train, X_val, y_val = X[trainIdx], y[trainIdx], X[valIdx], y[valIdx]
    
    if isDL:
        model = run_experiment_lstm(X_train, y_train, X_val, y_val, isDatasetBalanced)
        
    else:
        if isScaled:
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
        model = perform_classification(model_name, model, X_train, y_train, X_val, y_val, isDatasetBalanced, random_seed, val_size)        
    return model


# perform classification using a ginven model, featureset and label
def perform_classification(model_name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)    
    return model



def perform_classifications(model, X_train, y_train, X_val, y_val, test_size=0.2, repeat=10):
    
    scores = list()
    models = list()
    
    for i in range(repeat):        
        model.fit(X_train, y_train)
        scores.append(model.score(X_val, y_val))
        models.append(copy.deepcopy(model))

    best_model_index = np.where(np.max(scores) == scores)[0][0]
    best_model = models[best_model_index]
    best_score = best_model.score(X_val, y_val)
    return models, scores, best_model, best_score

def init_lstm(n_timesteps, n_features, n_outputs):
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model
    
# summarize scores
def summarize_results(scores):
    #print(scores)
    m, s = np.mean(scores), np.std(scores)
    print(f'Accuracy: {m:.2f} % (+/-{s:.2f})')
    
# fit and evaluate a model
def init_lstm_model(n_timesteps, n_features, n_classes):
    model = Sequential(name='LSTM-IMU')
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    #model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    #model.add(Bidirectional(LSTM(50)))
    model.add(Dense(n_classes, activation='softmax'))
    return model

def init_cnn_model(n_timesteps, n_features, n_classes):
    input_shape = (n_timesteps, n_features)
    input_layer = keras.layers.Input(input_shape)
    conv1 = keras.layers.Conv1D(filters=60, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=60, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=60, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(n_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer, name='CNN-IMU')

def find_top_k_models(scores, k):
    indices = []
    sorted_scores = sorted(scores, reverse=True)
    for index in range(0, k):
        indices.append(np.where(scores == sorted_scores[index])[0][0])
    return indices
    
def evaluate_model(model, epochs, verbose, batch_size, feature_type, X, y):
    #tf.config.set_visible_devices([], 'GPU')
    #n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    import datetime
    today = datetime.datetime.now().strftime("%Y%m%d")
    model_name = model.name
    file_name = "best_" + model_name + "_model_" + feature_type + "_" + str(len(np.unique(np.argmax(y, 1)))) + "_classes_" + today + ".h5"
    
    
    callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join("./models/dl_models", file_name), save_best_only=True, monitor="val_loss"),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]

    '''
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))'''
    #model = create_lstm_model()
    keys = sorted(Counter(np.argmax(y, 1)).keys())
    weights = [Counter(np.argmax(y, 1))[key] for key in keys]
    weights = 1. / np.array(weights)
    #weights = np.sum(weights)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], loss_weights=weights)
    # fit network
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks, validation_split=0.3, verbose=verbose, shuffle=False)
    return model, history


def get_zcr_counts(zcr):
    zcr_counts = np.where(zcr)[0].size

    return zcr_counts
    
def get_peaks_valleys(f, cur_psd):
    # for finding peaks
    peaks, _ = find_peaks(cur_psd)
    num_peaks = len(peaks)
    try:
        peak_freq = f[np.where(np.max(cur_psd[peaks]) == cur_psd)[0]][0]
    except:
        peak_freq = 0

    # for finding valleys
    cur_psd = -cur_psd
    valleys, _ = find_peaks(cur_psd)
    num_valleys = len(valleys)
    try:
        valley_freq = f[np.where(np.max(cur_psd[valleys]) == cur_psd)[0]][0]
    except:
        valley_freq = 0
    # print("peak_freq:{}, valley_freq:{}".format(peak_freq, valley_freq))
    # print("type(num_peaks):{}, type(peak_freq):{}, type(num_valleys):{}, type(valley_freq):{}".format(type(num_peaks), type(peak_freq), type(num_valleys), type(valley_freq)))
    return num_peaks, peak_freq, num_valleys, valley_freq

def calc_dominant_freq(f, psd):
    if 2 < len(psd.shape):
        axis = 1
    elif 1 < len(psd.shape) <= 2:
        axis = 0
    else:
        axis = 0
        psd = psd.reshape(-1)
    max_mag = np.argmax(psd, axis=axis)
    dominant_freq = f[max_mag]
    return dominant_freq


def extract_freq_domain_features(segments, fs):
    if 2 < len(segments.shape):
        axis = 1
    elif 1 < len(segments.shape) <= 2:
        axis = 0
    else:
        axis = 0
        segments = segments.reshape(-1)

    f, pxx_spec = welch(segments, axis=axis, fs=fs)
    # psd below 5hz
    ind_f5hz = f < 5
    f = f[ind_f5hz]

    if axis == 1:
        pxx_spec = pxx_spec[:, ind_f5hz, :]
    else:
        pxx_spec = pxx_spec[ind_f5hz]

    # stats features from freq domain
    mean_freq_mag = np.mean(pxx_spec, axis=axis)
    # median_freq = np.median(pxx_spec, axis=1)
    median_freq_mag = np.sum(pxx_spec, axis=axis) / 2
    integ_psd_mag = np.trapz(pxx_spec, x=f, axis=axis)
    entropy = scipy.stats.entropy(pxx_spec, axis=axis)
    dominant_freq = calc_dominant_freq(f, pxx_spec)
    ##########
    row_peak_valley_features = np.empty(0)

    if 2 < len(pxx_spec.shape):
        rows = np.arange(0, pxx_spec.shape[0])
        axes = np.arange(0, pxx_spec.shape[-1])

        for i, row in enumerate(rows):
            axis_peak_valley_features = np.empty(0)

            for axis in axes:
                cur_psd = pxx_spec[row, :, axis]
                num_peaks, peak_freq, num_valleys, valley_freq = get_peaks_valleys(f, cur_psd)
                energy = calc_freq_energy(cur_psd)
                spectral_centroid = calc_spectral_centroid(f, cur_psd)
                features = [num_peaks, peak_freq, num_valleys, valley_freq, energy, spectral_centroid]

                if len(axis_peak_valley_features) == 0:
                    axis_peak_valley_features = features
                else:
                    axis_peak_valley_features = np.hstack((axis_peak_valley_features, features))

            if len(row_peak_valley_features) == 0:
                row_peak_valley_features = axis_peak_valley_features
            else:
                row_peak_valley_features = np.vstack((row_peak_valley_features, axis_peak_valley_features))

            progressBar(i, len(rows), 20, "PSD Analysis")

    elif 1 < len(pxx_spec.shape) <= 2:
        axes = np.arange(0, pxx_spec.shape[-1])

        if pxx_spec.shape[-1] < 3:  # for 2d array with 1 column. ex) svm
            cur_psd = pxx_spec.reshape(-1)
            num_peaks, peak_freq, num_valleys, valley_freq = get_peaks_valleys(f, cur_psd)
            energy = calc_freq_energy(cur_psd)
            spectral_centroid = calc_spectral_centroid(f, cur_psd)
            features = [num_peaks, peak_freq, num_valleys, valley_freq, energy, spectral_centroid]
            row_peak_valley_features = features

        else:  # for 2d array with more than a column. ex) acc --> (2423, 3)
            for axis in axes:
                cur_psd = pxx_spec[:, axis]
                num_peaks, peak_freq, num_valleys, valley_freq = get_peaks_valleys(f, cur_psd)
                energy = calc_freq_energy(cur_psd)
                spectral_centroid = calc_spectral_centroid(f, cur_psd)
                axis_peak_valley_features = [num_peaks, peak_freq, num_valleys, valley_freq, energy, spectral_centroid]

                if len(row_peak_valley_features) == 0:
                    row_peak_valley_features = axis_peak_valley_features
                else:
                    row_peak_valley_features = np.hstack((row_peak_valley_features, axis_peak_valley_features))

    else:
        num_peaks, peak_freqs, num_valleys, valley_freqs = get_peaks_valleys(f, pxx_spec)
        energy = calc_freq_energy(pxx_spec)
        spectral_centroid = calc_spectral_centroid(f, pxx_spec)
        row_peak_valley_features = [num_peaks, peak_freqs, num_valleys, valley_freqs, energy, spectral_centroid]

    # print("mean_freq_mag.shape:{}, median_freq_mag.shape:{}, integ_psd_mag.shape:{}, row_peak_valley_features.shape:{}".format(mean_freq_mag.shape, median_freq_mag.shape, integ_psd_mag.shape, row_peak_valley_features.shape))
    row_peak_valley_features = np.hstack(
        (row_peak_valley_features, mean_freq_mag, median_freq_mag, integ_psd_mag, entropy, dominant_freq))

    return row_peak_valley_features


def zero_crossing_rate(val):
    import warnings
    warnings.filterwarnings("ignore")

    if 2 < len(val.shape):  # for 3 d array
        rows = np.arange(0, val.shape[0])
        axes = np.arange(0, val.shape[-1])
        zcr_counts_rows = np.empty(0)

        if val.shape[-1] == 1:  # check if the values are all positive values
            if not len(np.where(np.diff(np.sign(val)))[0]):
                val = scipy.stats.zscore(val)

        for row in rows:
            d = val[row]

            zcrs_axes = np.diff(np.sign(d), axis=0)

            zcr_counts_axes = np.empty(0)

            for axis in axes:
                zcr_counts_axis = get_zcr_counts(zcrs_axes[:, axis])

                if not zcr_counts_axes.size:
                    zcr_counts_axes = np.array(zcr_counts_axis)
                else:
                    zcr_counts_axes = np.append(zcr_counts_axes, zcr_counts_axis)

            if not zcr_counts_rows.size:
                zcr_counts_rows = zcr_counts_axes
            else:
                zcr_counts_rows = np.vstack((zcr_counts_rows, zcr_counts_axes))

    elif 1 < len(val.shape) <= 2:  # for 2d array
        axes = np.arange(0, val.shape[-1])
        zcr_counts_rows = np.empty(0)

        if len(axes) < 3:  # for 2d array with 1 column. ex) svm
            val = val.reshape(-1)
            if not np.where(np.diff(np.sign(val)))[0].size:  # if svm, transformed to zscore as svm has no negative valeus
                val = scipy.stats.zscore(val)
                zcrs = np.diff(np.sign(val))
                zcr_counts_rows = np.array([get_zcr_counts(zcrs)])
            else:
                zcrs = np.diff(np.sign(val))
                zcr_counts_rows = np.array([get_zcr_counts(zcrs)])

        else:  # for 2d array with more than a column. ex) acc --> (2423, 3)
            d = val
            zcrs_rows = np.diff(np.sign(d), axis=0)

            for axis in axes:
                zcr_counts_axis = get_zcr_counts(zcrs_rows[:, axis])

                if not zcr_counts_rows.size:
                    zcr_counts_rows = np.array(zcr_counts_axis)
                else:
                    zcr_counts_rows = np.append(zcr_counts_rows, zcr_counts_axis)

    else:  # for 1d array
        if not np.where(np.diff(np.sign(val)))[0].size:  # if svm, transformed to zscore as svm has no negative valeus
            val = scipy.stats.zscore(val)
            zcrs = np.diff(np.sign(val))
            zcr_counts_rows = np.array([get_zcr_counts(zcrs)])
        else:
            zcrs = np.diff(np.sign(val))
            zcr_counts_rows = np.array([get_zcr_counts(zcrs)])

    #print("zcr_counts_rows:{}".format(np.array(zcr_counts_rows).reshape(-1)))
    return zcr_counts_rows


def get_statistical_features(data, axis):
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)
    rms = np.sqrt(np.mean(data ** 2, axis=axis))
    max_amp = np.max(data, axis=axis)
    min_amp = np.min(data, axis=axis)
    median = np.median(data, axis=axis)
    skewness = scipy.stats.skew(data, axis=axis)
    kurtosis = scipy.stats.kurtosis(data, axis=axis)
    first_quartile = np.percentile(data, 25, axis=axis)
    third_quartile = np.percentile(data, 75, axis=axis)
    # print("mean:{}, std:{}, rms:{}, max_amp:{}, min_amp:{}, median:{}, first_quartile:{}, third_quartile:{}".format(mean, std, rms, max_amp, min_amp, median, first_quartile, third_quartile))
    # print("mean:{}, std:{}, rms:{}, max_amp:{}, min_amp:{}, median:{}, first_quartile:{}".format(mean.shape, std.shape, rms.shape, max_amp.shape, min_amp.shape, median.shape, first_quartile.shape))
    features = np.hstack((mean, std, rms, max_amp, min_amp, median, skewness, kurtosis, first_quartile, third_quartile))
    return features

def extract_time_domain_features(vec_data):
    basic_stat_features = get_basic_statistic_features(vec_data)
    autocorr_val = autocorr(vec_data)
    zcr_counts = zero_crossing_rate(vec_data)
    time_domain_features = np.hstack((basic_stat_features, autocorr_val, zcr_counts))  # , autocorr_val))
    return time_domain_features

def autocorr(seq):
    auto_corr_rows = np.empty(0)

    if 2 < len(seq.shape):  # for 3 d array
        rows = np.arange(0, seq.shape[0])
        axes = np.arange(0, seq.shape[-1])

        for i, row in enumerate(rows):
            auto_corr_axes = np.empty(0)

            for axis in axes:
                auto_corr_axis = np.correlate(seq[row, :, axis], seq[row, :, axis])

                if not auto_corr_axes.size:
                    auto_corr_axes = auto_corr_axis
                else:
                    auto_corr_axes = np.append(auto_corr_axes, auto_corr_axis)

            if not auto_corr_rows.size:
                auto_corr_rows = auto_corr_axes
            else:
                auto_corr_rows = np.vstack((auto_corr_rows, auto_corr_axes))

            progressBar(i, len(rows), 20, "Auto-Correlation")

    elif 1 < len(seq.shape) <= 2:  # for 2d array
        axes = np.arange(0, seq.shape[-1])

        if len(axes) < 3:  # for 2d array with 1 column. ex) svm
            seq = seq.reshape(-1)
            auto_corr_rows = np.correlate(seq, seq)
        else:  # for 2d array with more than a column. ex) acc --> (2423, 3)
            d = seq
            for axis in axes:
                auto_corr_axis = np.correlate(seq[:, axis], seq[:, axis])

                if not auto_corr_rows.size:
                    auto_corr_rows = auto_corr_axis
                else:
                    auto_corr_rows = np.append(auto_corr_rows, auto_corr_axis)

    else:  # for 1d array
        auto_corr_rows = np.correlate(seq, seq)
    return auto_corr_rows
    
def get_basic_statistic_features(seq):
    axes = np.arange(0, seq.shape[-1])
    if 2 < len(seq.shape):  # for 3 d array
        axis = 1

    elif 1 < len(seq.shape) <= 2:  # for 2d array
        if seq.shape[-1] < 3:  # for 2d array with 1 column. ex) svm
            seq = seq.reshape(-1)
            axis = -1
        else:  # for 2d array with more than a column. ex) acc --> (2423, 3)
            axis = 0

    else:  # for 1d array
        axis = -1

    statistical_features = get_statistical_features(seq, axis)

    return statistical_features



def calc_freq_energy(psd):
    length = len(psd)
    mean = np.mean(psd)
    mean_sqr = (psd - mean) ** 2
    sum_mean_sqr = np.sum(mean_sqr)
    return sum_mean_sqr / length


def calc_spectral_centroid(f, psd):
    spectrum = psd
    sum_spectrum = np.sum(spectrum)
    sum_spectrum = 1e-10 if sum_spectrum == 0 else sum_spectrum
    norm_spectrum = spectrum / sum_spectrum
    norm_freq = f
    spectral_centroid = np.sum(norm_freq * norm_spectrum)
    return spectral_centroid


# Extract Features From the Standardized Dataset
def extract_features(datasets, is_time_feature): # 0 for pocket_acc, 1 for imu
    feature_type = 'Time Domain' if is_time_feature else "Frequency Domain"
    print(f"{feature_type} Feature Extraction in progress...", end='\r', flush=True)
    if is_time_feature:
        num_feature = 12
    else: 
        num_feature = 11
        
    num_features = datasets.shape[-1] * num_feature
    feature_sets = np.empty((0, num_features)) #252 #36 #216
    
    for i, dataset in enumerate(datasets):
        if is_time_feature:
            features = extract_time_domain_features(dataset)  
        else: 
            features = extract_freq_domain_features(dataset, 60) # sampling rate =  60Hz
        feature_sets = np.vstack((feature_sets, features))
        text = f"{i}/{len(datasets)} Completed..."
        padding = " " * (80 - 30) if 80 else ""
    
        print(text + padding, end='\r', flush=True)
    feature_sets = pd.DataFrame(feature_sets).fillna(method='ffill').values
    print(f"{feature_type} Feature Extraction completed...", end='\r', flush=True)
    return feature_sets


def get_scaled_feature_set(feature_set, label, trainIdx, valIdx, testIdx):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(feature_set[trainIdx])
    X_val = scaler.transform(feature_set[valIdx])
    X_test = scaler.transform(feature_set[testIdx])

    y_train = label[trainIdx]
    y_val = label[valIdx]
    y_test = label[testIdx]
    return X_train, y_train, X_val, y_val, X_test, y_test
    
    
def get_balanced_feature_set(X_train, y_train, random_seed=1):
    sm = SMOTE(random_state=random_seed)
    if len(y_train.shape) > 1:
        y_train = y_train.reshape(-1)
    print(f'Before Balancing Dataset: {Counter(y_train)}', end='\r', flush=True)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f'After Balancing Dataset: {Counter(y_train)}', end='\r', flush=True)
    return X_train, y_train

def get_time2vec_feature_set(t2v, dataset, label, trainIdx, valIdx, testIdx):
    y_train = to_categorical(label[trainIdx])
    y_val = to_categorical(label[valIdx])
    y_test = to_categorical(label[testIdx])

    X_train = time2vec_feature_set(t2v, dataset[trainIdx])
    X_val =   time2vec_feature_set(t2v, dataset[valIdx])
    X_test =  time2vec_feature_set(t2v, dataset[testIdx])        
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_timeseries_feature_set(dataset, label, trainIdx, valIdx, testIdx):
    y_train = to_categorical(label[trainIdx])
    y_val = to_categorical(label[valIdx])
    y_test = to_categorical(label[testIdx])
    
    X_train = get_timeseries_dataset(dataset[trainIdx], dataset[trainIdx].shape[1])
    X_val =   get_timeseries_dataset(dataset[valIdx], dataset[valIdx].shape[1])
    X_test =  get_timeseries_dataset(dataset[testIdx], dataset[testIdx].shape[1])
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[2], X_train.shape[3]))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[2], X_val.shape[3]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[2], X_test.shape[3]))
    
    return X_train, y_train, X_val, y_val, X_test, y_test

#dataset = np.diff(dataset, axis=1, n=1)
def extract_tsfel_feature(dataset, feature_type): #  feature_type = "temporal", "statistical", "spectral"
    cfg_file = tsfel.get_features_by_domain(feature_type)# If no argument is passed retrieves all available features
    if feature_type == "temporal":
        n = 18
    elif feature_type == "statistical":
        n = 36
    elif feature_type == "spectral":
        n = 335
    else:
        n = 389
    n_features =  n * 108
    feature_set = np.empty((0, n_features))
    
    for i, item in enumerate(dataset):
        feature = tsfel.time_series_features_extractor(cfg_file, item, fs=60, window_size=dataset.shape[1], n_jobs=10, single_window=True, verbose=0)    # Receives a time series sampled at 50 Hz, divides into windows of size 250 (i.e. 5 seconds) and extracts all features
        feature_set = np.vstack((feature_set, feature))
    return feature_set
    
def get_rpy(dataset): # dataset format --> records X rows X columns
    data_length = 600
    n_col_for_each_sensor = 36
    total_rpy = np.empty((0, data_length, n_col_for_each_sensor))
    start_mag_idx = 72
    n_interval = 3
    for data in tqdm(dataset, desc='roll-pitch-yaw calculation'):
        rpy_arr = np.empty((data_length, 0))
        for sensor_loc_idx in np.arange(0, n_col_for_each_sensor, n_interval):
            acc = data[:, sensor_loc_idx:sensor_loc_idx+n_interval]
            mag = data[:, sensor_loc_idx+start_mag_idx:sensor_loc_idx+start_mag_idx+n_interval]
            tilt = Tilt(acc, mag, as_angles=True)
            rpy = tilt.Q.reshape((data_length, n_interval))
            rpy_arr = np.hstack((rpy_arr, rpy))    
        
        total_rpy = np.vstack((total_rpy, rpy_arr.reshape((1, data_length, n_col_for_each_sensor))))
    return total_rpy 

def convert_to_svm(arr, start_idx):
    n_row = 600
    n_col = 12
    acc_last_idx = 36
    interval = 3
    svm_arr = np.empty((0, n_row, n_col))
    for row_idx in tqdm(np.arange(arr.shape[0]), desc='SVM conversion'):
        svm_col_wise_arr = np.empty((n_row, 0))
        for col_idx in np.arange(start_idx, start_idx+acc_last_idx, interval):
            acc = arr[row_idx,:,col_idx:col_idx+interval]
            svm = np.sqrt(np.sum(acc**2, axis=1)).reshape((-1, 1))
            svm_col_wise_arr = np.hstack((svm_col_wise_arr, svm)) 
        svm_arr = np.vstack((svm_arr, svm_col_wise_arr.reshape(1, n_row, n_col)))
    return svm_arr

from numpy import sin, cos, pi
def R_x(x):
    return np.array([[1, 0, 0],
                     [0, cos(-x), -sin(-x)],
                     [0, sin(-x), cos(-x)]])
    
def R_y(y):
    return np.array([[cos(-y), 0, -sin(-y)],
                     [0, 1, 0],
                     [sin(-y), 0, cos(-y)]])

def R_z(z):
    return np.array([[cos(-z), -sin(-z), 0],
                     [sin(-z), cos(-z), 0],
                     [0, 0, 1]])

def transform_to_earth_frame(dataset, rpy):
    earth_accels = np.empty((0, 600, 36))
    gravities = np.empty((0, 600, 36))
    
    for i, arr in enumerate(zip(dataset, tqdm(rpy, desc= 'Converting to Earth Frame'))):
        earth_accel_col = np.empty((600, 0))
        gravity_col = np.empty((600, 0))
        #print(f'arr[0].shape:{arr[0].shape}, arr[1].shape: {arr[1].shape}')
        for col in range(0, 36, 3):
            acc = arr[0][:, col:col+3]
            gyro = arr[0][:, col+36:col+36+3]        
            roll = arr[1][:, col]
            pitch = arr[1][:, col+1]
            yaw = arr[1][:, col+2]
            earth_accel = []
            gravity = []
            #print(f'acc.shape:{acc.shape}, gyro.shape:{gyro.shape}')
            for row in range(arr[0].shape[0]):
                a = R_z(yaw[row]) @ R_y(roll[row]) @ R_x(pitch[row]) @ acc[row, :] 
                earth_accel.append(a.tolist())
                g = R_z(yaw[row]) @ R_y(roll[row]) @ R_x(pitch[row]) @ gyro[row, :] 
                gravity.append(g.tolist())
            
            earth_accel = np.array(earth_accel)
            #print(f'earth_accel.shape:{earth_accel.shape}, earth_accel_col.shape:{earth_accel_col.shape}')
            earth_accel_col = np.hstack((earth_accel_col, earth_accel))
            gravity = np.array(gravity)
            gravity_col = np.hstack((gravity_col, gravity))
        
        earth_accels = np.vstack((earth_accels, earth_accel_col.reshape(1, 600, 36)))
        gravities = np.vstack((gravities, gravity_col.reshape(1, 600, 36)))
    return earth_accels, gravities
    
def get_misclassified_indices(model, index, X_test, y_test):
    if len(X_test.shape) > 2:
        X_test = X_test.reshape(-1, X_test.shape[1] * X_test.shape[2])
    if len(y_test.shape) > 1:
        errorIdx = np.where((model.predict(X_test) == np.argmax(y_test, 1)) == False)[0]
    else:
        errorIdx = np.where((model.predict(X_test) == np.argmax(y_test, 1)) == False)[0]
    return index[errorIdx]

def save_ml_model(model, model_name, label, feature_type):
    print(f'Saving {model_name} model in porgress...', end='\r')
    import datetime
    today = datetime.datetime.today().strftime('%Y%m%d')
    file_name = "best_" + model_name + "_model_" + feature_type + "_" + str(len(np.unique(label))) + "_classes_" + today + ".sav"
    joblib.dump(model, os.path.join('./models/ml_models', file_name))# + clf_type + '_' + model_names[i].lower() + '_' + date_info + '.sav')
    print(f'Saving {model_name} model completed...\t\t\t\t\t')
    

def save_ml_models(models, model_names, label, feature_type):
    import datetime
    today = datetime.datetime.today().strftime('%Y%m%d')
    for i, model in enumerate(models):
        if len(np.unique(np.argmax(label, 1))) == 3:
            clf_type = 'fall_direction'
        elif len(np.unique(np.argmax(label, 1))) == 2:
            clf_type = 'fall_nonfall'
        else:
            clf_type = 'non_fall_direction'
            
        file_name = "best_" + model_names[i] + "_model_" + feature_type + "_" + str(len(np.unique(label))) + "_classes_" + today + ".sav"
        joblib.dump(model, os.path.join('./models/ml_models', file_name))# + clf_type + '_' + model_names[i].lower() + '_' + date_info + '.sav')
    
    print(f'Saving Models Completed...')

class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, kernel_size = 1):
        super(Time2Vec, self).__init__(trainable = True, name = 'Time2VecLayer')
        self.k = kernel_size

    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name = 'wb', shape = (input_shape[1],), initializer = 'uniform', trainable = True)
        self.bb = self.add_weight(name = 'bb', shape = (input_shape[1],), initializer = 'uniform', trainable = True)
        # periodic
        self.wa = self.add_weight(name = 'wa', shape = (1, input_shape[1], self.k), initializer = 'uniform', trainable = True)
        self.ba = self.add_weight(name = 'ba', shape = (1, input_shape[1], self.k), initializer = 'uniform', trainable = True)
        super(Time2Vec, self).build(input_shape)

    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = K.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp) # or K.cos(.)
        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1] * (self.k + 1)))
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * (self.k + 1))

