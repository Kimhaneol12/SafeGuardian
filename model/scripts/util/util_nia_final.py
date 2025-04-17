import numpy as np
import pandas as pd

import joblib
import pickle
import os
from tqdm import tqdm

from keras.models import load_model
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")


# ###sensor###
# %load_ext autoreload
# %autoreload 2

import os, sys, joblib
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from util.util_nia_project import Time2Vec
from util import util_nia_project as nia
from util.feature_extractions import feature_extraction

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

import datatable as dt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


def get_file_list():
    homepath = os.getcwd()
    target_path = 'test'
    test_folder_path = os.path.join(homepath, target_path)
    test_sensor_list = [test_folder_path+'/{}/{}.xlsx'.format(i,i) for i in os.listdir(test_folder_path)]
    test_npy_list = [test_folder_path+'/{}/{}'.format(i,i) for i in os.listdir(test_folder_path)]

    return test_sensor_list, test_npy_list
    
def load_data(fname, prefix=""):
    df = pd.read_excel(os.path.join(prefix, fname))
    return df

def get_npy_label(fname):
    non_fall = "N"; forward_fall = "FY"; backward_fall = "BY"; lateral_fall = "SY"
    label = fname.split("\\")[-1]
    if non_fall in label:
        label = 'N'
    elif forward_fall in label:
        label = 'FY'
    elif backward_fall in label:
        label = 'BY'
    else:
        label = 'SY'
    return label


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
            labels = np.hstack((labels, get_npy_label(fname)))
            new_fnames = np.hstack((new_fnames, fname)) 
        except:
            try: # dealing with data that contain more than 600 rows
                if data.shape[-1] == n_axis and data.shape[-2] > length:
                    data = data[-length:, :]
                    datasets = np.vstack((datasets, data.reshape(-1, length, n_axis)))
                    print(f'Recoverd file: {fname}')
                    labels = np.hstack((labels, get_npy_label(fname)))
                    new_fnames = np.hstack((new_fnames, fname))
                else:
                    n_error += 1
                    print(f'Error occured for file name: {fname}, index: {i}, shape: {data.shape}')
                
            except: # throw error when number of axes of sensor lower than 108
                n_error += 1
                print(f'Error occured for file name: {fname}, index: {i}, shape: {data.shape}')
        print(f'Loading dataset: {i+1}/{len(fnames)}', end='\r', flush=True)
    print(f'Total number of {datasets.shape[0]} sensor datasets loaded... Error occured for {n_error} files....', flush=True)
    
    return datasets, np.array(labels), new_fnames

def save_sensor_feature_FD():
    test_sensor_list, test_npy_list = get_file_list()
    dataset_4_classes, label_4_classes, valid_fnames = load_datasets(test_sensor_list)
    ord_label_4_classes = nia.encode_ordinal_label(label_4_classes)
    index_4_fall_data = nia.get_index_fall_only(valid_fnames)
    
    ##FD
    dataset_3_classes = dataset_4_classes[index_4_fall_data]
    label_3_classes = nia.get_labels_for_n_classes(valid_fnames[index_4_fall_data], 3)
    ord_label_3_classes = nia.encode_ordinal_label(label_3_classes)

    time_feature_set = nia.extract_features(dataset_3_classes, True)
    freq_feature_set = nia.extract_features(dataset_3_classes, False)
    spatio_temporal_feature_set = np.hstack((time_feature_set, freq_feature_set))
    print(f'Feature extraction for {len(np.unique(label_3_classes))}-class dataset completed....\t\t\t\t\t\t\t\t')

    return spatio_temporal_feature_set, dataset_3_classes, label_3_classes, ord_label_3_classes


def save_sensor_feature_FNF():
    test_sensor_list, test_npy_list = get_file_list()
    dataset_4_classes, label_4_classes, valid_fnames = load_datasets(test_sensor_list)
    ord_label_4_classes = nia.encode_ordinal_label(label_4_classes)
    index_4_fall_data = nia.get_index_fall_only(valid_fnames)

    # FNF
    dataset_2_classes = dataset_4_classes.copy()
    label_2_classes = nia.get_labels_for_n_classes(valid_fnames, 2)
    ord_label_2_classes = nia.encode_ordinal_label(label_2_classes)

    time_feature_set = nia.extract_features(dataset_2_classes, True)
    freq_feature_set = nia.extract_features(dataset_2_classes, False)
    spatio_temporal_feature_set = np.hstack((time_feature_set, freq_feature_set))
    print(f'Feature extraction for {len(np.unique(label_2_classes))}-class dataset completed....\t\t\t\t\t\t\t\t')

    return spatio_temporal_feature_set, dataset_2_classes, label_2_classes, ord_label_2_classes


def npy_reader(i, sequences):
    try:
        if len(os.listdir(i)) < 600:
            print('##### keypoint extraction failed #####')
            pass
        else:
            window = []
            for num in range(600):
                file_path = os.path.join(i, f'{num}.npy')
                npy = np.load(file_path)
                window.append(npy)
        sequences.append(window)
    except:
        print(f'Check the file path : {file_path}')
        
    return sequences


def npy_generator_FD():
    test_sensor_list, test_npy_list = get_file_list()

    sequences, label_npy = [], []

    for i in tqdm(test_npy_list):
        if i.__contains__('BY'):
            label_npy.append('BY')
            sequences = npy_reader(i, sequences)
        elif i.__contains__('FY'):
            label_npy.append('FY')
            sequences = npy_reader(i, sequences)
        elif i.__contains__('SY'):
            label_npy.append('SY')
            sequences = npy_reader(i, sequences)
        else:
            pass
    
    test_vid = np.array(sequences)
    nsamples, nx, ny = test_vid.shape
    test_vid = test_vid.reshape((nsamples, nx * ny))
    le = LabelEncoder()
    label_npy = le.fit_transform(label_npy)
    label_npy = label_npy.tolist()

    print(le.classes_)
    print('라벨 수', len(label_npy))

    scaler = joblib.load('./util/std_scaler_3_classes_spatio_temporal.bin')
    
    return test_vid, label_npy, scaler

def npy_generator_FNF():

    test_sensor_list, test_npy_list = get_file_list()

    sequences, label_npy = [], []
    # i = test_npy_list[0]
    for i in tqdm(test_npy_list):
        if i.__contains__('BY'):
            label_npy.append('Fall')
            sequences = npy_reader(i, sequences)
        elif i.__contains__('FY'):
            label_npy.append('Fall')
            sequences = npy_reader(i, sequences)
        elif i.__contains__('SY'):
            label_npy.append('Fall')
            sequences = npy_reader(i, sequences)
        elif i.__contains__('N'):
            label_npy.append('Non Fall')
            sequences = npy_reader(i, sequences)
        else:
            pass
    
    test_vid = np.array(sequences)
    nsamples, nx, ny = test_vid.shape
    test_vid = test_vid.reshape((nsamples, nx * ny))
    le = LabelEncoder()
    label_npy = le.fit_transform(label_npy)
    label_npy = label_npy.tolist()

    print(le.classes_)
    print('라벨 수', len(label_npy))

    scaler = joblib.load('./util/std_scaler_2_classes_spatio_temporal.bin')
    
    return test_vid, label_npy, scaler

def model_loading(test_type):
    homepath = os.getcwd()
    target_path = 'models'
    model_folder = os.path.join(homepath, target_path)
    type_model_folder = os.path.join(model_folder, test_type)

    model_name_list, npy_model_list, dl_model_list, ml_model_list= [],[], [], []    
    for i in os.listdir(type_model_folder):
        path_= os.path.join(type_model_folder, i)
        if '.pkl' in i:
            model = joblib.load(path_)
            npy_model_list.append(model)
            model_name_list.append(i)
        elif '.h5' in i:
            model = tf.keras.models.load_model(path_, compile =False)
            dl_model_list.append(model)
            model_name_list.append(i)
        elif '.sav' in i:
            model = joblib.load(path_)
            ml_model_list.append(model)
            model_name_list.append(i)


    print((len(model_name_list),'Load Success'))
    return model_name_list, npy_model_list, dl_model_list, ml_model_list

def npy_model_test(model_list, test_vid, model_name_list):
    pred_label_list, proba_list = [], []
    for num, model in enumerate(model_list):
        try:
            pred_label = model.predict(test_vid)
            pred_proba = model.predict_proba(test_vid)
            pred_label_list.append(pred_label)
            proba_list.append(pred_proba)
        except:
            model_name_ = model_name_list[num]
            print('"{}" is not for this data'.format(model_name_))
            pass
    return pred_label_list, proba_list

def dl_model_test(model_list, dataset_classes, model_name_list):
    dl_predicted_label, dl_proba_list =[], []
    for num, model in enumerate(model_list):
        try:
            proba_list = model.predict(dataset_classes).tolist()
            proba_list = np.array(proba_list)
            dl_proba_list.append(proba_list)
            pred_label_list = np.argmax(proba_list, 1).tolist()
            dl_predicted_label.append(pred_label_list)
        except:
            model_name_ = model_name_list[num]
            print('"{}" is not for this data'.format(model_name_))
            pass

    return dl_predicted_label, dl_proba_list
    

def ml_model_test_FD(model_list, spatio_temporal_feature_set, scaler):
    ml_predicted_label, ml_proba_list = [], [] 
    testX_sc = scaler.transform(spatio_temporal_feature_set)
    for num, model in enumerate(model_list):
        try: 
            pred_label = model.predict(testX_sc)
            pred_proba= model.predict_proba(testX_sc)
            ml_predicted_label.append(pred_label)
            ml_proba_list.append(pred_proba)

        except:
            ens_avg_proba = 0
            for i in model:
                ens_avg_proba += i.predict_proba(testX_sc)
            ens_avg_proba = ens_avg_proba/3

            pass

    return ml_predicted_label, ml_proba_list, ens_avg_proba

def ml_model_test_FNF(model_list, spatio_temporal_feature_set, scaler):
    ml_predicted_label, ml_proba_list = [], [] 
    testX_sc = scaler.transform(spatio_temporal_feature_set)
    for num, model in enumerate(model_list):
        try: 
            pred_label = model.predict(testX_sc)
            pred_proba= model.predict_proba(testX_sc)
            ml_predicted_label.append(pred_label)
            ml_proba_list.append(pred_proba)

        except:
            ens_avg_proba = 0
            for i in model:
                ens_avg_proba += i.predict_proba(testX_sc)
            ens_avg_proba = ens_avg_proba/2

            pass

    return ml_predicted_label, ml_proba_list, ens_avg_proba

def voting(npy_proba_list, dl_proba_list, ml_proba_list, ens_avg_proba):
    result = npy_proba_list[0] 
    for arr in npy_proba_list[1:]:
        result = np.hstack((result, arr))
    for arr in dl_proba_list:
        result = np.hstack((result, arr))
    for arr in ml_proba_list:
        result = np.hstack((result, arr))
    result = np.hstack((result, ens_avg_proba))
    return result

def final_pred_label_FD(result):
    final_label_list = []

    for one_folder in result:
        max_prob = max(one_folder.tolist())
        loc_prob = one_folder.tolist().index(max_prob)
        if loc_prob%3==0:
            final_label_list.append(0)
        elif loc_prob%3==1:
            final_label_list.append(1)
        elif loc_prob%3==2:
            final_label_list.append(2)
    
    return final_label_list

def final_pred_label_FNF(result):
    final_label_list=[]
    for one_folder in result:
        max_prob = max(one_folder.tolist())
        loc_prob = one_folder.tolist().index(max_prob)
        if loc_prob%2==0:
            final_label_list.append(1)
        elif loc_prob%2==1:
            final_label_list.append(0)

    print(len(final_label_list))

    return print(len(final_label_list))
        
    
def evaluation_FD(label_npy,final_label_list ):
    y_true = label_npy

    print(classification_report(y_true, final_label_list, labels=[0, 1, 2]))
    print('acc', accuracy_score(y_true, final_label_list))
    confusion_matrix(y_true, final_label_list, labels=[0, 1, 2])
    

def evaluation_FNF(label_npy,final_label_list ):
    y_true = label_npy

    print(classification_report(y_true, final_label_list, labels=[0, 1]))
    print('acc', accuracy_score(y_true, final_label_list))
    confusion_matrix(y_true, final_label_list, labels=[0, 1])

    
