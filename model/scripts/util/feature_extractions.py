import os, sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from tensorflow.keras.utils import to_categorical
from collections import Counter

from util import util_nia_project as nia
from util.util_nia_project import Time2Vec
import matplotlib.pyplot as plt
import pickle



def feature_extraction(home_path, output_plk, target_path='SENSOR', dataset='train'):
    target_dataset_path = target_path
    ds_default_path = os.path.join(home_path, target_dataset_path)
    ds_path = os.path.join(ds_default_path, 'test')
    # list up file names from N, Y (FY, BY, SY) folders
    fnames_training = nia.get_filenames(ds_path)
    
    # loading dataset (N, Y - FY, BY, SY)
    dataset_4_classes, label_4_classes, valid_fnames = nia.load_datasets(fnames_training)
    ord_label_4_classes = nia.encode_ordinal_label(label_4_classes)
    
    # creating dataset and labels according to number of classes - fall only (3 classes),  fall / non-fall (2 classes)
    dataset_2_classes = dataset_4_classes.copy()
    label_2_classes = nia.get_labels_for_n_classes(valid_fnames, 2)
    ord_label_2_classes = nia.encode_ordinal_label(label_2_classes)

    index_4_fall_data = nia.get_index_fall_only(valid_fnames)
    dataset_3_classes = dataset_4_classes[index_4_fall_data]
    label_3_classes = nia.get_labels_for_n_classes(valid_fnames[index_4_fall_data], 3)
    ord_label_3_classes = nia.encode_ordinal_label(label_3_classes)

    # put datasets into a list - will be used for model testing loop
    #datasets = [dataset_2_classes, dataset_3_classes, dataset_4_classes]
    #labels = [label_2_classes, label_3_classes, label_4_classes]
    #ord_labels = [ord_label_2_classes, ord_label_3_classes, ord_label_4_classes]

    datasets = [dataset_2_classes, dataset_3_classes]
    labels = [label_2_classes, label_3_classes]
    ord_labels = [ord_label_2_classes, ord_label_3_classes]

    # extracting features
    spatio_temporal_feature_sets = []
    for dataset, label in zip(datasets, labels):
        trainIdx, testIdx = train_test_split(np.arange(0, len(label)), test_size=0.3, shuffle=True)
        #dataset = ds[0]#[[0,100, 200,300, 400, 500, 600,-400, -300, -100, -1]]
        #label = ds[1]#[[0,100, 200,300, 400, 500, 600,-400, -300, -100, -1]]
        print(f'Feature extraction for  {len(np.unique(label))}-class dataset started....', end='\r', flush=True)
        time_feature_set = nia.extract_features(dataset, True)
        freq_feature_set = nia.extract_features(dataset, False)
        spatio_temporal_feature_set = np.hstack((time_feature_set, freq_feature_set))
        spatio_temporal_feature_sets.append(spatio_temporal_feature_set)
        print(f'Feature extraction for {len(np.unique(label))}-class dataset completed....\t\t\t\t\t\t\t\t')

    variables_to_save = {
        'spatio_temporal_feature_sets': spatio_temporal_feature_sets,
        'datasets': datasets,
        'labels': labels,
        'ord_labels' : ord_labels,
    }

    with open('{}.pkl'.format(output_plk), 'wb') as file:
        pickle.dump(variables_to_save, file)
        





        