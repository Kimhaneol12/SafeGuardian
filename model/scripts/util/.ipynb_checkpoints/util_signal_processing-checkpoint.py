from ahrs.filters import Tilt
from scipy.signal import butter, filtfilt

import random
import numpy as np
import pandas as pd

'''
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(Path(os.getcwd()).parent.absolute()))
'''
from constants import configure as cfg
from utilities.imu.util_common_operation import progressBar


# synchronize two datasets according to timestamp of each dataset by finding the common timestamps from the both datasets
def sync_dataset(timestamp_1, timestamp_2, precision_level=cfg.PRECESION_LEVEL_4_SYNC):
    print("Time synchrnonization in Progress...", end='\r', flush=True)
    
    # find intersection of timestamp of each dataset
    # resampling precision (int or float rounded at second decimal point)
    # get intersection of timestamps that are the same upto 1 digit of millisecond "xxxxxxxxxx.x"
    temp_timestamp_1 = np.round(timestamp_1.astype(float), precision_level)  # optimized precision --> round(value.astype('float'),1)
    temp_timestamp_2 = np.round(timestamp_2.astype(float), precision_level)
    intersect_index = set(np.intersect1d(temp_timestamp_1, temp_timestamp_2))

    # index of timestamp that is matched with an element in the datasets
    # get the indexes of timestamps that are common in both datasets
    synced_idx_4_dataset_1 = [i for i, item in enumerate(temp_timestamp_1) if item in intersect_index]
    synced_idx_4_dataset_2 = [i for i, item in enumerate(temp_timestamp_2) if item in intersect_index]

    if len(synced_idx_4_dataset_1) > len(synced_idx_4_dataset_2):
        synced_idx_4_dataset_1 = synced_idx_4_dataset_1[:len(synced_idx_4_dataset_2)]
    else:
        synced_idx_4_dataset_2 = synced_idx_4_dataset_2[:len(synced_idx_4_dataset_1)]
    print(f'Time synchrnonization Completed... The size of intersection of both sets:{len(intersect_index)}, Before - Timestamp1: {len(temp_timestamp_1)},  Timestamp2: {len(temp_timestamp_2)}, After - Timestamp1: {len(synced_idx_4_dataset_1)},  Timestamp2: {len(synced_idx_4_dataset_1)}')
    return synced_idx_4_dataset_1, synced_idx_4_dataset_2


def resample(timestamp, data, designated_fs=cfg.DOWNSAMPLING_FREQUENCY):
    print("Resampling started", end='\r', flush=True)

    ### find the indices where change in second occured
    diff = np.diff(timestamp.astype(int))
    idx_sec_changed = np.where(diff == 1)[0]

    ### find the first index where increase in the second value
    idx_to_resample = idx_sec_changed[np.where(np.diff(idx_sec_changed))[0]]

    if idx_to_resample.size == 0:  # when null
        idx_to_resample = idx_sec_changed  # [:-1]
    idx_to_resample = sorted(idx_to_resample)

    # set designated fs if the fs is greater than sampling rate
    found_fs = int(np.average(np.diff(idx_to_resample)))
    if designated_fs > found_fs:
        designated_fs = found_fs

    idx_interval = []
    random.seed(random.randint(0, designated_fs))
    for i, item in enumerate(idx_to_resample, start=0):

        # if found minimum freq is much smaller than initial sampling frequency, item + fs needs changing. Otherwise, the last part within a minute is discarded
        if i < len(idx_to_resample) - 1 and idx_to_resample[i + 1] - item > designated_fs:
            try:
                index_range = sorted(random.sample(range(item, idx_to_resample[i + 1]), k=designated_fs))
                print("len(index_range):{}".format(len(index_range)), end='\r')

                idx_interval.append(index_range)
            except:
                number_of_samples = idx_to_resample[i + 1] - item
                idx_range = sorted(random.sample(range(item, idx_to_resample[i + 1]), k=number_of_samples))
                # print("exception ----------- len(idx_range):{}, idx_range:{}".format(len(idx_range), idx_range))
                idx_interval.append(index_range)

            #if i % 1000 == 0:
                #progressBar(i, len(idx_to_resample), msg='Part 1 -- Resampling: from {} Hz to {} Hz'.format(found_fs, designated_fs))

    idx_generated = np.unique(np.array(idx_interval).reshape(-1))
    resampled_timestamp = timestamp[idx_generated]
    resampled_data = data[idx_generated]  # .reset_index(drop=True)
    
    print(f"Resampling completed... Length of Data - Before: {len(timestamp)}, After: {len(resampled_timestamp)}")

    return resampled_timestamp, resampled_data


## different style of implementation with better performance
def find_min_fs(t1, t2):  # compare timestamps and obtain minimum sampling frequency
    min_fs_t1 = np.min(np.diff(np.where(np.diff(t1.astype(int)) != 0)[0]))
    min_fs_t2 = np.min(np.diff(np.where(np.diff(t2.astype(int)) != 0)[0]))
    if min_fs_t1 > min_fs_t2:
        min_fs = min_fs_t2
    else:
        min_fs = min_fs_t1
    return min_fs


def lpf(seq, fc, fs, order, mode=3):
    print("Low pass filtering stared", end="\r", flush=True)
    if mode == 6:
        acc_x = butter_lpf(seq[:, 0], fc, fs, order)
        acc_y = butter_lpf(seq[:, 1], fc, fs, order)
        acc_z = butter_lpf(seq[:, 2], fc, fs, order)
        gyro_x = butter_lpf(seq[:, 3], fc, fs, order)
        gyro_y = butter_lpf(seq[:, 4], fc, fs, order)
        gyro_z = butter_lpf(seq[:, 5], fc, fs, order)
        lpf_seq = np.array([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]).transpose()
    elif mode == 3:
        x = butter_lpf(seq[:, 0], fc, fs, order)
        y = butter_lpf(seq[:, 1], fc, fs, order)
        z = butter_lpf(seq[:, 2], fc, fs, order)
        lpf_seq = np.array([x, y, z]).transpose()
    else:
        lpf_seq = butter_lpf(seq, fc, fs, order)
    '''elif mode == 2:
            x = butter_lpf(seq[:, 0], fc, fs, order)
            y = butter_lpf(seq[:, 1], fc, fs, order)
            lpf_seq = np.array([x, y]).transpose()'''
    print("Low pass filtering completed", end="\r", flush=True)
    return lpf_seq


def butter_lpf(data, cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq  # normalize the frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # , analog=False, output='sos')#, analog=False)
    y = filtfilt(b, a, data);
    return y


def normalization(x):
    if type(x) != np.array:
        x = np.array(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def butter_bpf(data, lfc, ufc, fs, order=3):
    nyq = 0.5 * fs
    lower = lfc / nyq
    upper = ufc / nyq
    b, a = butter(order, [lower, upper], btype='band', analog=False)  # , analog=False, output='sos')#, analog=False)
    y = filtfilt(b, a, data);
    return y


def filter_signals(seq, lfc, ufc, fs, order, mode, filter_type):
    if filter_type == 0:
        y = lpf(seq, ufc, fs, order, mode)
    else:
        y = bpf(seq, lfc, ufc, fs, order, mode)
    return y


def bpf(seq, lfc, ufc, fs, order=4, mode=1):
    print("Band pass filtering stared", end="\r", flush=True)
    if mode == 6:
        acc_x = butter_bpf(seq[:, 0], lfc, ufc, fs, order)
        acc_y = butter_bpf(seq[:, 1], lfc, ufc, fs, order)
        acc_z = butter_bpf(seq[:, 2], lfc, ufc, fs, order)
        gyr_x = butter_bpf(seq[:, 3], lfc, ufc, fs, order)
        gyr_y = butter_bpf(seq[:, 4], lfc, ufc, fs, order)
        gyr_z = butter_bpf(seq[:, 5], lfc, ufc, fs, order)
        bpf = np.array([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]).transpose()

    elif mode == 3:
        x = butter_bpf(seq[:, 0], lfc, ufc, fs, order)
        y = butter_bpf(seq[:, 1], lfc, ufc, fs, order)
        z = butter_bpf(seq[:, 2], lfc, ufc, fs, order)
        bpf = np.array([x, y, z]).transpose()

    else:
        bpf = butter_bpf(seq, lfc, ufc, fs, order)
        bpf = np.array([bpf])

        '''elif mode == 2: 
        x = butter_bpf(seq[:,0], lfc, ufc, fs, order)
        y = butter_bpf(seq[:,1], lfc, ufc, fs, order)
        bpf = np.array([x, y]).transpose()    
        '''
    print("Band pass filtering completed...")
    return bpf


def remove_gravity(data):
    if data.shape[1] > 3:
        acc = data[:, :3]
    else:
        acc = data

    q = get_quaternions(acc);
    # since q is given in the form of [w, x, y, z] = [q[0], q[1], q[2], q[3]]
    # get expected direction of gravity
    g = np.zeros([len(q), 3])
    # get expected direction of gravity
    g[:, 0] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    g[:, 1] = 2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3])
    g[:, 2] = q[:, 0] * q[:, 0] - q[:, 1] * q[:, 1] - q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
    # compensate accelerometer readings with the expected direction of gravity

    if data.shape[1] > 3:  # retain gyro data
        data_to_return = np.array(
            [acc[:, 0] - g[:, 0], acc[:, 1] - g[:, 1], acc[:, 2] - g[:, 2], data[:, 3], data[:, 4],
             data[:, 5]]).transpose()
    elif data.shape[1] <= 3:
        data_to_return = np.array([acc[:, 0] - g[:, 0], acc[:, 1] - g[:, 1], acc[:, 2] - g[:, 2]]).transpose()
    return data_to_return


def compensate_gravity(ds):
    print("Removing Gravity component started", end="\r", flush=True)
    if 1 < len(ds) < 4:
        seq = ds
    else:
        seq = ds[:, :3]
    q = get_quaternions(seq);
    # since q is given in the form of [w, x, y, z] = [q[0], q[1], q[2], q[3]]
    # get expected direction of gravity
    g = np.zeros([len(q), 3])
    # get expected direction of gravity
    g[:, 0] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    g[:, 1] = 2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3])
    g[:, 2] = q[:, 0] * q[:, 0] - q[:, 1] * q[:, 1] - q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
    # compensate accelerometer readings with the expected direction of gravity
    print("Removing Gravity component completed...", end="\r", flush=True)

    return np.array([seq[:, 0] - g[:, 0], seq[:, 1] - g[:, 1], seq[:, 2] - g[:, 2]]).transpose()


def get_quaternions(seq):  # if panda dataframe is given
    tilt = Tilt(seq, as_angles=False)
    q = tilt.Q
    if len(np.where(np.isnan(q))[0]):
        q = pd.DataFrame(q).fillna(method='ffill').to_numpy()
    ## reutrn q in the format of [w, x, y, z]
    # permuatation = [0,1,2,3]
    # q = q[:, permutation]
    return q  # np.array([tilt.Q[:,0], tilt.Q[:,1]]).transpose()


def get_vertical_component(x, winsize):
    if len(x.shape) == 1:
        axis = 0
    elif len(x.shape) == 2:
        axis = 0
    elif len(x.shape) > 2:
        axis = 1

    vertcomp = [];
    i = 0;

    while i < len(x):
        a = x[i:i + winsize, :]
        # print("a.shape:{}".format(a.shape))
        if i - winsize > 0:
            v_ = np.nanmean(x[i - winsize:i, :], axis=axis)
            # v_norm = np.sqrt(np.mean(x[i:i+winsize]**2, axis=0))
            v_norm = np.nanmean(x[i - winsize:i, :] ** 2, axis=axis)
            thiswindow = list(((a * v_) / v_norm) * a)
        else:
            v_ = np.nanmean(x[i:i + winsize], axis=axis)
            # v_norm = np.sqrt(np.mean(x[i:]**2, axis=0))
            v_norm = np.nanmean(x[i:i + winsize, :] ** 2, axis=axis)
            thiswindow = list(((a * v_) / v_norm) * a)

        vertcomp.extend(thiswindow)
        progressBar(i, len(x), bar_length=20, msg="Obtaining Vertical Component")
        i += winsize  # 1
    return np.array(vertcomp)  # np.sqrt(np.sum(vertcomp**2, axis=1))


def get_horizontal_component(x, v):
    return np.array(x - v)

def pitch(acc): # rotation via x axis
    x = acc[:, 0]
    y = acc[: ,1]
    z = acc[:, 2]
    roll = np.arctan2(z, np.sqrt(x**2 + y**2))
    return roll * 180 / np.pi

def roll(acc):
    x = acc[:, 0]
    y = acc[: ,1]
    z = acc[:, 2]
    roll = np.arctan2(x, np.sqrt(y**2 + z**2))
    return roll * 180 / np.pi

def yaw(acc):
    x = acc[:, 0]
    y = acc[: ,1]
    z = acc[:, 2]
    roll = np.arctan2(y, np.sqrt(x**2 + y**2))
    return roll * 180 / np.pi
    

def ang_r(roll, pitch):
    obliquity = roll
    tilt = pitch
    ang_rss = np.sqrt(tilt**2 + obliquity**2)
    return ang_rss.reshape(-1, 1)

######################################## Deprecation ########################################
def resample_dep(timestamp, data, fs=100):
    print("Resampling started", end='\r', flush=True)
    ### sychronization by comparing timestamps of both datasets
    diff = np.diff(timestamp.astype(int))
    idx_sec_changed = np.where(diff == 1)[0]
    # fs = np.min(np.diff(np.where(np.diff(time.astype(int))!=0)[0]))

    # idx = np.linspace(0, len(time)-1, num=len(time)).astype(int)
    # to find indexes where second value changes not the millisecond by identiyfing indexes that are greater than fs (200Hz)
    idx_to_resample = idx_sec_changed[np.where(np.diff(idx_sec_changed) >= fs)[0]]
    # print("idx_to_resample:{}".format(idx_to_resample))
    if idx_to_resample.size == 0:  # when not null
        idx_to_resample = idx_sec_changed[:-1]

    idx_interval = []
    for i, item in enumerate(idx_to_resample, start=0):
        random.seed(cfg.RANDOM_STATE_SEED)
        # if found minimum freq is much smaller than initial sampling frequency, item + fs needs changing. Otherwise, the last part within a minute is discarded
        index_range = sorted(random.sample(range(item, item + cfg.SAMPLING_FREQUENCY), k=fs))
        idx_interval.append(index_range)
        progressBar(i, len(idx_to_resample), msg='Part 1 -- Resampling')

    idx_generated = np.array(idx_interval).reshape(-1)

    # print("len(idx_generated): {}, idx_generated[0]:{}, idx_generated[-1]:{}".format(len(idx_generated), idx_generated[0], idx_generated[-1]))
    # print("idx_generated:{}".format(idx_generated[:params.RESAMPLING_FREQUENCY]))

    resampled_timestamp = timestamp[idx_generated]
    resampled_data = data[idx_generated]  # .reset_index(drop=True)
    print("Resampling completed")

    return resampled_timestamp, resampled_data
