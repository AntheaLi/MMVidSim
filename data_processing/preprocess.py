'''
Created on Tue Oct 10 21:02:07 2023

@author: Kendrick

@description: This script contains some signal preprocessing approach.
'''
import numpy as np
import time
import pandas as pd
from scipy import interpolate
from scipy.signal import butter, lfilter
from utils.data_utils import action_decode
from utils.time_utils import str2timestamp

def lowpass_filter(data, cutoff, Fs, order=5):
  nyq = 0.5 * Fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  y = lfilter(b, a, data.T).T
  return y

def resample(data, time_s, resample_rate, kind="linear"):
    target_s = np.linspace(time_s[0],
                         time_s[-1],
                         int(round(1+resample_rate*(time_s[-1] - time_s[0]))),
                         endpoint=True)
    
    f = interpolate.interp1d(time_s, data, axis=0, kind=kind, fill_value='extrapolate')
    resampled_data = f(target_s)
    if np.any(np.isnan(resampled_data)):
        print('\n'*5)
        print('='*50)
        print('='*50)
        print('FOUND NAN')
        time.sleep(3)
    return resampled_data, target_s

def tactile_aggregate(tactile):
    num_rows = tactile.shape[1]
    num_cols = tactile.shape[2]
    num_tactile_rows_aggregated = 4
    num_tactile_cols_aggregated = 4

    row_stride = int(num_rows / num_tactile_rows_aggregated + 0.5)
    col_stride = int(num_rows / num_tactile_cols_aggregated + 0.5)
    data_aggregated = np.zeros(shape=(tactile.shape[0], num_tactile_rows_aggregated, num_tactile_cols_aggregated))
    for r, row_offset in enumerate(range(0, num_rows, row_stride)):
        for c, col_offset in enumerate(range(0, num_cols, col_stride)):
            mask = np.zeros(shape=(num_rows, num_cols))
            mask[row_offset:(row_offset+row_stride), col_offset:(col_offset+col_stride)] = 1
            data_aggregated[:,r,c] = np.sum(tactile*mask, axis=(1,2))/np.sum(mask)
    return data_aggregated

def normalize(sig, max_val=None, min_val=None):
    max_val = np.amax(sig) if max_val == None else max_val
    min_val = np.amin(sig) if min_val == None else min_val
    return (2 *(sig - min_val) / (max_val - min_val)) -1

def tactile_normalize(tactile, aggregate=False):
    """ Clip in a range, and normalize to [-1, 1]
    """
    mean_map = np.mean(tactile, axis=0)
    std_map = np.std(tactile, axis=0)
    # clip_low = mean_map - 2*std_map # shouldn't be much below the mean, since the mean should be rest basically
    # clip_high = mean_map + 3*std_map
    clip_low = 530
    clip_high = 600
    tactile = np.clip(tactile, clip_low, clip_high)

    return tactile / (clip_high - clip_low)

def tactile_normalize_sequence(tactile):
    sequnce_len, frame_size, frame_size = tactile.shape

    tactile = tactile / (np.linalg.norm(tactile.reshape(sequnce_len, frame_size * frame_size), axis=-1)[...,np.newaxis, np.newaxis] + 1e-5)

    return tactile

def joint_rotation_normalize(sig, max_val=180, min_val=-180):
    """ Normalize to [-1, 1] with min-max [-180, 180]
    """
    return sig / ((max_val - min_val)/2)

def emg_normalize(sig):
    """ Normalize to [-1, 1] with min-max normalization along axis.
    """
    max_vals = np.amax(sig, axis=0)
    min_vals = np.amin(sig, axis=0)
    return (sig - min_vals) / (max_vals - min_vals)

def extract_action(content):
    byte_action = content["experiment-activities"]["activities"]["data"][()]
    action_timestamp = content["experiment-activities"]["activities"]["time_str"][()].flatten()
    action_timestamp = np.array(list(map(str2timestamp, action_timestamp)))
    for j in range(byte_action.shape[1]):
        if j == 0:
            action = action_decode(byte_action[:, j])
        else:
            action = np.hstack((action, action_decode(byte_action[:, j])))
    action_time = content["experiment-activities"]["activities"]["time_s"][()].flatten()
    return action, action_time, action_timestamp

def extract_tactile(content, tactile_key, denoise=True):
    tactile = content[tactile_key]["tactile_data"]["data"][()]
    # Filter data
    if denoise:
        t = content[tactile_key]['tactile_data']['time_s']
        Fs = (t.size - 1) / (t[-1] - t[0])
        tactile = lowpass_filter(tactile, 2, Fs)
        tactile[0:int(Fs*30),:,:] = np.mean(tactile, axis=0)
        tactile[tactile.shape[0]-int(Fs*30):tactile.shape[0]+1,:,:] = np.mean(tactile, axis=0)

    tactile_timestamp = content[tactile_key]["tactile_data"]["time_str"][()].flatten()
    tactile_timestamp = np.array(list(map(str2timestamp, tactile_timestamp)))
    tactile_time = content[tactile_key]["tactile_data"]["time_s"][()].flatten()
    return tactile, tactile_time, tactile_timestamp

def extract_emg(content, myo_key, denoise=True):
    emg = content[myo_key]["emg"]["data"][()]
    # Filter data
    if denoise:
        t = content[myo_key]['emg']['time_s']
        Fs = (t.size - 1) / (t[-1] - t[0])
        emg = np.abs(emg)
        emg = lowpass_filter(emg, 5, Fs)

    emg_timestamp = content[myo_key]["emg"]["time_str"][()].flatten()
    emg_timestamp = np.array(list(map(str2timestamp, emg_timestamp)))
    emg_time = content[myo_key]['emg']['time_s'][()].flatten()
    return emg, emg_time, emg_timestamp

def extract_xsens_rotation(content, body_list_indices=None):
    if body_list_indices is not None:
        joint_rotation_data = content['xsens-joints']['rotation_xzy_deg']["data"][:, body_list_indices]
    else:
        joint_rotation_data = content['xsens-joints']['rotation_xzy_deg']["data"][()]

    joint_timestamp = content['xsens-joints']['rotation_xzy_deg']["time_str"][()].flatten()
    joint_timestamp = np.array(list(map(str2timestamp, joint_timestamp)))
    joint_time = content['xsens-joints']['rotation_xzy_deg']["time_s"][()].flatten()
    return joint_rotation_data, joint_time, joint_timestamp

def extract_xsens_position(h5_data, body_list_indices, normalize=True): 
    joint_position_data = np.array(h5_data['xsens-segments']['position_cm']["data"])[:, body_list_indices]
    xsense_com = np.array(h5_data['xsens-CoM']['position_cm']["data"])[()][:, np.newaxis, ...]
    if normalize: # first normalize the world coordiante
        joint_position_data = joint_position_data - xsense_com
    joint_timestamp = h5_data['xsens-segments']['position_cm']["time_str"][()].flatten()
    joint_timestamp = np.array(list(map(str2timestamp, joint_timestamp)))
    joint_time = h5_data['xsens-segments']['position_cm']['time_s'][()].flatten()

    return joint_position_data, joint_time, joint_timestamp

def parse_handpose_data(h5_data, hand_list_indices, return_tree=False, normalize=True):
    # visualize hand pose data
    hand_pose = np.array(h5_data['xsens-segments']['position_cm']['data'])[:, hand_list_indices]
    # if normalize: # first normalize the world coordiante

    handpose_time = h5_data['xsens-segments']['position_cm']['time_s'][()].flatten()
    handpose_timestamp = h5_data['xsens-segments']['position_cm']["time_str"][()].flatten()
    handpose_timestamp = np.array(list(map(str2timestamp, handpose_timestamp)))
    
    carpus = hand_pose[:, 0][:, np.newaxis, ...]
    # normalize to only capture how the fingers move
    if normalize: 
        hand_pose = hand_pose - hand_pose[:, 0][:, np.newaxis, ...].repeat(24, 1)
        # replace the actual carpus
        carpus = np.zeros((hand_pose.shape[0], 1, 3))
        hand_pose = np.concatenate([carpus, hand_pose], axis=1)
    
    return hand_pose, handpose_time, handpose_timestamp

def extract_eyes(content):
    gaze_data = content['eye-tracking-gaze']["position"]["data"][()]
    gaze_time = content['eye-tracking-gaze']["position"]["time_s"][()].flatten()
    Fs = (gaze_time.size - 1) / (gaze_time[-1] - gaze_time[0])
    clip_low = 0.05
    clip_high = 0.95
    gaze_data = np.clip(gaze_data, clip_low, clip_high)
    gaze_data[gaze_data == clip_low] = np.nan
    gaze_data[gaze_data == clip_high] = np.nan
    gaze_data = pd.DataFrame(gaze_data).interpolate(method='zero').to_numpy()

    gaze_data[np.isnan(gaze_data)] = 0.5
    gaze_data = lowpass_filter(gaze_data, 5, Fs)
    
    return gaze_data, gaze_time

