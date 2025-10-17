'''
Created on Wed Oct 04 19:22:33 2023

@author: Kendrick
'''
import json
import pickle
import numpy as np
import torch
import json
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence

def read_json(path):
    with open(path, "r") as fp:
        data = json.load(fp)
    return data

def write_json(path, data):
    with open(path, "w") as fp:
        json.dump(data, fp)

def read_pickle(path):
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    return data

def write_picke(path, data):
    with open(path, "wb") as fp:
        pickle.dump(data, fp)

def action_decode(data):
    # print(np.array(list(map(lambda b: b.decode(), data))))
    return np.array(list(map(lambda b: b.decode(), data))).reshape(-1,1)

def pad_sequence2d(data_lst:list, batch_first:bool=False):
    new_data_lst = []
    batch_size = len(data_lst)
    for i, data in enumerate(data_lst):
        w, h = data.shape[1], data.shape[2]
        data_length = data.shape[0]
        new_data_lst.append(data.view(data_length, -1))
    
    pad_data = pad_sequence(new_data_lst, batch_first)
    if not batch_first:
        pad_data = pad_data.view(-1, batch_size, w, h)
    else:
        pad_data = pad_data.view(batch_size, -1, w, h)
    return pad_data

def get_data_from_period(data, start, stop, count, duration, resample_rate) -> dict:
    segments = []
    data = deepcopy(data)
    segment_start_times = np.linspace(start, stop - duration, count, endpoint=True)
    for segment_start_time in segment_start_times:
        samples = {}
        segment_end_time = segment_start_time + duration

        for key in data:
            if key == "activities":
                continue
            if key in ["tactile-glove-left", "tactile-glove-right"]:
                sequent_len = resample_rate["tactile"] * duration
            elif key in ["joint-rotation", "joint-position", "left-hand-pose", "right-hand-pose"]:
                sequent_len = resample_rate["joint"] * duration
            elif key in ["myo-left", "myo-right"]:
                 sequent_len = resample_rate["emg"] * duration
            

            sensor_data = data[key]["data"]
            sensor_timestamp = data[key]["time_s"]
            time_indice = np.where((sensor_timestamp>=segment_start_time) & \
                                   (sensor_timestamp<=segment_end_time))[0].tolist()
            
            if len(time_indice) == 0:
                print(key)
                continue
            
            while len(time_indice) < sequent_len:

                if time_indice[0] > 0:
                    time_indice = [time_indice[0]-1] + time_indice
                elif time_indice[-1] < len(sensor_timestamp)-1:
                    time_indice.append(time_indice[-1]+1)
            
            while len(time_indice) > sequent_len:
                time_indice.pop()

            time_indice = np.array(time_indice)
            samples[key] = sensor_data[time_indice]
        segments.append(samples)
    return segments

def write_training_record(path, log):
    with open(path, "w") as fp:
        json.dump(log, fp, indent=4)

def read_training_record(path):
    with open(path, "r") as fp:
        data = json.load(fp)
    return data

def resample(current_data_len, resample_data_len, resample_type='avg'):
    if current_data_len < resample_data_len:
        print("illegal operation in resampling")
        return np.arange(current_data_len)
    
    if resample_type == 'avg':
        sample_freq = current_data_len // resample_data_len
        return np.arange(start=0, stop=current_data_len, step=sample_freq)

    
def parse_compact(signal, data):
    if signal == 'tactile':
        return np.concatenate([data['tactile-glove-left'][:, np.newaxis, ...], data['tactile-glove-right'][:, np.newaxis, ...]], axis=1)
    elif signal == 'myo':
        return np.concatenate([data['myo-left'], data['myo-right']], axis = -1)
    elif signal == 'hand-pose':
        return np.concatenate([data['left-hand-pose'], data['right-hand-pose']], axis= -2)
    else:
        return data[signal]


