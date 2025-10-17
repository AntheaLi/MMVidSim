'''
Created on Fri Sep 29 14:39:17 2023

@author: Kendrick

@description: This script is for extracting/segmenting the dataset from the raw data.

'''
import h5py
import numpy as np
import os, glob, shutil
from copy import deepcopy
from utils.time_utils import extract_sortkey
from utils.data_utils import write_picke, get_data_from_period
from preprocess import *
from settings import *

src_dir = DATA_SETTINGS.src_dir
output_dir = DATA_SETTINGS.output_dir
normalized = DATA_SETTINGS.normalize
denoise = DATA_SETTINGS.denoise
myo_resample_rate = DATA_SETTINGS.myo_resample
gaze_resample_rate = DATA_SETTINGS.gaze_resample
joint_resample_rate = DATA_SETTINGS.joint_resample
tactile_resample_rate = DATA_SETTINGS.tactile_resample
second_per_samples = DATA_SETTINGS.second_per_samples
sample_per_class = DATA_SETTINGS.sample_per_class
left_hand_list_indices = LEFT_FINGER_TREESTR_LIST_INDICES
right_hand_list_indices = RIGHT_FINGER_TREESTR_LIST_INDICES
body_joint_list_indices = JOINT_TREESTR_LIST_INDICES
hand_start_indices = DATA_SETTINGS.hand_start_indices
resample_rates = {"emg":myo_resample_rate, "gaze": gaze_resample_rate, "joint": joint_resample_rate, "tactile": tactile_resample_rate}
tactile_aggregate = DATA_SETTINGS.tactile_aggregate

def create_samples():
    """ Extract from the raw data, apply processing, organize by the subject.
    """
    # Extract, denoise and normalized
    data_by_subject = extract_from_raw(src_dir, normalized, denoise, tactile_aggregate)

    # Resample
    data_by_subject = resample_stream(
        data_by_subject,
        myo_resample_rate,
        joint_resample_rate,
        tactile_resample_rate)

    # Segment the signal & label into samples.
    segment_data_by_subject = segment_stream(
        data_by_subject,
        second_per_samples,
        sample_per_class,
        resample_rates)
    return segment_data_by_subject

def split_data_subject(segment_data_by_subject:dict, valid_id:int) -> None:
    """Split the data to train/valid set.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    for subject in segment_data_by_subject:
        valid_flag = subject == valid_id
        for action, data in segment_data_by_subject[subject].items():
            label = LABEL_NAMES.index(action)
            for i, segment in enumerate(data):
                sample = {"label": label, "signal": segment, "label_name": action}
                split = "valid" if valid_flag else "train"
                output_folder = os.path.join(output_dir, split, subject, str(label))
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                output_path = os.path.join(output_folder, f"{i+1}.p")
                write_picke(output_path, sample)

def extract_from_raw(src_dir:str,
                     normalized:bool,
                     denoise:bool, 
                     tactile_aggregate:bool) -> dict:
    """ Read from hdf5, denoise and normalize.
    """
    # subject_lst = sorted(os.listdir(src_dir), key=lambda x:int(x[1:]))
    subject_lst = ['S00', 'S05'] #['S00',  'S02', 'S03', 'S04', 'S05']
    data_by_subject = {}

    # Extract & Denoise & Normalize for the data
    for subject in subject_lst:
        if subject not in data_by_subject:
            data_by_subject[subject] = {}
        filename_lst = glob.glob(os.path.join(src_dir, subject, "wearable_sensor", '**/*.hdf5'), recursive=True)
        filename_lst = sorted(filename_lst, key=extract_sortkey)

        for i, fpath in enumerate(filename_lst):
            print(f"Extracting {fpath}")
            data_by_subject[subject][i] = {"activities":{}}
            content = h5py.File(fpath, 'r')
            action, action_time, action_timestamp = extract_action(content)
            if len(action) == 0:
                print("Skip the empty activities record.")
                del data_by_subject[subject][i] # Remove the data if there is no label in the file.
                continue
            data_by_subject[subject][i]["activities"]["data"] = action
            data_by_subject[subject][i]["activities"]["time_s"] = action_time
            
            for tactile_key in ['tactile-glove-left', 'tactile-glove-right']:
                if tactile_key not in content:
                    del data_by_subject[subject][i] # Remove the data if there is no label in the file.
                    continue
                if tactile_key not in data_by_subject[subject][i]:
                    data_by_subject[subject][i][tactile_key] = {}
                tactile, tactile_time, tactile_timestamp = extract_tactile(content, tactile_key, denoise)
                if normalized:
                    tactile = tactile_normalize(tactile, tactile_aggregate)
                    # tactile = tactile_normalize_sequence(tactile)
                data_by_subject[subject][i][tactile_key]["data"] = tactile
                data_by_subject[subject][i][tactile_key]["time_s"] = tactile_time

            for myo_key in ['myo-left', 'myo-right']:
                if myo_key not in data_by_subject[subject][i]:
                    data_by_subject[subject][i][myo_key] = {}
                emg, emg_time, emg_timestamp = extract_emg(content, myo_key, denoise)
                if normalized:
                    emg = emg_normalize(emg)
                data_by_subject[subject][i][myo_key]["data"] = emg
                data_by_subject[subject][i][myo_key]["time_s"] = emg_time   
            
            data_by_subject[subject][i]["joint-rotation"] = {}
            joint_rotation_data, joint_rotation_time, joint_rotation_timestamp = extract_xsens_rotation(content)            
            if normalized:
                joint_rotation_data = joint_rotation_normalize(joint_rotation_data, 180, -180)
            data_by_subject[subject][i]["joint-rotation"]["data"] = joint_rotation_data
            data_by_subject[subject][i]["joint-rotation"]["time_s"] = joint_rotation_time

            data_by_subject[subject][i]["joint-position"] = {}
            joint_position_data, joint_position_time, joint_position_timestamp = extract_xsens_position(content, body_joint_list_indices) # , np.arange(22) --> without the tree structure
            data_by_subject[subject][i]["joint-position"]["data"] = joint_position_data
            data_by_subject[subject][i]["joint-position"]["time_s"] = joint_position_time

            
            data_by_subject[subject][i]["right-hand-pose"] = {}
            data_by_subject[subject][i]["left-hand-pose"] = {}
            left_hand_pose_data, hand_time, hand_time_stamp = parse_handpose_data(content, left_hand_list_indices, normalize=normalized)
            right_hand_pose_data, hand_time, hand_time_stamp = parse_handpose_data(content, right_hand_list_indices, normalize=normalized)
            data_by_subject[subject][i]["right-hand-pose"]["data"] = right_hand_pose_data
            data_by_subject[subject][i]["left-hand-pose"]["data"] = left_hand_pose_data
            data_by_subject[subject][i]["left-hand-pose"]["time_s"] = hand_time
            data_by_subject[subject][i]["right-hand-pose"]["time_s"] = hand_time

    content.close()
    return data_by_subject

def resample_stream(
        data_by_subject:dict,
        myo_resample_rate:int,
        joint_resample_rate:int,
        tactile_resample_rate:int
        ) -> dict:
    """ Resample the signal according given sample rate.
    """

    # Resample all the signal except for action channel
    for subject, subject_data in data_by_subject.items():
        print(f"Resampling {subject}")
        for file_index in subject_data:
            data = deepcopy(subject_data[file_index])
            for sensor_key in data:
                if sensor_key == "activities":
                    continue

                signal = data[sensor_key]["data"]
                time_s = data[sensor_key]["time_s"]
                
                if sensor_key in ["myo-left", "myo-right"]:
                    if myo_resample_rate == 0:
                        continue
                    signal_resampled, time_resampled = resample(signal, time_s, myo_resample_rate)
                    data[sensor_key]["data"] = signal_resampled
                    data[sensor_key]["time_s"] = time_resampled

                elif sensor_key in ["hand-pose-left", "hand-pose-right"]:
                    if myo_resample_rate == 0:
                        continue
                    signal_resampled, time_resampled = resample(signal, time_s, joint_resample_rate)
                    data[sensor_key]["data"] = signal_resampled
                    data[sensor_key]["time_s"] = time_resampled

                elif sensor_key in ["joint-position"]:
                    if myo_resample_rate == 0:
                        continue
                    signal_resampled, time_resampled = resample(signal, time_s, joint_resample_rate)
                    data[sensor_key]["data"] = signal_resampled
                    data[sensor_key]["time_s"] = time_resampled

                elif sensor_key in ["joint-rotation"]:
                    if joint_resample_rate == 0:
                        continue
                    signal_resampled, time_resampled = resample(signal, time_s, joint_resample_rate)
                    data[sensor_key]["data"] = signal_resampled[:, :22, :] #take only the 22 nodes for Xsens.
                    data[sensor_key]["time_s"] = time_resampled

                elif sensor_key in ["tactile-glove-left", "tactile-glove-right"]:
                    if tactile_resample_rate == 0:
                        continue
                    signal_resampled, time_resampled = resample(signal, time_s, tactile_resample_rate)                    
                    data[sensor_key]["data"] = signal_resampled
                    data[sensor_key]["time_s"] = time_resampled
                else:
                    print("Skip resampling...")

            data_by_subject[subject][file_index] = data
    return data_by_subject

def segment_stream(data_by_subject:dict, 
                   second_per_samples:int, 
                   sample_per_class:int, 
                   resample_rates:int) -> dict:
    segment_data_by_subject = {}
    for subject, subject_data in data_by_subject.items():
        if subject not in segment_data_by_subject:
            segment_data_by_subject[subject] = {}
        print("==="*10)
        print(f"Taking {subject}")
        segements = {}
        instance_count = {}
        for file_index in subject_data:
            
            act_start_time = []
            act_end_time = []
            act_labels = []

            data = deepcopy(subject_data[file_index])
            action_info = data["activities"]["data"]
            action_time = data["activities"]["time_s"]
            
            for i in range(len(action_info)):
                action_label = action_info[i, 0]
                is_start = action_info[i, 1] == "Start"
                is_stop = action_info[i, 1] == "Stop"
                rating = action_info[i, 3]

                if rating in ["Bad", "Maybe"]:
                    print("Drop samples due to bad rating...")
                    continue

                if action_label == 'Open a jar of almond butter':
                    action_label = 'Open/close a jar of almond butter'
                elif action_label == 'Get items from refrigerator/cabinets/drawers':
                    action_label = 'Get/replace items from refrigerator/cabinets/drawers'

                label = action_label
                if is_start:
                    act_labels.append(label)
                    act_start_time.append(action_time[i])
                
                if is_stop:
                    act_end_time.append(action_time[i])

            # segment those action with certain tasks.
            for i in range(len(act_labels)):
                label = act_labels[i]
                if label not in segements:
                    segements[label] = []
                    instance_count[label] = 0
                start = act_start_time[i]
                stop = act_end_time[i]
                assert stop > start
                instance_count[label] += 1 
                print(label)
                segements[label].extend(get_data_from_period(data, start, stop, sample_per_class, second_per_samples, resample_rates))

            # segment action which is no activity.
            for i in range(len(act_labels)):
                if i == len(act_labels) - 1:
                    continue
                label = LABEL_NAMES[0]
                if label not in segements:
                    segements[label] = []
                    instance_count[label] = 0
                start = act_end_time[i]
                stop = act_start_time[i+1]
                assert stop > start
                if stop - start < second_per_samples:
                    continue
                instance_count[label] += 1 
                segements[label].extend(get_data_from_period(data, start, stop, 10, second_per_samples, resample_rates))

        # Select the segments for creating the samples and exporting.
        for k, v in instance_count.items():
            print(f"{k}:{v}") # Show the number for each instance.
        
        for label in segements:
            print(f"Found instance for label {label}: {len(segements[label])}")
            if len(segements[label]) > sample_per_class:
                select_segments = []
                indice = np.linspace(0, len(segements[label])-1, num=sample_per_class, endpoint=True, dtype=int)
                indice = np.array(indice, dtype=int)
                for index in indice:
                    select_segments.append(segements[label][index])
            else:
                select_segments = segements[label]
            segment_data_by_subject[subject][label] = select_segments
    return segment_data_by_subject



if __name__ == "__main__":
    samples = create_samples()
    split_data_subject(samples, valid_id='All')