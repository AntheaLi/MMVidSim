import numpy as np
import h5py
import os
import cv2
import sys
import skimage
import matplotlib.pyplot as plt
from scipy import interpolate # for the resampling example
from scipy.signal import butter, lfilter
from utils.data_utils import write_picke
from settings import LABEL_NAMES

start_ind = 0
PARSE_VIDEO=True
subject_ind = sys.argv[1]
all_subjects = [ '1', '3', '4', '5', '2/2_2', '2/2_0', '2/2_1']
subject = all_subjects[int(subject_ind)]
sample_rate = int(sys.argv[2]) if len(sys.argv)>2 else 2
RESIZE = int(sys.argv[3]) if len(sys.argv)>3 else 64
# sample_rate = 2


def resample_sequence(sequence_len, total_sequence_len):
    inds = []
    start_ind = 0
    while start_ind < (total_sequence_len - sequence_len * sample_rate):
        inds.append(np.arange(start_ind, total_sequence_len, sample_rate)[:sequence_len])
        start_ind += 1
    
    return inds

def lowpass_filter(data, cutoff, Fs, order=5):
  nyq = 0.5 * Fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  y = lfilter(b, a, data.T).T
  return y

# helper functions
def get_bodyskeleton_tree():
    segment_labels = [
    'Pelvis',
    'L5',
    'L3',
    'T12',
    'T8',
    'Neck',
    'Head',
    'Right Shoulder',
    'Right Upper Arm',
    'Right Forearm',
    'Right Hand',
    'Left Shoulder',
    'Left Upper Arm',
    'Left Forearm',
    'Left Hand',
    'Right Upper Leg',
    'Right Lower Leg',
    'Right Foot',
    'Right Toe',
    'Left Upper Leg',
    'Left Lower Leg',
    'Left Foot',
    'Left Toe',
    ]
    
    # Define how to visualize the person by connecting segment positions.
    segment_chains_labels_toPlot = {
      'Left Leg':  ['Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe'],
      'Right Leg': ['Right Upper Leg', 'Right Lower Leg', 'Right Foot', 'Right Toe'],
      'Spine':     ['Head', 'Neck', 'T8', 'T12', 'L3', 'L5', 'Pelvis'], 
      'Hip':       ['Left Upper Leg', 'Pelvis', 'Right Upper Leg'],
      'Shoulders': ['Left Upper Arm', 'Left Shoulder', 'Right Shoulder', 'Right Upper Arm'],
      'Left Arm':  ['Left Upper Arm', 'Left Forearm', 'Left Hand'],
      'Right Arm': ['Right Upper Arm', 'Right Forearm', 'Right Hand'],
    }

    all_items = [len(segment_chains_labels_toPlot[k]) for k in segment_chains_labels_toPlot]

    segment_chains_indexes_toPlot = dict()
    segment_list_indices = []
    for (chain_name, chain_labels) in segment_chains_labels_toPlot.items():
        segment_indexes = []
        for chain_label in chain_labels:
            segment_indexes.append(segment_labels.index(chain_label))
        segment_chains_indexes_toPlot[chain_name] = segment_indexes
        segment_list_indices.extend(segment_indexes)
    
    assert len(segment_list_indices) == sum(all_items), f'body: should have {sum(all_items)} items but found {len(segment_list_indices)} '

    return segment_list_indices, segment_chains_indexes_toPlot

def get_handskeleton_tree():
    # # load hand pose data
    # meta_data = dict(h5_data['xsens-segments']['position_cm'].attrs.items())
    # position_sequence_data = h5_data['xsens-segments']['position_cm']['data']

    meta_data_hand = '''Pelvis (x)', 'Pelvis (y)', 'Pelvis (z)', 'L5 (x)', 'L5 (y)', 'L5 (z)', 'L3 (x)', 'L3 (y)', 'L3 (z)', 'T12 (x)', 'T12 (y)', 'T12 (z)', 'T8 (x)', 'T8 (y)', 'T8 (z)', 'Neck (x)', 'Neck (y)', 'Neck (z)', 'Head (x)', 'Head (y)', 'Head (z)', 'Right Shoulder (x)', 'Right Shoulder (y)', 'Right Shoulder (z)', 'Right Upper Arm (x)', 'Right Upper Arm (y)', 'Right Upper Arm (z)', 'Right Forearm (x)', 'Right Forearm (y)', 'Right Forearm (z)', 'Right Hand (x)', 'Right Hand (y)', 'Right Hand (z)', 'Left Shoulder (x)', 'Left Shoulder (y)', 'Left Shoulder (z)', 'Left Upper Arm (x)', 'Left Upper Arm (y)', 'Left Upper Arm (z)', 'Left Forearm (x)', 'Left Forearm (y)', 'Left Forearm (z)', 'Left Hand (x)', 'Left Hand (y)', 'Left Hand (z)', 'Right Upper Leg (x)', 'Right Upper Leg (y)', 'Right Upper Leg (z)', 'Right Lower Leg (x)', 'Right Lower Leg (y)', 'Right Lower Leg (z)', 'Right Foot (x)', 'Right Foot (y)', 'Right Foot (z)', 'Right Toe (x)', 'Right Toe (y)', 'Right Toe (z)', 'Left Upper Leg (x)', 'Left Upper Leg (y)', 'Left Upper Leg (z)', 'Left Lower Leg (x)', 'Left Lower Leg (y)', 'Left Lower Leg (z)', 'Left Foot (x)', 'Left Foot (y)', 'Left Foot (z)', 'Left Toe (x)', 'Left Toe (y)', 'Left Toe (z)', 'Left Carpus (x)', 'Left Carpus (y)', 'Left Carpus (z)', 'Left First Metacarpal (x)', 'Left First Metacarpal (y)', 'Left First Metacarpal (z)', 'Left First Proximal Phalange (x)', 'Left First Proximal Phalange (y)', 'Left First Proximal Phalange (z)', 'Left First Distal Phalange (x)', 'Left First Distal Phalange (y)', 'Left First Distal Phalange (z)', 'Left Second Metacarpal (x)', 'Left Second Metacarpal (y)', 'Left Second Metacarpal (z)', 'Left Second Proximal Phalange (x)', 'Left Second Proximal Phalange (y)', 'Left Second Proximal Phalange (z)', 'Left Second Middle Phalange (x)', 'Left Second Middle Phalange (y)', 'Left Second Middle Phalange (z)', 'Left Second Distal Phalange (x)', 'Left Second Distal Phalange (y)', 'Left Second Distal Phalange (z)', 'Left Third Metacarpal (x)', 'Left Third Metacarpal (y)', 'Left Third Metacarpal (z)', 'Left Third Proximal Phalange (x)', 'Left Third Proximal Phalange (y)', 'Left Third Proximal Phalange (z)', 'Left Third Middle Phalange (x)', 'Left Third Middle Phalange (y)', 'Left Third Middle Phalange (z)', 'Left Third Distal Phalange (x)', 'Left Third Distal Phalange (y)', 'Left Third Distal Phalange (z)', 'Left Fourth Metacarpal (x)', 'Left Fourth Metacarpal (y)', 'Left Fourth Metacarpal (z)', 'Left Fourth Proximal Phalange (x)', 'Left Fourth Proximal Phalange (y)', 'Left Fourth Proximal Phalange (z)', 'Left Fourth Middle Phalange (x)', 'Left Fourth Middle Phalange (y)', 'Left Fourth Middle Phalange (z)', 'Left Fourth Distal Phalange (x)', 'Left Fourth Distal Phalange (y)', 'Left Fourth Distal Phalange (z)', 'Left Fifth Metacarpal (x)', 'Left Fifth Metacarpal (y)', 'Left Fifth Metacarpal (z)', 'Left Fifth Proximal Phalange (x)', 'Left Fifth Proximal Phalange (y)', 'Left Fifth Proximal Phalange (z)', 'Left Fifth Middle Phalange (x)', 'Left Fifth Middle Phalange (y)', 'Left Fifth Middle Phalange (z)', 'Left Fifth Distal Phalange (x)', 'Left Fifth Distal Phalange (y)', 'Left Fifth Distal Phalange (z)', 'Right Carpus (x)', 'Right Carpus (y)', 'Right Carpus (z)', 'Right First Metacarpal (x)', 'Right First Metacarpal (y)', 'Right First Metacarpal (z)', 'Right First Proximal Phalange (x)', 'Right First Proximal Phalange (y)', 'Right First Proximal Phalange (z)', 'Right First Distal Phalange (x)', 'Right First Distal Phalange (y)', 'Right First Distal Phalange (z)', 'Right Second Metacarpal (x)', 'Right Second Metacarpal (y)', 'Right Second Metacarpal (z)', 'Right Second Proximal Phalange (x)', 'Right Second Proximal Phalange (y)', 'Right Second Proximal Phalange (z)', 'Right Second Middle Phalange (x)', 'Right Second Middle Phalange (y)', 'Right Second Middle Phalange (z)', 'Right Second Distal Phalange (x)', 'Right Second Distal Phalange (y)', 'Right Second Distal Phalange (z)', 'Right Third Metacarpal (x)', 'Right Third Metacarpal (y)', 'Right Third Metacarpal (z)', 'Right Third Proximal Phalange (x)', 'Right Third Proximal Phalange (y)', 'Right Third Proximal Phalange (z)', 'Right Third Middle Phalange (x)', 'Right Third Middle Phalange (y)', 'Right Third Middle Phalange (z)', 'Right Third Distal Phalange (x)', 'Right Third Distal Phalange (y)', 'Right Third Distal Phalange (z)', 'Right Fourth Metacarpal (x)', 'Right Fourth Metacarpal (y)', 'Right Fourth Metacarpal (z)', 'Right Fourth Proximal Phalange (x)', 'Right Fourth Proximal Phalange (y)', 'Right Fourth Proximal Phalange (z)', 'Right Fourth Middle Phalange (x)', 'Right Fourth Middle Phalange (y)', 'Right Fourth Middle Phalange (z)', 'Right Fourth Distal Phalange (x)', 'Right Fourth Distal Phalange (y)', 'Right Fourth Distal Phalange (z)', 'Right Fifth Metacarpal (x)', 'Right Fifth Metacarpal (y)', 'Right Fifth Metacarpal (z)', 'Right Fifth Proximal Phalange (x)', 'Right Fifth Proximal Phalange (y)', 'Right Fifth Proximal Phalange (z)', 'Right Fifth Middle Phalange (x)', 'Right Fifth Middle Phalange (y)', 'Right Fifth Middle Phalange (z)', 'Right Fifth Distal Phalange (x)', 'Right Fifth Distal Phalange (y)', 'Right Fifth Distal Phalange (z)'''
    # meta_data_hand = '''Left Carpus (x)', 'Left Carpus (y)', 'Left Carpus (z)', 'Left First Metacarpal (x)', 'Left First Metacarpal (y)', 'Left First Metacarpal (z)', 'Left First Proximal Phalange (x)', 'Left First Proximal Phalange (y)', 'Left First Proximal Phalange (z)', 'Left First Distal Phalange (x)', 'Left First Distal Phalange (y)', 'Left First Distal Phalange (z)', 'Left Second Metacarpal (x)', 'Left Second Metacarpal (y)', 'Left Second Metacarpal (z)', 'Left Second Proximal Phalange (x)', 'Left Second Proximal Phalange (y)', 'Left Second Proximal Phalange (z)', 'Left Second Middle Phalange (x)', 'Left Second Middle Phalange (y)', 'Left Second Middle Phalange (z)', 'Left Second Distal Phalange (x)', 'Left Second Distal Phalange (y)', 'Left Second Distal Phalange (z)', 'Left Third Metacarpal (x)', 'Left Third Metacarpal (y)', 'Left Third Metacarpal (z)', 'Left Third Proximal Phalange (x)', 'Left Third Proximal Phalange (y)', 'Left Third Proximal Phalange (z)', 'Left Third Middle Phalange (x)', 'Left Third Middle Phalange (y)', 'Left Third Middle Phalange (z)', 'Left Third Distal Phalange (x)', 'Left Third Distal Phalange (y)', 'Left Third Distal Phalange (z)', 'Left Fourth Metacarpal (x)', 'Left Fourth Metacarpal (y)', 'Left Fourth Metacarpal (z)', 'Left Fourth Proximal Phalange (x)', 'Left Fourth Proximal Phalange (y)', 'Left Fourth Proximal Phalange (z)', 'Left Fourth Middle Phalange (x)', 'Left Fourth Middle Phalange (y)', 'Left Fourth Middle Phalange (z)', 'Left Fourth Distal Phalange (x)', 'Left Fourth Distal Phalange (y)', 'Left Fourth Distal Phalange (z)', 'Left Fifth Metacarpal (x)', 'Left Fifth Metacarpal (y)', 'Left Fifth Metacarpal (z)', 'Left Fifth Proximal Phalange (x)', 'Left Fifth Proximal Phalange (y)', 'Left Fifth Proximal Phalange (z)', 'Left Fifth Middle Phalange (x)', 'Left Fifth Middle Phalange (y)', 'Left Fifth Middle Phalange (z)', 'Left Fifth Distal Phalange (x)', 'Left Fifth Distal Phalange (y)', 'Left Fifth Distal Phalange (z)', 'Right Carpus (x)', 'Right Carpus (y)', 'Right Carpus (z)', 'Right First Metacarpal (x)', 'Right First Metacarpal (y)', 'Right First Metacarpal (z)', 'Right First Proximal Phalange (x)', 'Right First Proximal Phalange (y)', 'Right First Proximal Phalange (z)', 'Right First Distal Phalange (x)', 'Right First Distal Phalange (y)', 'Right First Distal Phalange (z)', 'Right Second Metacarpal (x)', 'Right Second Metacarpal (y)', 'Right Second Metacarpal (z)', 'Right Second Proximal Phalange (x)', 'Right Second Proximal Phalange (y)', 'Right Second Proximal Phalange (z)', 'Right Second Middle Phalange (x)', 'Right Second Middle Phalange (y)', 'Right Second Middle Phalange (z)', 'Right Second Distal Phalange (x)', 'Right Second Distal Phalange (y)', 'Right Second Distal Phalange (z)', 'Right Third Metacarpal (x)', 'Right Third Metacarpal (y)', 'Right Third Metacarpal (z)', 'Right Third Proximal Phalange (x)', 'Right Third Proximal Phalange (y)', 'Right Third Proximal Phalange (z)', 'Right Third Middle Phalange (x)', 'Right Third Middle Phalange (y)', 'Right Third Middle Phalange (z)', 'Right Third Distal Phalange (x)', 'Right Third Distal Phalange (y)', 'Right Third Distal Phalange (z)', 'Right Fourth Metacarpal (x)', 'Right Fourth Metacarpal (y)', 'Right Fourth Metacarpal (z)', 'Right Fourth Proximal Phalange (x)', 'Right Fourth Proximal Phalange (y)', 'Right Fourth Proximal Phalange (z)', 'Right Fourth Middle Phalange (x)', 'Right Fourth Middle Phalange (y)', 'Right Fourth Middle Phalange (z)', 'Right Fourth Distal Phalange (x)', 'Right Fourth Distal Phalange (y)', 'Right Fourth Distal Phalange (z)', 'Right Fifth Metacarpal (x)', 'Right Fifth Metacarpal (y)', 'Right Fifth Metacarpal (z)', 'Right Fifth Proximal Phalange (x)', 'Right Fifth Proximal Phalange (y)', 'Right Fifth Proximal Phalange (z)', 'Right Fifth Middle Phalange (x)', 'Right Fifth Middle Phalange (y)', 'Right Fifth Middle Phalange (z)', 'Right Fifth Distal Phalange (x)', 'Right Fifth Distal Phalange (y)', 'Right Fifth Distal Phalange (z)'''
    meta_data_hand = meta_data_hand.replace("\'", "")
    meta_data_hand = meta_data_hand.replace(' (x)', "")
    meta_data_hand = meta_data_hand.split(', ')
    meta_data_hand = [x for x in meta_data_hand if not x.endswith(')')]
    hands_data_start_index = meta_data_hand.index('Left Carpus')
    # print(meta_data_hand)
    # print(hands_data_start_index)


    hand_segments_labels = [
    'Carpus',
    'First Metacarpal',
    'First Proximal Phalange',
    'First Distal Phalange',
    'Second Metacarpal',
    'Second Proximal Phalange',
    'Second Middle Phalange',
    'Second Distal Phalange',
    'Third Metacarpal',
    'Third Proximal Phalange',
    'Third Middle Phalange',
    'Third Distal Phalange',
    'Fourth Metacarpal',
    'Fourth Proximal Phalange',
    'Fourth Middle Phalange',
    'Fourth Distal Phalange',
    'Fifth Metacarpal',
    'Fifth Proximal Phalange',
    'Fifth Middle Phalange',
    'Fifth Distal Phalange',
    ]

    # to transfer to mano hand it needs an interpolation between 'First Metacarpal','First Proximal Phalange' to get another joint
    segment_chains_indexes_toPlot = {
    # 'palm': ['Carpus', 'First Metacarpal', 'Second Metacarpal', 'Third Metacarpal', 'Fourth Metacarpal', 'Fifth Metacarpal'],
    'thumb': ['Carpus', 'First Metacarpal','First Proximal Phalange', 'First Distal Phalange'],
    'index': ['Carpus', 'Second Metacarpal','Second Proximal Phalange', 'Second Middle Phalange', 'Second Distal Phalange'],
    'middle': ['Carpus', 'Third Metacarpal', 'Third Proximal Phalange', 'Third Middle Phalange', 'Third Distal Phalange'],
    'ring': ['Carpus', 'Fourth Metacarpal', 'Fourth Proximal Phalange', 'Fourth Middle Phalange', 'Fourth Distal Phalange'],
    'pinky': ['Carpus', 'Fifth Metacarpal', 'Fifth Proximal Phalange', 'Fifth Middle Phalange', 'Fifth Distal Phalange'],
    }

    all_items = [len(segment_chains_indexes_toPlot[k]) for k in segment_chains_indexes_toPlot]

    # Make chains for the left and right sides.
    hand_segments_labels_left  = ['Left %s' % segment_label for segment_label in hand_segments_labels]
    hand_segments_labels_right = ['Right %s' % segment_label for segment_label in hand_segments_labels]

    hand_segment_chains_labels_tree_left_indices = {}
    hand_segment_chains_labels_tree_right_indices = {}
    hand_segment_chains_labels_list_left_indices = []
    hand_segment_chains_labels_list_right_indices = []
    hand_segment_chains_labels_toPlot_left_name = {}
    hand_segment_chains_labels_toPlot_right_name = {}
    for chain_name, segment_labels in segment_chains_indexes_toPlot.items():
        hand_segment_chains_labels_toPlot_left_name[chain_name] = ['Left %s' % segment_label for segment_label in segment_labels]
        hand_segment_chains_labels_toPlot_right_name[chain_name] = ['Right %s' % segment_label for segment_label in segment_labels]

    for chain_name, chain_labels in hand_segment_chains_labels_toPlot_left_name.items():
        segment_indexes = []
        for chain_label in chain_labels:
            segment_indexes.append(meta_data_hand.index(chain_label))
        hand_segment_chains_labels_tree_left_indices[chain_name] = segment_indexes
        hand_segment_chains_labels_list_left_indices.extend(segment_indexes)
    

    for chain_name, chain_labels in hand_segment_chains_labels_toPlot_right_name.items():
        segment_indexes = []
        for chain_label in chain_labels:
            segment_indexes.append(meta_data_hand.index(chain_label))
        hand_segment_chains_labels_tree_right_indices[chain_name] = segment_indexes
        hand_segment_chains_labels_list_right_indices.extend(segment_indexes)


    assert len(hand_segment_chains_labels_list_right_indices) == sum(all_items), f'right hand: should have {sum(all_items)} items but found {len(hand_segment_chains_labels_list_right_indices)} '
    assert len(hand_segment_chains_labels_list_left_indices) == sum(all_items), f'left hand: should have {sum(all_items)} items but found {len(hand_segment_chains_labels_list_right_indices)} '

    # print(hand_segment_chains_labels_toPlot_left_indices)
    # print(hand_segment_chains_labels_toPlot_right_indices)
    
    return hand_segment_chains_labels_tree_left_indices, hand_segment_chains_labels_list_left_indices, hand_segment_chains_labels_tree_right_indices, hand_segment_chains_labels_list_right_indices

def tactile_normalize_sequence(tactile):
    sequnce_len, frame_size = tactile.shape

    tactile = tactile / (np.linalg.norm(tactile.reshape(sequnce_len, frame_size * frame_size), axis=-1)[...,np.newaxis, np.newaxis] + 1e-5)

    return tactile

def tactile_normalize(tactile):
    """ Clip in a range, and normalize to [-1, 1]
    """
    mean_map = np.mean(tactile, axis=0)
    std_map = np.std(tactile, axis=0)
    clip_low = 530
    clip_high = 600
    tactile = np.clip(tactile, clip_low, clip_high)

    tactile = (tactile - clip_low) / (clip_high - clip_low)
    # print('normalized', tactile.shape)
    return tactile

def parse_pose(list_indices, chains_indices, position_sequence_data, return_tree=False):
    if return_tree:
        fig, axs = plt.subplots(nrows=1, ncols=2,
                                squeeze=False, # if False, always return 2D array of axes
                                # sharex=True, sharey=True,
                                subplot_kw={
                                'projection': '3d',
                                },
                                figsize=(7, 5)
                                )
        ax_scatter = axs[0][0]
        ax_skeleton = axs[0][1]
    
        # visualize one frame
        frame_index = 0

        pose = {}
        # parse the latest segment positions.
        segment_positions_cm = np.array(position_sequence_data[frame_index])
        for (chain_index, chain_name) in enumerate(chains_indices.keys()):
            segment_indexes = chains_indices[chain_name]
            segment_xyz_cm = segment_positions_cm[segment_indexes, :]
            pose[chain_index] = segment_xyz_cm
            ax_scatter.scatter3D(segment_xyz_cm[:,0], segment_xyz_cm[:,1], segment_xyz_cm[:,2])

            for i in range(segment_xyz_cm.shape[0]-1) : 
                ax_skeleton.plot([segment_xyz_cm[i, 0], segment_xyz_cm[i+1, 0]], [segment_xyz_cm[i, 1] , segment_xyz_cm[i+1, 1]],zs=[segment_xyz_cm[i, 2],segment_xyz_cm[i+1, 2]])
    else:
        # parse the latest segment positions.
        pose = np.array(position_sequence_data[:, list_indices])
    
    return pose

def parse_handpose_data(h5_data, return_tree=False, normalize=True):
    # visualize hand pose data
    hand_segment_chains_labels_toPlot_left_indices, hand_segment_chains_labels_list_left_indices, hand_segment_chains_labels_toPlot_right_indices, hand_segment_chains_labels_list_right_indices = get_handskeleton_tree()
    position_sequence_data = h5_data['xsens-segments']['position_cm']['data'][:]
    time_s = np.array(h5_data['xsens-segments']['position_cm']['time_s'])
    
    left_hand_pose = parse_pose(hand_segment_chains_labels_list_left_indices, hand_segment_chains_labels_toPlot_left_indices, position_sequence_data)
    right_hand_pose = parse_pose(hand_segment_chains_labels_list_right_indices, hand_segment_chains_labels_toPlot_right_indices, position_sequence_data)
    
    left_carpus = left_hand_pose[:, 0][:, np.newaxis, ...]
    right_carpus = right_hand_pose[:, 0][:, np.newaxis, ...]

    # normalize to only capture how the fingers move
    if normalize: 
        left_hand_pose = left_hand_pose - left_carpus
        right_hand_pose = right_hand_pose - right_carpus
        left_hand_pose_min = left_hand_pose.reshape(-1, 3).min(axis=0)
        left_hand_pose_max = left_hand_pose.reshape(-1, 3).max(axis=0)

        right_hand_pose_min = right_hand_pose.reshape(-1, 3).min(axis=0)
        right_hand_pose_max = right_hand_pose.reshape(-1, 3).max(axis=0)

        left_hand_pose = (left_hand_pose - left_hand_pose_min) / (left_hand_pose_max - left_hand_pose_min)
        right_hand_pose = (right_hand_pose - right_hand_pose_min) / (right_hand_pose_max - right_hand_pose_min)

    return left_hand_pose, right_hand_pose, time_s

def parse_bodypose_data(h5_data, return_tree=False, normalize=True):
    # visualize hand pose data
    segment_list_indices, segment_chains_indices = get_bodyskeleton_tree()
    position_sequence_data = h5_data['xsens-segments']['position_cm']['data'][:, 0:]
    position_com = h5_data['xsens-CoM']['position_cm']['data'][()][:, np.newaxis, ...]
    assert position_com.shape[0] == position_sequence_data.shape[0], f'center of mass does not match data'
    time_s = np.array(h5_data['xsens-segments']['position_cm']['time_s'])

    
    body_pose = parse_pose(segment_list_indices, segment_chains_indices, position_sequence_data)

    if normalize:
        body_pose = body_pose - position_com
        data_min = body_pose.reshape(-1, 3).min(axis=0)[np.newaxis, np.newaxis, ...]
        data_max = body_pose.reshape(-1, 3).max(axis=0)[np.newaxis, np.newaxis, ...]
        # body_pose = np.clip(body_pose, data_min, data_max)
        body_pose = (body_pose - data_min) / (data_max - data_min)

    return body_pose, time_s

def parse_tactile_readings(h5_data, normalize=True):
    right_hand_data = h5_data['tactile-glove-right']['tactile_data']['data']
    if normalize: right_hand_data = tactile_normalize(right_hand_data)
    right_time_s = np.array(h5_data['tactile-glove-right']['tactile_data']['time_s'])
    left_hand_data = h5_data['tactile-glove-left']['tactile_data']['data']

    if normalize: left_hand_data = tactile_normalize(left_hand_data)
    left_time_s = np.array(h5_data['tactile-glove-left']['tactile_data']['time_s'])

    if left_time_s.shape[0] < right_time_s.shape[0]:
        right_hand_data_resampled_index = rough_mapping(left_time_s, right_time_s, np.arange(right_hand_data.shape[0])).reshape(-1)
        right_hand_data = right_hand_data[right_hand_data_resampled_index]
        time_s = left_time_s
    else:
        left_hand_data_resampled_index = rough_mapping(right_time_s, left_time_s, np.arange(left_hand_data.shape[0])).reshape(-1)
        left_hand_data = left_hand_data[left_hand_data_resampled_index]
        time_s = right_time_s
        
    # plt.imshow(left_hand_data[0])
    # plt.title(f'{left_hand_data.shape[0] } frames')
    return left_hand_data, right_hand_data, time_s



def parse_video_data(path, start_frames, end_frames):
    # load video data
    print(f'process video data for {path}')
    cap = cv2.VideoCapture(path)
    id = 0
    count = start_frames[id]
    # img_frames_720 = {i:[] for i in range(len(start_frames))}
    # img_frames_256 = {i:[] for i in range(len(start_frames))}
    # img_frames_128 = {i:[] for i in range(len(start_frames))}
    # img_frames_64 = {i:[] for i in range(len(start_frames))}
    # img_frames_32 = {i:[] for i in range(len(start_frames))}
    img_frames = {i:[] for i in range(len(start_frames))}


    if path.endswith('.mp4'):
        w = 720
        h = 720
    else:
        w = 1200 #720
        h = 1200 #720

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frames[0])
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:

            if count < end_frames[id]:    
                center = frame.shape
                x = center[1]/2 - w/2
                y = center[0]/2 - h/2

                crop_img = frame[int(y):int(y+h), int(x):int(x+w)]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                # resized_720 = crop_img
                resized = cv2.resize(crop_img, [RESIZE, RESIZE], interpolation = cv2.INTER_AREA)
                img_frames[id].append(resized)


                # img_frames_720[id].append(crop_img)
                # # img_frames_128[id].append(resized_128)


                count += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            else:

                id += 1
                if id >= len(start_frames):
                    cap.release()
                    break
                count = start_frames[id]
                cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            cap.release()
            break

    # img_frames_720_dict = {i: np.array(img_frames_720[i]) for i in range(len(start_frames))}
    img_frames_dict = {i: np.array(img_frames[i]) for i in range(len(start_frames))}

    return img_frames_dict

def parse_timestamps_for_video_data(h5_data, txt_path=None):
    # parse video timestamps for activities
    timestamps = []
    if txt_path is not None:
        print('parse third-eye view video data')

        with open(txt_path, 'r') as f:
            for i, l in enumerate(f.readlines()):
                ts = l.rstrip().replace(' ', '.')
                ts = float(ts)
                timestamps.append(ts)
        timestamps = np.array(timestamps)
    else:
        print('parse egocentric video data')

        timestamps = h5_data['eye-tracking-video-world']['frame_timestamp']['data']

    activities_labels, activities_start_times_s, activities_end_times_s = get_all_activity_labels(h5_data)
    assert len(activities_labels) == len(activities_start_times_s) == len(activities_end_times_s), f'activity inconsistency: {len(activities_labels), len(activities_start_times_s), len(activities_end_times_s)}'

    start_frames = []
    end_frames = []
    id_s = 0
    id_e = 0
    start_time = activities_start_times_s[id_s]
    end_time = activities_end_times_s[id_e]

    for i, t in enumerate(timestamps):
        t = float(t)

        if id_e >= len(activities_labels):
            break
        
        if id_s < len(activities_labels): 
            start_time = activities_start_times_s[id_s]
            if t > start_time:    
                start_frames.append(i)
                id_s += 1

        end_time = activities_end_times_s[id_e]
        
        if t > end_time:
            end_frames.append(i-1)
            id_e += 1


    if len(end_frames) == len(start_frames) - 1:
        end_frames.append(-1)
    
    assert len(start_frames) == len(end_frames) == len(activities_labels), f'video frame inconsistency: {len(start_frames), len(end_frames), len(activities_labels)}'

    video_activity_timestamps = {i: timestamps[start_frames[i]:end_frames[i]] for i in range(len(activities_labels))}


    return start_frames, end_frames, video_activity_timestamps

def parse_gaze_data(h5_data, gaze_key='position', normalize=True):
    # valid keys are: 'eye_center_3d', 'normal_3d', 'point_3d', 'position', 
    assert gaze_key in ['eye_center_3d', 'normal_3d', 'point_3d', 'position'], f'illegal gaze key access {gaze_key}'
    gaze_data = h5_data['eye-tracking-gaze'][gaze_key]['data'][()]

    if normalize:
        if gaze_key == 'position': gaze_data = np.clip(gaze_data, 0.0, 1.0)

    gaze_time = h5_data['eye-tracking-gaze'][gaze_key]['time_s'][()].flatten()
    return gaze_data, gaze_time

def parse_myo_emg(content, myo_key, denoise=True, normalize=True):
    emg = content[myo_key]["emg"]["data"][()]

    if normalize:
        max_vals = emg.max()
        min_vals = emg.min()
        emg = (emg - min_vals) / (max_vals - min_vals)

    emg_time = content[myo_key]['emg']['time_s'][()].flatten()
    return emg, emg_time

def parse_myo_acc(content, myo_key, denoise=True, normalize=True):
    acc_g = content[myo_key]["acceleration_g"]["data"][()]

    if normalize:
        max_vals = acc_g.reshape(-1, 3).max(axis=0)[np.newaxis, ...]
        min_vals = acc_g.reshape(-1, 3).min(axis=0)[np.newaxis, ...]
        acc_g = (acc_g - min_vals) / (max_vals - min_vals)

    acc_g_time = content[myo_key]['acceleration_g']['time_s'][()].flatten()
    return acc_g, acc_g_time

def rough_mapping(referrence_time_s, to_sample_time_s, to_sample_data):
    fn_interpolate_acceleration = interpolate.interp1d(
                                to_sample_time_s.reshape(-1), # x values
                                to_sample_data,   # y values
                                axis=0,              # axis of the data along which to interpolate
                                kind='linear',       # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
                                fill_value='extrapolate' # how to handle x values outside the original range
                                )
    data_resampled = fn_interpolate_acceleration(referrence_time_s).astype(np.int64)
    data_resampled = data_resampled[np.where((data_resampled >= 0) & (data_resampled < to_sample_data.max()))[0]]

    return data_resampled


def get_all_activity_labels(h5_data):
    activity_datas = h5_data['experiment-activities']['activities']['data']
    activity_times_s = h5_data['experiment-activities']['activities']['time_s']
    activity_times_s = np.squeeze(np.array(activity_times_s))  # squeeze (optional) converts from a list of single-element lists to a 1D list
    # Convert to strings for convenience.
    activity_datas = [[x.decode('utf-8') for x in datas] for datas in activity_datas]

    exclude_bad_labels = True # some activities may have been marked as 'Bad' or 'Maybe' by the experimenter; submitted notes with the activity typically give more information
    activities_labels = []
    activities_start_times_s = []
    activities_end_times_s = []
    activities_ratings = []
    activities_notes = []
    for (row_index, time_s) in enumerate(activity_times_s):
        label    = activity_datas[row_index][0]
        is_start = activity_datas[row_index][1] == 'Start'
        is_stop  = activity_datas[row_index][1] == 'Stop'
        rating   = activity_datas[row_index][2]
        notes    = activity_datas[row_index][3]
        if exclude_bad_labels and rating in ['Bad', 'Maybe']:
            continue
        # Record the start of a new activity.
        if is_start:
            activities_labels.append(label)
            activities_start_times_s.append(time_s)
            activities_ratings.append(rating)
            activities_notes.append(notes)
        # Record the end of the previous activity.
        if is_stop:
            activities_end_times_s.append(time_s)
    return activities_labels, activities_start_times_s, activities_end_times_s


if __name__ == '__main__':
    base_dir = './raw-data'
    video_dir = 'video' # video 
    save_dir = f'Dataset_sampler{sample_rate}'
    all_subjects = [ '1', '3', '4', '5', '2/2_2', '2/2_0', '2/2_1']
    all_h5_paths = []
    all_video_paths = []
    all_video_time_paths = []
    test = subject
    print(f'parsing subject {subject} to save at ./{save_dir}/S0{subject[0]}/')


    # for test in all_subjects:
    if True:
        d = [x for x in os.listdir(f'{base_dir}/{test}/sensors/') if x.endswith('hdf5')]
        all_h5_paths.append(f'{base_dir}/{test}//sensors/{d[0]}')
        if PARSE_VIDEO:
            v = [x for x in os.listdir(f'{base_dir}/{test}/{video_dir}/') if x.endswith('.mp4')]
            all_video_paths.append(f'{base_dir}/{test}/{video_dir}/{v[0]}')
            txt = [x for x in os.listdir(f'{base_dir}/{test}/{video_dir}/') if x.endswith('.txt')]
            all_video_time_paths.append(f'{base_dir}/{test}/{video_dir}/{txt[0]}') if len(txt) > 0 else all_video_time_paths.append(None)
            # assert len(all_h5_paths) == len(all_video_paths) == len(all_subjects), f'{len(all_h5_paths)} {len(all_video_paths)} {len(all_subjects)}'

    
    for i in range(len(all_h5_paths)):
        print(all_h5_paths[i])
        subject_suffix = all_subjects[i].replace('/', '_')
            
        # if os.path.exists(f'./{save_dir}/clean_a_pan_with_a_sponge/{video_dir}_256_{subject_suffix}.npz'):
        #     print('exits, continue to next subject')
        #     continue

        h5_data = h5py.File(all_h5_paths[i], "r")

        activities_labels, activities_start_times_s, activities_end_times_s = get_all_activity_labels(h5_data)
        left_hand_tactile, right_hand_tactile, tactile_time_s = parse_tactile_readings(h5_data)
        left_hand_pose, right_hand_pose, handpose_time_s = parse_handpose_data(h5_data)
        body_pose, bodypose_time_s = parse_bodypose_data(h5_data)
        gaze_data, gaze_time_s = parse_gaze_data(h5_data)
        myo_emg_left, myo_emg_left_time = parse_myo_emg(h5_data, 'myo-left')
        myo_emg_right, myo_emg_right_time = parse_myo_emg(h5_data, 'myo-right')
        myo_acc_left, myo_acc_left_time = parse_myo_acc(h5_data, 'myo-left')
        myo_acc_right, myo_acc_right_time = parse_myo_acc(h5_data, 'myo-right')        

        if PARSE_VIDEO:
            # start_frames, end_frames, video_time_s = parse_timestamps_for_egocentric_video_data(h5_data)
            start_frames, end_frames, video_time_s = parse_timestamps_for_video_data(h5_data, txt_path=all_video_time_paths[i])
            img_frames_64  = parse_video_data(all_video_paths[i], start_frames, end_frames)
            # img_frames_256 = parse_video_data(all_video_paths[i], start_frames, end_frames)
            print(video_time_s[0].shape, img_frames_64[0].shape)


        # resample to sync data
        for i_act, activity_label in enumerate(activities_labels):
            # for i_act in range(3): 
            target_label_ori = activity_label
            label_start_time_s = activities_start_times_s[i_act]
            label_end_time_s = activities_end_times_s[i_act]

            # tactile pose
            tactile_indexes_forLabel = np.where((tactile_time_s >= label_start_time_s) & (tactile_time_s <= label_end_time_s))[0]
            left_tactile_data_forLabel = left_hand_tactile[tactile_indexes_forLabel, :]
            right_tactile_data_forLabel = right_hand_tactile[tactile_indexes_forLabel, :]
            tactile_time_s_forLabel = tactile_time_s[tactile_indexes_forLabel]
            
            # hand pose
            handpose_indexes_forLabel = np.where((handpose_time_s >= label_start_time_s) & (handpose_time_s <= label_end_time_s))[0]
            handpose_time_s_forLabel = handpose_time_s[handpose_indexes_forLabel]
            left_handpose_data_forLabel = left_hand_pose[handpose_indexes_forLabel, :]
            right_handpose_data_forLabel = right_hand_pose[handpose_indexes_forLabel, :]

            # body pose
            bodypose_indexes_forLabel = np.where((bodypose_time_s >= label_start_time_s) & (bodypose_time_s <= label_end_time_s))[0]
            bodypose_time_s_forLabel = bodypose_time_s[bodypose_indexes_forLabel]
            bodypose_data_forLabel = body_pose[bodypose_indexes_forLabel, :]

            # gaze pose
            gaze_indexes_forLabel = np.where((gaze_time_s >= label_start_time_s) & (gaze_time_s <= label_end_time_s))[0]
            gaze_time_s_forLabel = gaze_time_s[gaze_indexes_forLabel]
            gaze_data_forLabel = gaze_data[gaze_indexes_forLabel, :]

            # myo
            myo_acc_left_indexes_forLabel = np.where((myo_acc_left_time >= label_start_time_s) & (myo_acc_left_time <= label_end_time_s))[0]
            myo_acc_left_forLabel = myo_acc_left[myo_acc_left_indexes_forLabel]
            myo_acc_left_times_forLabel = myo_acc_left_time[myo_acc_left_indexes_forLabel]

            myo_emg_left_indexes_forLabel = np.where((myo_emg_left_time >= label_start_time_s) & (myo_emg_left_time <= label_end_time_s))[0]
            myo_emg_left_forLabel = myo_emg_left[myo_emg_left_indexes_forLabel]
            myo_emg_left_times_forLabel = myo_emg_left_time[myo_emg_left_indexes_forLabel]
            
            myo_emg_right_indexes_forLabel = np.where((myo_emg_right_time >= label_start_time_s) & (myo_emg_right_time <= label_end_time_s))[0]
            myo_emg_right_forLabel = myo_emg_right[myo_emg_right_indexes_forLabel]
            myo_emg_right_times_forLabel = myo_emg_right_time[myo_emg_right_indexes_forLabel]
            
            myo_acc_right_indexes_forLabel = np.where((myo_acc_right_time >= label_start_time_s) & (myo_acc_right_time <= label_end_time_s))[0]
            myo_acc_right_forLabel = myo_acc_right[myo_acc_right_indexes_forLabel]
            myo_acc_right_times_forLabel = myo_acc_right_time[myo_acc_right_indexes_forLabel]
            
            print('tactile Data for Label "%s"' % (target_label_ori))
            print('Label instance start time  :', label_start_time_s)
            print('Label instance end time    :', label_end_time_s)
            print('Label instance duration [s]:', (label_end_time_s-label_start_time_s))
            
            if target_label_ori == 'Open a jar of almond butter':
                target_label_ori = 'Open/close a jar of almond butter'
            elif target_label_ori == 'Get items from refrigerator/cabinets/drawers':
                target_label_ori = 'Get/replace items from refrigerator/cabinets/drawers'
            
            # use tactile to anchor other modality
            handpose_index_resampled_forLabel = rough_mapping(tactile_time_s_forLabel, handpose_time_s, np.arange(left_hand_pose.shape[0])).reshape(-1)
            left_hand_pose_resampled_forLabel = left_hand_pose[handpose_index_resampled_forLabel]
            right_hand_pose_resampled_forLabel = right_hand_pose[handpose_index_resampled_forLabel]
            
            bodypose_index_resampled_forLabel = rough_mapping(tactile_time_s_forLabel, bodypose_time_s, np.arange(body_pose.shape[0])).reshape(-1)
            body_pose_resampled_forLabel = body_pose[bodypose_index_resampled_forLabel]

            gaze_index_resampled_forLabel = rough_mapping(tactile_time_s_forLabel, gaze_time_s, np.arange(gaze_data.shape[0])).reshape(-1)
            gaze_data_resampled_forLabel = gaze_data[gaze_index_resampled_forLabel]

            left_myo_emg_index_resampled_forLabel = rough_mapping(tactile_time_s_forLabel, myo_emg_left_time, np.arange(myo_emg_left.shape[0])).reshape(-1)
            left_myo_emg_resampled_forLabel = myo_emg_left[left_myo_emg_index_resampled_forLabel]
            
            right_myo_emg_index_resampled_forLabel = rough_mapping(tactile_time_s_forLabel, myo_emg_right_time, np.arange(myo_emg_right.shape[0])).reshape(-1)
            right_myo_emg_resampled_forLabel = myo_emg_right[right_myo_emg_index_resampled_forLabel]

            left_myo_acc_index_resampled_forLabel = rough_mapping(tactile_time_s_forLabel, myo_acc_left_time, np.arange(myo_acc_left.shape[0])).reshape(-1)
            left_myo_acc_resampled_forLabel = myo_acc_left[left_myo_acc_index_resampled_forLabel]
            
            right_myo_acc_index_resampled_forLabel = rough_mapping(tactile_time_s_forLabel, myo_acc_right_time, np.arange(myo_acc_right.shape[0])).reshape(-1)
            right_myo_acc_resampled_forLabel = myo_acc_right[right_myo_acc_index_resampled_forLabel]
            

            sample_index = resample_sequence(16, left_tactile_data_forLabel.shape[0])

            if PARSE_VIDEO:
                video_index_resampled_forLabel = rough_mapping(tactile_time_s_forLabel, video_time_s[i_act], np.arange(img_frames_64[i_act].shape[0]))
                # video_resampled_forLabel_256 = img_frames_256[i_act][video_index_resampled_forLabel].reshape(-1, 256, 256, 3)
                video_resampled_forLabel_64 = img_frames_64[i_act][video_index_resampled_forLabel].reshape(-1, 64, 64, 3)
                # video_resampled_forLabel_32 = img_frames_32[i_act][video_index_resampled_forLabel].reshape(-1, 32, 32, 3)

                sample_index = resample_sequence(16, video_resampled_forLabel_64.shape[0])
                print(len(sample_index))
            
            
            for data_save_id, inds in enumerate(sample_index):
                if inds.shape[0] != 16:
                    continue
                
                cur_data_dict = {'signal': {'tactile-glove-left': left_tactile_data_forLabel[inds],
                                 'tactile-glove-right': right_tactile_data_forLabel[inds],
                                 'gaze': gaze_index_resampled_forLabel[inds],
                                 'myo-emg-left': left_myo_emg_resampled_forLabel[inds],
                                 'myo-emg-right': right_myo_emg_resampled_forLabel[inds],
                                 'myo-acc-left': left_myo_acc_resampled_forLabel[inds],
                                 'myo-acc-right': right_myo_acc_resampled_forLabel[inds],
                                 'joint-position': body_pose_resampled_forLabel[inds],
                                 'right-hand-pose': right_hand_pose_resampled_forLabel[inds],
                                 'left-hand-pose': left_hand_pose_resampled_forLabel[inds]},
                                 'label_text': target_label_ori ,
                                 'label_idx': LABEL_NAMES.index(target_label_ori),
                                }
                


                cur_data_dir = f'./{save_dir}/S0{subject[0]}/{LABEL_NAMES.index(target_label_ori)}/'
                os.makedirs(cur_data_dir, exist_ok=True)
                
                write_picke(os.path.join(cur_data_dir, f'{data_save_id}.p'),  cur_data_dict)
                
                if PARSE_VIDEO:
                    print(inds)
                    # np.savez(os.path.join(cur_data_dir, f'{video_dir}_{data_save_id}_32.npz'), video_resampled_forLabel_32[inds])
                    np.savez(os.path.join(cur_data_dir, f'{video_dir}_{data_save_id}_64.npz'), video_resampled_forLabel_64[inds])
                    # np.savez(os.path.join(cur_data_dir, f'{video_dir}_{data_save_id}_256.npz'), video_resampled_forLabel_256[inds])

            
            # video_resampled_forLabel_64 = img_frames_tiny[target_label][video_index_resampled_forLabel.astype(np.int64)].reshape(-1, 32, 32, 3)

            # # saving convention should save text label
            target_label = target_label_ori.replace(' ', '_').replace('/', '_').lower()

            # np.save(f'./{save_dir}/{target_label}/left_tactile_{subject_suffix}.npy', left_tactile_data_forLabel)
            # np.save(f'./{save_dir}/{target_label}/right_tactile_{subject_suffix}.npy', right_tactile_data_forLabel)
            # np.save(f'./{save_dir}/{target_label}/left_hand_pose_{subject_suffix}.npy', left_hand_pose_resampled_forLabel)
            # np.save(f'./{save_dir}/{target_label}/right_hand_pose_{subject_suffix}.npy', right_hand_pose_resampled_forLabel)
            # np.save(f'./{save_dir}/{target_label}/body_pose_{subject_suffix}.npy', body_pose_resampled_forLabel)
            # np.savez(f'./{save_dir}/{target_label}/{video_dir}_256_{subject_suffix}.npz', video_data=video_resampled_forLabel_256, sampled_index=video_index_resampled_forLabel)
            # np.savez(f'./{save_dir}/{target_label}/{video_dir}_64_{subject_suffix}.npz', video_data=video_resampled_forLabel_64, sampled_index=video_index_resampled_forLabel)




