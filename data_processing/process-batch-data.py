'''
    parse data into action: hand pose / body pose / force map
    parse data into start frame / end frame / sequence frame
'''
import numpy as np
import h5py
import os
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence


def resample_sequence(sequence_len, total_sequence_len, sample_num):
    inds = []
    start_ind = 0
    while start_ind < (total_sequence_len - sequence_len):
        inds.append(np.arange(start_ind, total_sequence_len, 3)[:sequence_len])
        start_ind += 1
    
    return inds
    

if __name__ == '__main__':
    # action
    os.makedirs()

    # 