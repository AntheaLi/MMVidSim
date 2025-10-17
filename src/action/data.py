import os, sys, glob
import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
sys.path.append('../../')
from data_processing.utils.data_utils import read_pickle, resample, parse_compact
from data_processing.settings import LABELS, LABEL_NAMES, LABEL_ALTERNATES

from PIL import Image
import skimage
from torchvision.transforms import v2


def resample_sequence(sequence_len, total_sequence_len, sample_rate):
    inds = []
    start_ind = 0
    while start_ind < (total_sequence_len - sequence_len * sample_rate):
        inds.append(np.arange(start_ind, total_sequence_len, sample_rate)[:sequence_len])
        start_ind += 1
    return inds


def collate_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class ActionSenseVideo(Dataset):
    """ ActionSense dataset class for sensor caption.
    """
    def __init__(self,
                 path="Dataset/",
                 split="val",
                 parse_signal_keys=['tactile-glove-left', 'tactile-glove-right', 'myo-emg-left', 'myo-emg-right', 'joint-position', 'right-hand-pose', 'left-hand-pose'],
                 resample_len=None, 
                 resample_type='avg',
                 compact=False,
                 parse_signal=True,
                 video_type='video',
                 video_res=256,
                 ) -> None:
        super().__init__()

        self.resample_len = resample_len
        self.parse_signal_keys = parse_signal_keys
        self.resample_type = resample_type
        self.compact = compact
        self.parse_signal = parse_signal
        self.video_type = video_type
        self.video_res = video_res
        self.center_crop = None
        if video_res not in [64, 256]: 
            self.video_res = 256
            self.center_crop = (self.video_res // 2 - video_res // 2, self.video_res // 2 + video_res // 2)
                        
        signal_paths = glob.glob(os.path.join(path, 'signals', split, '**/*.p'), recursive=True)
        signal_paths1 = glob.glob(os.path.join(path, 'signals1', split, '**/*.p'), recursive=True)
        signal_paths2 = glob.glob(os.path.join(path, 'signals2', split, '**/*.p'), recursive=True)
        self.signal_paths = signal_paths + signal_paths1 + signal_paths2


        video_path = [os.path.join( path, 'videos', split, p.split('/')[-3], p.split('/')[-2], 'video_'+p.split('/')[-1].split('.')[0]+f'_{self.video_res}.npz') for p in signal_paths]
        video_path1 = [os.path.join( path, 'videos1', split, p.split('/')[-3], p.split('/')[-2], 'video_'+p.split('/')[-1].split('.')[0]+f'_{self.video_res}.npz') for p in signal_paths1]
        video_path2 = [os.path.join( path, 'videos2', split, p.split('/')[-3], p.split('/')[-2], 'video_'+p.split('/')[-1].split('.')[0]+f'_{self.video_res}.npz') for p in signal_paths2]

        self.video_path = video_path + video_path1 + video_path2
        self.video_path = [x for x in self.video_path if os.path.exists(x)]

        assert len(self.video_path) == len(self.signal_paths), f'video {len(self.video_path)} signal {len(self.signal_paths)}'

    def __getitem__(self, index:int) -> dict:
        fpath = self.signal_paths[index]
        vpath = self.video_path[index]
        meta_data = read_pickle(fpath)
        video_data = np.load(vpath)['arr_0']
        data = meta_data["signal"]
        data_signal_name = list(data.keys())
        label_index = np.array([meta_data["label_idx"]])
        label_name = [meta_data["label_text"].lower()]
        data_dict = {}

        if self.center_crop is not None:
            video_data = video_data[:, self.center_crop[0]: self.center_crop[1], self.center_crop[0]: self.center_crop[1]]

        if self.resample_len is not None: data_dict['video'] = video_data[:self.resample_len] / 256
        else: data_dict['video'] = video_data / 256

        # data_dict['video_dist'] = np.array([data_dict['video'][i+1] - data_dict['video'][i] for i in range(data_dict['video'].shape[0]-1)])
        data_dict['video_dist'] = np.array([data_dict['video'][i+1] - data_dict['video'][0] for i in range(data_dict['video'].shape[0]-1)])
        # data_dict['video_dist'] = np.array([data_dict['video'][i+1] - data_dict['video'][i] for i in range(data_dict['video'].shape[0]-1)])
        data_dict['label'] = label_index
        data_dict['label_text'] = label_name
        data_dict['negative_label_text'] = [""]

        if self.parse_signal:
            signal_dict = {}
            for signal_key in self.parse_signal_keys:

                if self.compact: signal_data = parse_compact(signal_key, data)
                else: signal_data = data[signal_key]

                if self.resample_len is not None:
                    # resampled_indices = resample(data[signal_key].shape[0], self.resample_len, self.resample_type) 
                    signal_data = signal_data[:self.resample_len]

                if 'tactile' in signal_key:
                    signal_data = signal_data[:,np.newaxis, ...]

                signal_dict[signal_key] = signal_data
            data_dict['signal'] = signal_dict

        data_dict['path'] = self.video_path[index]

        return data_dict

    def __len__(self):
        return len(self.signal_paths)
    



class ActionSenseVideoLongDescription(Dataset):
    """ ActionSense dataset class for sensor caption.
    """
    def __init__(self,
                 path="Dataset/",
                 split="val",
                 parse_signal_keys=['tactile-glove-left', 'tactile-glove-right', 'myo-emg-left', 'myo-emg-right', 'joint-position', 'right-hand-pose', 'left-hand-pose'],
                 resample_len=None, 
                 resample_type='avg',
                 compact=False,
                 parse_signal=True,
                 video_type='video',
                 video_res=256,
                 ) -> None:
        super().__init__()

        self.resample_len = resample_len
        self.parse_signal_keys = parse_signal_keys
        self.resample_type = resample_type
        self.compact = compact
        self.parse_signal = parse_signal
        self.video_type = video_type
        self.video_res = video_res
        self.center_crop = None
        if video_res not in [64, 256]: 
            self.video_res = 256
            self.center_crop = (self.video_res // 2 - video_res // 2, self.video_res // 2 + video_res // 2)
                        
        signal_paths = glob.glob(os.path.join(path, 'signals', split, '**/*.p'), recursive=True)
        signal_paths1 = glob.glob(os.path.join(path, 'signals1', split, '**/*.p'), recursive=True)
        signal_paths2 = glob.glob(os.path.join(path, 'signals2', split, '**/*.p'), recursive=True)
        self.signal_paths = signal_paths + signal_paths1 + signal_paths2


        video_path = [os.path.join( path, 'videos', split, p.split('/')[-3], p.split('/')[-2], 'video_'+p.split('/')[-1].split('.')[0]+f'_{self.video_res}.npz') for p in signal_paths]
        video_path1 = [os.path.join( path, 'videos1', split, p.split('/')[-3], p.split('/')[-2], 'video_'+p.split('/')[-1].split('.')[0]+f'_{self.video_res}.npz') for p in signal_paths1]
        video_path2 = [os.path.join( path, 'videos2', split, p.split('/')[-3], p.split('/')[-2], 'video_'+p.split('/')[-1].split('.')[0]+f'_{self.video_res}.npz') for p in signal_paths2]

        self.video_path = video_path + video_path1 + video_path2
        self.video_path = [x for x in self.video_path if os.path.exists(x)]

        assert len(self.video_path) == len(self.signal_paths), f'video {len(self.video_path)} signal {len(self.signal_paths)}'

    def __getitem__(self, index:int) -> dict:
        fpath = self.signal_paths[index]
        vpath = self.video_path[index]
        meta_data = read_pickle(fpath)
        video_data = np.load(vpath)['arr_0']
        data = meta_data["signal"]
        data_signal_name = list(data.keys())
        label_index = np.array([meta_data["label_idx"]])
        label_name = [meta_data["label_text"].lower()]

        if 'video1' in self.signal_paths[index]:
            add_description = 'in a very slow manner'
        elif 'video2' in self.signal_paths[index]:
            add_description = 'in pace'
        else:
            add_description = 'in a very fast manner'


        data_dict = {}

        if self.center_crop is not None:
            video_data = video_data[:, self.center_crop[0]: self.center_crop[1], self.center_crop[0]: self.center_crop[1]]

        if self.resample_len is not None: data_dict['video'] = video_data[:self.resample_len] / 256
        else: data_dict['video'] = video_data / 256

        
        if self.parse_signal:
            signal_dict = {}
            for signal_key in self.parse_signal_keys:

                if self.compact: signal_data = parse_compact(signal_key, data)
                else: signal_data = data[signal_key]

                if self.resample_len is not None:
                    # resampled_indices = resample(data[signal_key].shape[0], self.resample_len, self.resample_type) 
                    signal_data = signal_data[:self.resample_len]

                if 'tactile' in signal_key:
                    signal_data = signal_data[:,np.newaxis, ...]
                    if signal_data.mean() > 0.55:
                        add_description += ' with strong force holding the tools'
                    elif signal_data.mean() < 0.45:
                        add_description += ' while gently holding the tools'
                    else:
                        add_description += ' while holding the tools'
                        

                signal_dict[signal_key] = signal_data
            data_dict['signal'] = signal_dict

        # data_dict['video_dist'] = np.array([data_dict['video'][i+1] - data_dict['video'][i] for i in range(data_dict['video'].shape[0]-1)])
        data_dict['video_dist'] = np.array([data_dict['video'][i+1] - data_dict['video'][0] for i in range(data_dict['video'].shape[0]-1)])
        # data_dict['video_dist'] = np.array([data_dict['video'][i+1] - data_dict['video'][i] for i in range(data_dict['video'].shape[0]-1)])
        data_dict['label'] = label_index
        data_dict['label_text'] = [ meta_data["label_text"] + add_description]
        data_dict['negative_label_text'] = [""]

        data_dict['path'] = self.video_path[index]

        return data_dict

    def __len__(self):
        return len(self.signal_paths)
    
    
    
class ActionSenseVideoVerb(Dataset):
    """ ActionSense dataset class for sensor caption.
    """
    def __init__(self,
                 path="Dataset/",
                 split="val",
                 parse_signal_keys=['tactile-glove-left', 'tactile-glove-right', 'myo-emg-left', 'myo-emg-right', 'joint-position', 'right-hand-pose', 'left-hand-pose'],
                 resample_len=None, 
                 resample_type='avg',
                 compact=False,
                 parse_signal=True,
                 video_type='video',
                 video_res=256,
                 ) -> None:
        super().__init__()

        self.resample_len = resample_len
        self.parse_signal_keys = parse_signal_keys
        self.resample_type = resample_type
        self.compact = compact
        self.parse_signal = parse_signal
        self.video_type = video_type
        self.video_res = video_res
        self.center_crop = None
        
        self.verb_list = ['clean', 'clear', 'get items', 'replace items', 'load', 'open', 'peel', 'close', 'pour', 'stack', 'spread', 'slice', 'set', 'unload']
        
        if video_res not in [64, 256]: 
            self.video_res = 256
            self.center_crop = (self.video_res // 2 - video_res // 2, self.video_res // 2 + video_res // 2)
                        
        signal_paths = glob.glob(os.path.join(path, 'signals', split, '**/*.p'), recursive=True)
        signal_paths1 = glob.glob(os.path.join(path, 'signals1', split, '**/*.p'), recursive=True)
        signal_paths2 = glob.glob(os.path.join(path, 'signals2', split, '**/*.p'), recursive=True)
        self.signal_paths = signal_paths + signal_paths1 + signal_paths2
        


        video_path = [os.path.join( path, 'videos', split, p.split('/')[-3], p.split('/')[-2], 'video_'+p.split('/')[-1].split('.')[0]+f'_{self.video_res}.npz') for p in signal_paths]
        video_path1 = [os.path.join( path, 'videos1', split, p.split('/')[-3], p.split('/')[-2], 'video_'+p.split('/')[-1].split('.')[0]+f'_{self.video_res}.npz') for p in signal_paths1]
        video_path2 = [os.path.join( path, 'videos2', split, p.split('/')[-3], p.split('/')[-2], 'video_'+p.split('/')[-1].split('.')[0]+f'_{self.video_res}.npz') for p in signal_paths2]

        self.video_path = video_path + video_path1 + video_path2
        self.video_path = [x for x in self.video_path if os.path.exists(x)]

        assert len(self.video_path) == len(self.signal_paths), f'video {len(self.video_path)} signal {len(self.signal_paths)}'

    def __getitem__(self, index:int) -> dict:
        fpath = self.signal_paths[index]
        vpath = self.video_path[index]
        meta_data = read_pickle(fpath)
        video_data = np.load(vpath)['arr_0']
        data = meta_data["signal"]
        data_signal_name = list(data.keys())
        label_index = np.array([meta_data["label_idx"]])
        label_name = [meta_data["label_text"]]
        
        verb = [x for x in self.verb_list if x in meta_data["label_text"].lower()]
        if len(verb) == 0:
            verb = [meta_data['label_text'].split('_')[0]]
        verb = ' '.join(verb)
        
        data_dict = {}

        if self.center_crop is not None:
            video_data = video_data[:, self.center_crop[0]: self.center_crop[1], self.center_crop[0]: self.center_crop[1]]

        if self.resample_len is not None: data_dict['video'] = video_data[:self.resample_len] / 256
        else: data_dict['video'] = video_data / 256

        
        if self.parse_signal:
            signal_dict = {}
            for signal_key in self.parse_signal_keys:

                if self.compact: signal_data = parse_compact(signal_key, data)
                else: signal_data = data[signal_key]

                if self.resample_len is not None:
                    # resampled_indices = resample(data[signal_key].shape[0], self.resample_len, self.resample_type) 
                    signal_data = signal_data[:self.resample_len]

                if 'tactile' in signal_key:
                    signal_data = signal_data[:,np.newaxis, ...]
                   

                signal_dict[signal_key] = signal_data
            data_dict['signal'] = signal_dict

        # data_dict['video_dist'] = np.array([data_dict['video'][i+1] - data_dict['video'][i] for i in range(data_dict['video'].shape[0]-1)])
        data_dict['video_dist'] = np.array([data_dict['video'][i+1] - data_dict['video'][0] for i in range(data_dict['video'].shape[0]-1)])
        # data_dict['video_dist'] = np.array([data_dict['video'][i+1] - data_dict['video'][i] for i in range(data_dict['video'].shape[0]-1)])
        data_dict['label'] = label_index
        data_dict['label_text'] = [verb]
        data_dict['negative_label_text'] = [""]

        data_dict['path'] = self.video_path[index]

        return data_dict

    def __len__(self):
        return len(self.signal_paths)
    



class H2ODataset(Dataset):
    """ ActionSense dataset class for sensor caption.
    """
    def __init__(self,
                 path="H2O/",
                 split="val",
                 resample_len=None, 
                 video_res=256,
                 ) -> None:
        super().__init__()

        self.resample_len = resample_len if resample_len is not None else 9
        self.video_res = video_res
        self.sample_rate = [1,2,3]
        
        self.all_actions = {}
        with open(f'{path}/action_labels.txt', 'r') as f:
            for l in f:
                self.all_actions[l.split()[0]] = ' '.join(l.split()[1:])
        
        self.data_path_list = []
        self.data_action_label = []
        self.data_action_index = []
        self.data_start_frames = []
        self.sample_rate_list = []
        with open(f'{path}/label_split/action_{split}.txt', 'r') as f:
            for l in f:
                l = l.strip()
                data_path = l.strip().split()[1]
                data_path = os.path.join(path, data_path.split('/')[0]+'_ego' , '/'.join(data_path.split('/')[1:]))
                if not os.path.exists(data_path):
                    continue
                action_label = l.strip().split()[2]
                start_frame = int(l.strip().split()[3])
                end_frame = int(l.strip().split()[4])
                for s in self.sample_rate:
                    num_samples = end_frame - start_frame - self.resample_len * s
                    self.data_start_frames.extend(list(range(start_frame, end_frame - self.resample_len * s, 1)))
                    self.data_path_list.extend([data_path]*num_samples)
                    self.data_action_label.extend([self.all_actions[action_label]]*num_samples)
                    self.data_action_index.extend([action_label]*num_samples)
                    self.sample_rate_list.extend([s]*num_samples)

        assert len(self.data_start_frames) == len(self.data_path_list) == len(self.data_action_label), f'{len(self.data_start_frames)} {len(self.data_path_list)} {len(self.data_action_label)}'
        
        
    def _parse_handpose_h2o(self, data):
        origin = data[0]
        thumb = data[:5]
        
        index = data[4:9]
        index[0] = origin
        
        middle = data[8:13]
        middle[0] = origin
        
        ring = data[12:17]
        ring[0] = origin
        
        pinky = data[17:]
        
        hand_pose = np.concatenate([thumb, index, middle, ring, pinky], axis=0)
        
        return hand_pose
    

    def _process_item(self, index):
        data_path = self.data_path_list[index]
        start_frame = self.data_start_frames[index]
        sample_rate = self.sample_rate_list[index]
        
        transforms = v2.Compose([
            v2.CenterCrop(size=(256, 256)),
            v2.Resize(size=(self.video_res, self.video_res)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        
        video = [] 
        left_hand = []
        right_hand = []
        for i in range(self.resample_len): 
            frame = str(start_frame + sample_rate * i).zfill(6)
            img = Image.open(data_path+f'/cam4/rgb256/{frame}.jpg')
            img = transforms(img)
            video.append(img)
            with open(data_path+f'/cam4/hand_pose/{frame}.txt', 'r') as f:
                l = f.readline()
                l = [float(x) for x in l.split()]

                if l[0] > 0 and l[64] > 0:
                    left_hand.append(self._parse_handpose_h2o(np.array(l[1:64]).reshape(21, 3)).reshape(24, 3))
                    right_hand.append(self._parse_handpose_h2o(np.array(l[65:]).reshape(21, 3)).reshape(24, 3))
                else: 
                    print('pose no annotation corrupted data idx', index, self.data_path_list[index])
                    return None 
        
        right_hand = np.array(right_hand)
        left_hand = np.array(left_hand)
        video_data = torch.stack(video).permute(0, 2, 3, 1)
        if video_data.shape != torch.Size([self.resample_len, self.video_res, self.video_res, 3]):
            print('video shape wrong corrupted data idx', index, video_data.shape, self.data_path_list[index])
            return None
        
        video_data = video_data.numpy() 

        return video_data, left_hand, right_hand

    def __getitem__(self, index:int) -> dict:
        ''' every index give you '''
        
        output = self._process_item(index)
        # while None in output or int(output[0].shape[0]) != self.resample_len:
        #     index = np.random.randint(len(self.data_path_list))
        #     output = self._process_item(index)

        video_data, left_hand, right_hand = output

        data_dict = {}
        data_dict['signal'] = {}
        data_dict['signal']['left-hand-pose'] = left_hand
        data_dict['signal']['right-hand-pose'] = right_hand
        data_dict['video'] = video_data 
        data_dict['video_dist'] = np.array([data_dict['video'][i+1] - data_dict['video'][0] for i in range(data_dict['video'].shape[0]-1)])
        data_dict['label'] = self.data_action_index[index]
        data_dict['label_text'] = [self.all_actions[self.data_action_index[index]]]
        data_dict['negative_label_text'] = ""
        
        return data_dict


    def __len__(self):
        return len(self.data_path_list)



class HoloDataset(Dataset):
    """ Holoassist dataset class for sensor caption.
    """
    def __init__(self,
                 path="holo/",
                 split="val",
                 resample_len=None, 
                 video_res=256,
                 ) -> None:
        super().__init__()

        self.resample_len = resample_len if resample_len is not None else 9
        self.video_res = video_res
        # self.sample_rate = [1,2,3]
        self.sample_rate = [2,4]
        self.video_struct = path + '/video/{}/Export_py/Video/images_aligned/' # {}.png
        self.action_struct = path + '/video/{}/Export_py/Video/action_aligned/' # {}.npz
        
        self.all_actions = {}
        
        self.data_path_list = []
        self.data_action_label = []
        self.data_action_index = []
        self.data_start_frames = []
        self.sample_rate_list = []
        # action_label = ['printer_small', 'espresso', 'nespresso', 'marius_assemble']
        # action_label = ['printer_small']
        action_label = ['marius_assemble']
        # action_label = ['espresso']
        
    
        with open(f'{path}/label_split/{split}-v1.txt', 'r') as f:
            for l in f:
                l = l.strip()
                data_path = l.strip()
                video_path = self.video_struct.format(data_path)
                action_path = self.video_struct.format(data_path)
                

                # if ('printer_small' not in data_path) and ('espresso' not in data_path) and ('nespresso' not in data_path) and ('marius_assemble' not in data_path):
                # if ('-espresso' not in data_path):
                if (action_label[0] not in data_path):
                    continue

                if not os.path.exists(video_path) or not os.path.exists(action_path):
                    continue
                
                if len(os.listdir(video_path)) <= 0 or len(os.listdir(action_path)) <= 0:
                    continue

                video_frames = sorted([int(x.split('.')[0]) for x in os.listdir(video_path)])
                action_frames = sorted([int(x.split('.')[0]) for x in os.listdir(action_path)])

                start_frame = max(min(video_frames), min(action_frames))
                end_frame = min(max(action_frames), max(video_frames))

                for s in self.sample_rate:
                    num_samples = end_frame - start_frame - self.resample_len * s
                    self.data_start_frames.extend(list(range(start_frame, end_frame - self.resample_len * s, 1)))
                    self.data_path_list.extend([data_path]*num_samples)
                    self.data_action_label.extend([data_path.split('-')[-1]]*num_samples)
                    self.data_action_index.extend([action_label.index(data_path.split('-')[-1])]*num_samples)
                    self.sample_rate_list.extend([s]*num_samples)

        assert len(self.data_start_frames) == len(self.data_path_list) == len(self.data_action_label), f'{len(self.data_start_frames)} {len(self.data_path_list)} {len(self.data_action_label)}'
        

    def _process_item(self, index):
        data_path = self.data_path_list[index]
        start_frame = self.data_start_frames[index]
        sample_rate = self.sample_rate_list[index]
        
        transforms = v2.Compose([
            v2.CenterCrop(size=(256, 256)),
            v2.Resize(size=(self.video_res, self.video_res)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        
        video = [] 
        left_hand = []
        right_hand = []
        acc = []
        for i in range(self.resample_len): 
            frame = str(start_frame + sample_rate * i).zfill(6)
            img = Image.open(self.video_struct.format(data_path)+f'{frame}.png')
            img = transforms(img)
            video.append(img)
            
            action = np.load(self.action_struct.format(data_path)+f'{frame}.npz')
            if 'left_hand' not in action or 'right_hand' not in action or 'acc' not in action:
                return None
            left_hand.append(action['left_hand'][:, 1:-1])
            right_hand.append(action['right_hand'][:, 1:-1])
            acc.append(action['acc'])
        
        video_data = torch.stack(video).permute(0, 2, 3, 1)
        # if video_data.shape != torch.Size([self.resample_len, self.video_res, self.video_res, 3]):
        #     print('video shape wrong corrupted data idx', index, video_data.shape, self.data_path_list[index])
        #     return None
        
        video_data = video_data.numpy() 
        left_hand = np.array(left_hand)
        right_hand = np.array(right_hand)
        acc = np.array(acc)

        return video_data, left_hand, right_hand, acc

    def __getitem__(self, index:int) -> dict:
        ''' every index give you '''
        
        output = self._process_item(index)
        # while None in output or int(output[0].shape[0]) != self.resample_len:
        #     index = np.random.randint(len(self.data_path_list))
        #     output = self._process_item(index)

        video_data, left_hand, right_hand, acc = output

        data_dict = {}
        data_dict['signal'] = {}
        data_dict['signal']['left-hand-pose'] = left_hand
        data_dict['signal']['right-hand-pose'] = right_hand
        # data_dict['signal']['acc'] = acc
        data_dict['video'] = video_data 
        data_dict['video_dist'] = np.array([data_dict['video'][i+1] - data_dict['video'][0] for i in range(data_dict['video'].shape[0]-1)])
        data_dict['label'] = self.data_action_index[index]
        data_dict['label_text'] = [self.data_action_label[index]]
        data_dict['negative_label_text'] = ""
        
        return data_dict
    
    def __len__(self):
        return len(self.data_start_frames)
        
if __name__ == '__main__':
    dataset = ActionSenseVideoVerb(path='Dataset/', video_res=64, split='train')
    dl = DataLoader(dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 32)
    for i, d in enumerate(dl):
        print(d['label_text'][0])
    
    # dataset = HoloDataset(path='holo/', video_res=64, split='train')
    # dl = DataLoader(dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 32)
    # for i, d in enumerate(dl):
    #     if d['video'].shape != torch.Size([1, 9, 64, 64, 3]):
    #         print(i, d['video'].shape)

