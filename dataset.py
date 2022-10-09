# import mne
import re
import torch
import numpy as np
from torch.utils.data import Dataset
# from torch.utils.data import Dataloader
import os
from pathlib import Path

# class MIdataset(Dataset):
#     def __init__(self):
#         self.work_path = 'f:/Project/score-based4inverse'
#         self.root = 'train_data'
#         self.seg_lens = 512

#     def __len__(self):
#         lens = 0
#         for file in os.listdir(self.root):
#             for data_name in os.listdir(os.path.join(self.root, file)):
#                 data = np.load(os.path.join(self.root, file, data_name))   
#                 data_lens = int(data.shape[0] / self.seg_lens)
#                 lens = lens + data_lens
#                 os.chdir(self.work_path)
#             os.chdir(self.work_path)
#         return lens

#     def __getitem__(self, index):
#         for file in os.listdir(self.root):
#             for data_name in os.listdir(os.path.join(self.root, file)):
#                 data = np.load(os.path.join(self.root, file, data_name))   
#                 data_lens = int(data.shape[1] / self.seg_lens)
#                 if index - lens > 0 & index - lens <= data_lens:
#                     index_in_seg = index - lens
#                     data_seg = data[:, index_in_seg*self.seg_lens:(index_in_seg+1)*self.seg_lens]
#                     return data_seg.unsqueeze(0)    # (B, 1, K, L)  feature_channels = 1
#                 lens = lens + data_lens
#                 os.chdir(self.work_path)
#             os.chdir(self.work_path)
#         return

def data_slice(data, i):
    if i == 0:
        return data[:, :2000]
    elif i == 1:
        return data[:, 2000:]
    elif i == 2:
        return data[:, 1000:3000]
    return

class MIhandData(Dataset):
    def __init__(self):
        self.path = '/home/wyl/project/train_data'
        self.file = os.listdir(self.path)[0]
        self.data_list = sorted(os.listdir(os.path.join(self.path, self.file)))
        self.ses_lens = 20
        self.slice = 3

    def __len__(self):
        lens = len(self.data_list) * self.ses_lens * self.slice
        return lens

    def __getitem__(self, index):
        slice_ind = index % self.slice
        index = int(index / self.slice)
        ses_ind = int(index / self.ses_lens)
        data_ind = int(index % self.ses_lens)
        ses_data = np.load(os.path.join(self.path, self.file, self.data_list[ses_ind]))
        data = ses_data[data_ind]
        get_data = data_slice(data, slice_ind)
        return get_data.astype(np.float32)
