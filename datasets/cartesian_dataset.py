import os
import numpy as np
import scipy.io as sio
import json
import torch
import random
import torch.utils.data as data

def to_tensor(data):
    return torch.from_numpy(data)

class MRIDataset_Cartesian(data.Dataset):
    def __init__(self, opts, mode):
        self.mode = mode
        if self.mode == 'TRAIN':
            self.data_dir_flair = os.path.join(opts.data_root, 'train')
            self.sample_list = open(os.path.join(opts.list_dir, self.mode + '.txt')).readlines()
            self.seed = None

        if self.mode == 'VALI':
            self.data_dir_flair = os.path.join(opts.data_root, 'vali')
            self.sample_list = open(os.path.join(opts.list_dir, self.mode + '.txt')).readlines()
            self.seed = 1234

        if self.mode == 'TEST':
            self.data_dir_flair = os.path.join(opts.data_root, 'test')
            self.sample_list = open(os.path.join(opts.list_dir, self.mode + '.txt')).readlines()
            self.seed = 5678

        self.data_dir_flair = os.path.join(self.data_dir_flair)    # ref kspace directory (T1)

    def __getitem__(self, idx):

        ######### load data ##########
        slice_name = self.sample_list[idx].strip('\n')
        data_img = sio.loadmat(os.path.join(self.data_dir_flair, slice_name))

        ######### get HR  ##########
        HR = data_img['HR']
        HR = HR/HR.max()
        # HR_1 = HR[np.newaxis, :, : ,:]  # 1,w,h,t
        # HR_2 = HR[np.newaxis, :, :, :]  # 2,w,h,t
        # HR_3 = HR[np.newaxis, :, :, :]  # 3,w,h,t
        # HR = np.concatenate([HR_1, HR_2, HR_3], axis=0)  # 3,w,h,t
        HR = HR[np.newaxis, :, :, :]
        HR = to_tensor(HR).float()

        ######### get LR  ##########
        LR = data_img['LR_4']
        LR = LR / LR.max()
        # LR_1 = LR[np.newaxis, :, :, :]  # 1,w,h,t
        # LR_2 = LR[np.newaxis, :, :, :]  # 2,w,h,t
        # LR_3 = LR[np.newaxis, :, :, :]  # 3,w,h,t
        # LR = np.concatenate([LR_1, LR_2, LR_3], axis=0)  # 3,w,h,t
        LR = LR[np.newaxis, :, :, :]
        LR = to_tensor(LR).float()


        HR_img = HR.permute(3, 0, 1, 2)
        LR_img = LR.permute(3, 0, 1, 2)


# ---------------------over------
        return {
                'hr': HR_img,
                'lr': LR_img,
                }

    def __len__(self):
        return len(self.sample_list)

if __name__ == '__main__':
    a = 1
