# data utils

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SceneDataset(Dataset):
    def __init__(self, data_path, obs_len=9, pred_len=12, div=1):
        super().__init__()
        # ['scene', 'id', 'frame', 'label', 'x', 'y', 'w', 'h', 'curv']
        self.data = np.load(data_path)
        self.scenes, cnts = np.unique(self.data[:, 0, 0], return_counts=True)
        self.splits = np.concatenate([[0], np.cumsum(cnts)])
        self.obs_len = obs_len
        self.length = obs_len + pred_len
        # normalize
        self.data[:, :, 4:6] -= np.mean(self.data[:, :, 4:6].reshape(-1, 2), axis=0)
        self.data[:, :, 4:6] /= div

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        # self.data -> [num_obj, length, 2]
        x = self.data[self.splits[idx]:self.splits[idx+1]].transpose(1, 0, 2)
        x = torch.from_numpy(x)
        return {
            'obs': x[:self.obs_len, :, 4:6].float(),
            'pred': x[self.obs_len:self.length, :, 4:6].float(),
            'start': x[-2, :, 4:6].float(),
            'goal': x[-1, :, 4:6].float(),
            'label': x[0, :, 3].long(),
            'curv': x[0, :, 8].float(),
        }


class SceneLoader(DataLoader):
    def __init__(self, dataset, shuffle=True, num_workers=0, pin_memory=True, prefetch_factor=100):
        super().__init__(
            dataset, batch_size=None, shuffle=shuffle, drop_last=False, 
            num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor
        )

