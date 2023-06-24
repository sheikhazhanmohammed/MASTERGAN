from torch.utils.data import Dataset
from utils import normalize
import glob
import numpy as np
import torch.utils.data as data

class DEMDataset(data.Dataset):
    def __init__(self, transform = None, load_dir = '/kaggle/input/demdataset8020/train/'):
        self.load_dir_hr = glob.glob(load_dir +'hr/*.npy')
        self.load_dir_lr = glob.glob(load_dir +'lr/*.npy')
        self.tranform = transform
        
    def __getitem__(self, index):
        hr = normalize(np.load(self.load_dir_hr[index])).astype(np.float32)
        lr = normalize(np.load(self.load_dir_lr[index])).astype(np.float32)
        
        
        if self.tranform:
            hr, lr = self.tranform(hr), self.tranform(lr)
        
        return hr, lr
    
    def __len__(self):
        return len(self.load_dir_hr)