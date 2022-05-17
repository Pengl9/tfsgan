
import numpy as np
import pandas as pd
#import torch
import os
import scipy.io as scio
import numpy as np
from sklearn import datasets
import torch
from torch.utils.data import Dataset,DataLoader


class PenglDataset(Dataset):
    def __init__(self,args):

        if args.type == 'wifi':
            self.path = getpath(os.path.join(f'./Data/{args.type}/mo_6/'))
            self.len = len(self.path)

        elif args.type == 'rfid':
            label = pd.DataFrame(np.load('./Data/rfid/rfid_labels.npy'))
            self.labels = label[label[0] == 6].index.tolist()
            self.path = np.load('./Data/rfid/rfid_data.npy')
            self.len = len(self.labels)
        
        elif args.type == 'uwb':
            label = pd.DataFrame(np.load('./Data/uwb/uwb_labels.npy'))
            self.labels = label[label[0] == 5].index.tolist()
            self.path = np.load('./Data/uwb/uwb_data.npy')
            self.len = len(self.labels)
        
        elif args.type == 'mmwave':
            self.path = np.load('./Data/mmwave/mmwave_stand_data.npy')
            self.len = len(self.path)

        self.type = args.type
        self.n_seg = args.n_seg

    def __len__(self):

        return self.len

    def __getitem__(self,index):

        if self.type == 'wifi':
            data = scio.loadmat(self.path[index])
            item = torch.from_numpy(data['csi_latent_seg']).permute(1,0)

        elif self.type == 'rfid':
            data = self.path[self.labels[index]]
            item = torch.from_numpy(data)

        elif self.type == 'uwb':
            data = self.path[self.labels[index]]
            item = torch.from_numpy(data)

        elif self.type == 'mmwave':
            data = self.path[index]
            item = torch.from_numpy(data)

        else: raise ValueError(' Wrong data type selection ')

        label = '0'

        return item,label


def getpath(dfspath):
    path = os.listdir(dfspath)
    latent_files = []
    for pathname in path:
        files_path = os.path.join(dfspath+pathname)
        latent_files.append(files_path)
    
    return latent_files


# eval()
if __name__ == '__main__':
    from io_utils import parse_args

    opt = parse_args('wifi')
    dataset = PenglDataset(opt)
    dataload = DataLoader(
        dataset = dataset,
        batch_size = 32,
        shuffle = True
    )
    for item, label in dataload:
        print(item.shape)
    print('----------------')

    opt = parse_args('rfid')
    dataset = PenglDataset(opt)
    dataload = DataLoader(
        dataset = dataset,
        batch_size = 32,
        shuffle = True
    )
    for item, label in dataload:
        print(item.shape)
    print('----------------')

    opt = parse_args('uwb')
    dataset = PenglDataset(opt)
    dataload = DataLoader(
        dataset = dataset,
        batch_size = 32,
        shuffle = True
    )
    for item, label in dataload:
        print(item.shape)
    print('----------------')
    
    opt = parse_args('mmwave')
    dataset = PenglDataset(opt)
    dataload = DataLoader(
        dataset = dataset,
        batch_size = 32,
        shuffle = True
    )
    for item, label in dataload:
        print(item.shape)
    print('----------------')
