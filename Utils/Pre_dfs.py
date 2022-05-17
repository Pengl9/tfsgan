import scipy.io as scio
import numpy as np
import os

import torch

# def dfs_preprocess(DfsFile,Nseg):

#     Dfs = scio.loadmat(DfsFile)
    
#     n_len = np.shape(Dfs['doppler_spectrum'])[0]
#     dfs_len = int(np.shape(Dfs['doppler_spectrum'])[2] / Nseg)

#     # Data reorganization
#     dfs_seg = np.ones([4 , 121 , Nseg])
#     for i in range(n_len):
#         for j in range(121):
#             for k in range(Nseg):
#                 latent_dfs = Dfs['doppler_spectrum'][i , j , k*dfs_len:(k+1)*dfs_len]
#                 dfs_seg[i , j , k] = np.mean(latent_dfs)

#     # Normalization Between Receivers(Compensate Path-Loss)            
#     for i in range(3):
#         dfs_seg[i+1 , : , :] = dfs_seg[i+1 , : , :] * np.sum(dfs_seg[0 , : , :]) / np.sum(dfs_seg[i+1 , : , :])

#     #print("\n dfs is loaded \n")

#     return dfs_seg

def getpath(dfspath):
    path = os.listdir(dfspath)
    latent_files = []
    for pathname in path:
        files_path = os.path.join(dfspath+pathname)
        latent_files.append(files_path)
    
    return latent_files

def data_get(files_path):
    latent_files = getpath(files_path)
    dfs = []
    for i in range(len(latent_files)):
        dfs.append(np.load(latent_files[i]))
    
    return dfs