import os
import sys
sys.path.append('src')
from src.simvp_model import *
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '55000'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tmp_dir = '~/tmp'
os.environ['TMPDIR'] = tmp_dir
import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')

import numpy as np
from src.pytorch_losses import *
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import csv
import datetime

#pre-computed global normalisation stats
mean_ssh = 0.074
std_ssh = 0.0986
mean_sst = 293.307
std_sst = 8.726

class SSH_Dataset_data_challenge(Dataset):
    def __init__(self, data_dir, region, N_t, mean_ssh, std_ssh, mean_sst, std_sst):
        self.data_dir = data_dir
        self.data = np.load(data_dir + f'input_data_region{region}.npy',mmap_mode = 'r')
        self.N_t = N_t
        self.mean_ssh = mean_ssh
        self.std_ssh = std_ssh
        self.mean_sst = mean_sst
        self.std_sst = std_sst

    def __len__(self):
        return 365

    def __getitem__(self, idx):
        in_data = self.data[idx:idx+self.N_t,:,:,:].copy()
        ssh = in_data[:,:,:,1]
        sst = in_data[:,:,:,0]
        ssh[ssh!=0] = (ssh[ssh!=0]-self.mean_ssh)/self.std_ssh
        sst[sst<273] = 0
        sst[sst!=0] = (sst[sst!=0]-self.mean_sst)/self.std_sst
        invar = torch.from_numpy(np.stack((ssh, sst), axis = 1).astype(np.float32))
        outvar = torch.from_numpy(np.zeros((400,3)).astype(np.float32))
        
        return invar, outvar

class SSH_Dataset_data_challenge_ssh_only(Dataset):
    def __init__(self, data_dir, region, N_t, mean_ssh, std_ssh, mean_sst, std_sst):
        self.data_dir = data_dir
        self.data = np.load(data_dir + f'input_data_region{region}.npy',mmap_mode = 'r')
        self.N_t = N_t
        self.mean_ssh = mean_ssh
        self.std_ssh = std_ssh
        self.mean_sst = mean_sst
        self.std_sst = std_sst

    def __len__(self):
        return 365

    def __getitem__(self, idx):
        in_data = self.data[idx:idx+self.N_t,:,:,:].copy()
        ssh = in_data[:,:,:,1]
        ssh[ssh!=0] = (ssh[ssh!=0]-self.mean_ssh)/self.std_ssh
        invar = torch.from_numpy(np.expand_dims(ssh, axis = 1).astype(np.float32))
        outvar = torch.from_numpy(np.zeros((400,3)).astype(np.float32))
        
       	return invar, outvar
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Specify the GPU device
pred_dir = './uninterpolated_preds/'
weight_dir = './model_weights/'
data_challenge_dir = './pre-processed/testing/'

available_regions = [i for i in range(5615)]

n_cpus = 10
n_t = 30
L_x = 960e3
L_y = 960e3
n = 128
batch_size = 50 # DON'T CHANGE, THIS IS FIXED IN THE PRE-PROCESSING TO BE 1 BATCH PER FILE
n_obs_max = 400 # max number of SSH observations on any day in loss function, allows to have fixed size inputs/outputs with zero padding making it easier to create TFRecord dataset
n_train_samples = 1000000
experiment_name = f'simvp_ssh_sst_ns1000000_global_'
weight_epoch = 48
n_regions = 5615
              
lr = 0.001
n_train_batches = int(n_train_samples/batch_size)
n_val_batches = 500

model = SimVP_Model_no_skip_sst(in_shape=(n_t,2,128,128),model_type='gsta',hid_S=8,hid_T=128,drop=0.2,drop_path=0.15).to(device)

saved_weights_path = weight_dir + experiment_name + f'_weights_epoch{weight_epoch}'
state_dict = torch.load(saved_weights_path)['model_state_dict']

if "module." in list(state_dict.keys())[0]:
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

for region in range(5615):
    dataset = SSH_Dataset_data_challenge(data_challenge_dir, region, n_t, mean_ssh, std_ssh, mean_sst, std_sst)
    data_loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=n_cpus)
    i = 0
    pred = np.zeros((365,128,128))
    
    print(region)
    
    with torch.no_grad():
        for torch_input_batch, _ in data_loader:

            torch_input_batch = torch_input_batch.to(device)

            preds = model(torch_input_batch)
            preds = preds.cpu().numpy()[:,15,0,:,:]
            preds = preds*std_ssh+mean_ssh
            pred[i:i+preds.shape[0],:,:] = preds
            i+=preds.shape[0]
            print(i)
    np.save(pred_dir + experiment_name + f'preds_region{region}.npy', pred)

# Clean up
if os.path.exists(tmp_dir):
    if len(os.listdir(tmp_dir)) > 0:
        os.system('rm -r '+tmp_dir+'/*')

###############################
# REFACTOR PREDICTIONS TO BE SAVED PER DAY RATHER THAN PER REGION
###############################

input_directory = './uninterpolated_preds/'
output_directory = './preds_refactored/'
num_files = np.size(available_regions)
output_shape = (num_files, 128, 128)
start_date = datetime.date(2019,1,1)
for day in range(365):
    print(f'Refactoring: day {day}')
    output_file = output_directory + experiment_name + f'_pred_{start_date + datetime.timedelta(days=day)}.npy'

    day_data = np.empty(output_shape, dtype=np.float64)

    for i, file_index in enumerate(available_regions):
        input_file = input_directory + experiment_name + f'preds_region{file_index}.npy'
        data = np.load(input_file, mmap_mode='r')
        chunk = data[day,:,:].copy()
        day_data[i,:,:] = chunk
    # save day data to the output file
    np.save(output_file, day_data)

os.system('rm '+input_directory+'*')
print("Refactoring completed.")

