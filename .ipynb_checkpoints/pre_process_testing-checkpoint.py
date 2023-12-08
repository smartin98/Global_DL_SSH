# 2023-05-02 Scott Martin
# Code to pre-process the subsetted data into ML-ready input-output pairs, save the pairs in TFRecord chunks of size ~100MB for optimal data pipeline performance.
# stationary gridded variables (bathymetry and MDT) will be appended as additional day at end of time series to allow easy passing to the keras model.

import numpy as np
import datetime
import os
from scipy import stats
import random
import tensorflow as tf
import time
import multiprocessing

# function to list all files within a directory including within any subdirectories
def GetListOfFiles(dirName, ext = '.nc'):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + GetListOfFiles(fullPath)
        else:
            if fullPath.endswith(ext):
                allFiles.append(fullPath)               
    return allFiles

def serialize_example(input_array, output_array):
        feature = {
            'input': tf.train.Feature(float_list=tf.train.FloatList(value=input_array.flatten())),
            'output': tf.train.Feature(float_list=tf.train.FloatList(value=output_array.flatten()))   
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

def parse_example(serialized_example):
    feature_description = {
        'input': tf.io.FixedLenFeature(int(batch_size*(N_t+1)*n*n*2), tf.float32),
        'output': tf.io.FixedLenFeature(int(batch_size*N_t*n_obs_max*3), tf.float32)
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)

    input_data = tf.reshape(example['input'], [batch_size,N_t+1,n,n,2])
    output_data = tf.reshape(example['output'], [batch_size,N_t,n_obs_max,3])

    return input_data, output_data

def find_indices(lst, target):
    return [index for index, value in enumerate(lst) if value == target]

def move_element_to_last(lst, idx):
    if idx < 0 or idx >= len(lst):
        # Index out of range
        return lst

    element = lst[idx]
    before = lst[:idx]
    after = lst[idx+1:]
    new_lst = before + after + [element]

    return new_lst


def bin_ssh(data_tracks, sats, L_x, L_y, n, filtered = False):
    ssh_grid = np.zeros((n,n,1))
    
    #6sat constellation:
    keep_sats = list(set(sats).intersection(['j3','j3n','s3a','c2','c2n','h2b','s3b','h2a','h2ag']))
    keep_sats_indices = [index for index, element in enumerate(sats) if element in ['j3','j3n','s3a','c2','c2n','h2b','s3b','h2a','h2ag']]
    if len(keep_sats)>0:
        data = np.concatenate([data_tracks[i] for i in keep_sats_indices],axis=0)
        input_grid, _,_,_ = stats.binned_statistic_2d(data[:,0], data[:,1], data[:,-2], statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
        input_grid = np.rot90(input_grid)
        input_grid[np.isnan(input_grid)] = 0

        ssh_grid[:,:,0] = input_grid
        
    return ssh_grid

sats_all = ['s3b','s3a','j3','h2b','h2ag','h2a','c2','c2n','j3n']
# n_unique_sats = 6
# constellations used:
# 1sat: Sentinel 3A
# 2sat: Sentinel 3A + Jason 3
# 3sat: Jason 3 + Sentinel 3A + Cryosat 2
# 4sat: Jason 3 + Sentinel 3A + Cryosat 2 + Haiyang 2B
# 5sat: Jason 3 + Sentinel 3A + Cryosat 2 + Haiyang 2B + Sentinel 3B
# 6sat: Jason 3 + Sentinel 3A + Cryosat 2 + Haiyang 2B + Sentinel 3B + Haiyang 2A


test_sats = ['alg','al'] # independent test satellite used for testing purposes, withhold from training data for all years
data_challenge_sat = 'alg'

sats = [s for s in sats_all if s not in test_sats]

batch_size = 25
n_obs_max = 400 # max number of SSH observations on any day in loss function, allows to have fixed size inputs/outputs with zero padding making it easier to create TFRecord dataset
N_t = 30 # length of single input time series in days
n = 128 # no. grid points per side of domain
L_x = 960e3 # size of domain
L_y = 960e3  # size of domain
filtered = False # whether to use the 65km band-pass filtered or unfiltered SSH observations
sst_high_res = True # True = L4 MUR SST with MW+IR (highest spatial resolution but time-varying effective resolution since IR resolution depends on clouds), False = L4 MUR SST with just MW (lower res but more constant spatial resolution)

test_year = 2019

n_regions = 5615

test_dates = []
for t in range(365):
    test_dates.append(datetime.date(2019,1,1)+datetime.timedelta(days=t))

    
save_dir = './pre-processed/testing'

def save_batches(region):
    
    print(region)
    
    raw_dir = f'./raw/{region}/'

    files_raw = os.listdir(raw_dir)

    files_tracks = [f for f in files_raw if 'tracks' in f]

    files_sst_hr = [f for f in files_raw if 'sst_hr' in f]

    input_data_final = np.zeros((395,n,n,2))
    # output_npy = np.zeros((395,n_obs_max,3))
    max_lengths = []
    start_date = datetime.date(2019,1,1)-datetime.timedelta(days = N_t/2)
    output_data_final = []
    n_tot = []
    for t in range(395):
        date_loop = start_date + datetime.timedelta(days = t)
        
        if t==0:
            bathymetry = np.load(raw_dir+'bathymetry.npy')
            mdt = np.load(raw_dir+'mdt.npy')

        
        # 
        # for t_loop in range(N_t):
        
        ssh_files = [f for f in files_tracks if f'{date_loop}' in f]
        sst_hr_files = [f for f in files_sst_hr if f'{date_loop}' in f]
        sst_lr_files = [f for f in files_sst_lr if f'{date_loop}' in f]
        n_tot.append(len(ssh_files)) # number of sats passing over on that day
        if len(sst_hr_files)>0:
            try:
                sst_loop_hr = np.load(raw_dir+sst_hr_files[0])
            except:
                sst_loop_hr = np.zeros((n,n))
        else:
            sst_loop_hr = np.zeros((n,n))
        
        data_tracks = []
        sats = []
        for f in ssh_files:
            try:
                data_tracks.append(np.load(raw_dir+f)[1:,:])
                sats.append(f[11:-15])
            except: 
                data_tracks.append(np.zeros((1,3)))
        input_ssh = bin_ssh(data_tracks,sats,L_x,L_y, n, filtered)
    
        input_data_final[t,:,:,0] = sst_loop_hr
        input_data_final[t,:,:,1] = input_ssh
        


    np.save(save_dir + f'/input_data_region{region}.npy', input_data_final)
    
def worker(lock, batches):
    while True:
        #acquire lock to check and update the directories list
        with lock:
            if not batches:
                break  

            batch = batches.pop(0)  # Get the next directory
            print(f"Worker {multiprocessing.current_process().name} processing batch: {batch}")

        save_batches(batch)

def create_sublists(large_list, n):
    sublists = [[] for _ in range(n)]

    for i, element in enumerate(large_list):
        sublist_index = i % n
        sublists[sublist_index].append(element)

    return sublists

if __name__ == '__main__':
    centers = [i for i in range(n_regions)]
    
    lock = multiprocessing.Lock()
    num_workers = 12
    batches_split = create_sublists(centers, num_workers)
    
    processes = []
    
    for i in range(num_workers):
        worker_batches = batches_split[i]
        
        process = multiprocessing.Process(target=worker, args=(lock, worker_batches))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()    

