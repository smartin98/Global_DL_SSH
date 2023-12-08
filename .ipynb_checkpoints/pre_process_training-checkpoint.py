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

# take available along-track altimetry, randomly select up to n_sats_max sats on each day to use as input, bin average input sats onto zero-padded grid, save output sat(s) un-binned for use in loss function:
def bin_ssh(data_tracks,L_x,L_y, n, n_sats_max, filtered = False):
    random.shuffle(data_tracks)

    if len(data_tracks)>n_sats_max+1:
        tracks_in = np.concatenate(data_tracks[:n_sats_max], axis = 0)
        tracks_out = np.concatenate(data_tracks[n_sats_max:], axis = 0)
    elif len(data_tracks)==1:
        tracks_in = data_tracks[0]
    else:
        tracks_in = np.concatenate(data_tracks[:(len(data_tracks)-1)], axis = 0)
        tracks_out = np.concatenate(data_tracks[(len(data_tracks)-1):], axis = 0)
    
    if filtered:
        input_grid, _,_,_ = stats.binned_statistic_2d(tracks_in[:,0], tracks_in[:,1], tracks_in[:,-2], statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
        input_grid = np.rot90(input_grid)
        input_grid[np.isnan(input_grid)] = 0
        if len(data_tracks)>1:
            output_tracks = np.stack((tracks_out[:,0],tracks_out[:,1],tracks_out[:,-2]),axis=-1)
            output_tracks[np.isnan(output_tracks)] = 0
        else:
            output_tracks = np.zeros((1,3))
    else:
        input_grid, _,_,_ = stats.binned_statistic_2d(tracks_in[:,0], tracks_in[:,1], tracks_in[:,-1], statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
        input_grid = np.rot90(input_grid)

        input_grid[np.isnan(input_grid)] = 0
        
        if len(data_tracks)>1:
            output_tracks = np.stack((tracks_out[:,0],tracks_out[:,1],tracks_out[:,-1]),axis=-1)
            output_tracks[np.isnan(output_tracks)] = 0
        else:
            output_tracks = np.zeros((1,3))
    
    return input_grid, output_tracks

sats_all = ['alg','tpn','tp','s3b','s3a','j3','j2n','j2g','j2','j1n','j1g','j1','h2b','h2ag','h2a','g2','enn','en','e2','e1g','al','c2','c2n','s3b','s6a','j3n','h2b']

test_sats = ['alg','al'] # independent test satellite used for testing purposes, withhold from training data for all years

sats = [s for s in sats_all if s not in test_sats]

batch_size = 25
n_obs_max = 400 # max number of SSH observations on any day in loss function, allows to have fixed size inputs/outputs with zero padding making it easier to create TFRecord dataset
N_t = 30 # length of single input time series in days
n = 128 # no. grid points per side of domain
L_x = 960e3 # size of domain
L_y = 960e3  # size of domain
n_sats_max = 6 # maximum number of altimeters to use in input
filtered = False # whether to use the 65km band-pass filtered or unfiltered SSH observations
test_year = 2019

n_regions = 5615

start_date = datetime.date(2010,1,1)
end_date = datetime.date(2022,12,31)
n_days = (end_date-start_date).days + 1
val_dates = []
for t in range(73-N_t):
    val_dates.append(datetime.date(2010,1,1)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2011,1,1)+datetime.timedelta(days = 73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2012,1,1)+datetime.timedelta(days = 2*73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2013,1,1)+datetime.timedelta(days = 3*73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2014,1,1)+datetime.timedelta(days = 4*73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2016,1,1)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2017,1,1)+datetime.timedelta(days = 73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2018,1,1)+datetime.timedelta(days = 2*73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2020,1,1)+datetime.timedelta(days = 3*73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2021,1,1)+datetime.timedelta(days = 4*73)+datetime.timedelta(days = 15+t))
test_dates = []
for t in range(365):
    test_dates.append(datetime.date(2019,1,1)+datetime.timedelta(days=t))

train_dates = []
for t in range(n_days-15):
    check_date = start_date+ datetime.timedelta(days=t)
    diffs_val = [np.abs((check_date-date).days) for date in val_dates]
    diffs_test = [np.abs((check_date-date).days) for date in test_dates]
    
    if (np.min(diffs_val)>=30) and (np.min(diffs_test)>=30):
        train_dates.append(check_date)

save_regions = False
mode = 'training' # 'validation'
domain = 'global'
if mode == 'training':
    save_dir = 'pre-processed/training'
elif mode == 'validation':
    save_dir = 'pre-processed/validation'

regions_available = np.array([r for r in range(5615)])

def save_batch(batch):
    batch_no = batch
    filename = save_dir+f'/batch_{batch_no}.tfrecord'

    input_data_final = np.zeros((batch_size,N_t+1,n,n,2))
    output_npy = np.zeros((batch_size,N_t,n_obs_max,3))
    max_lengths = []
    regions = np.zeros(batch_size,dtype='int')
    for sample in range(batch_size):
        trying=True
        while trying:
            r = np.random.randint(0,regions_available.shape[0])
            r = regions_available[r]
            raw_dir = f'raw/{r}/'
            if mode=='training':
                available_dates = train_dates
            elif mode=='validation':
                available_dates = val_dates
            mid_date = random.choice(available_dates)

            files_raw = os.listdir(raw_dir)

            files_tracks = [f for f in files_raw if 'tracks' in f]
            files_tracks = [f for f in files_tracks if not any(substring in f for substring in test_sats)] # removes the test sat for all years
            files_sst = [f for f in files_raw if 'sst_hr' in f]
            
            bathymetry = np.load(raw_dir+'bathymetry.npy')
            mdt = np.load(raw_dir+'mdt.npy')

            output_data_final = []
            n_tot = []
            for t_loop in range(N_t):
                date_loop = mid_date - datetime.timedelta(days = N_t/2-t_loop)
                ssh_files = [f for f in files_tracks if f'{date_loop}' in f]
                sst_files = [f for f in files_sst if f'{date_loop}' in f]
                n_tot.append(len(ssh_files)) # number of sats passing over on that day
                if len(sst_files)>0:
                    try:
                        sst_loop = np.load(raw_dir+sst_files[0])
                    except:
                        sst_loop = np.zeros((n,n))
                else:
                    sst_loop = np.zeros((n,n))
                data_tracks = []
                for f in ssh_files:
                    try:
                        data_tracks.append(np.load(raw_dir+f)[1:,:])
                    except: 
                        data_tracks.append(np.zeros((1,3)))
                if len(data_tracks)>0:
                    input_ssh, output_ssh = bin_ssh(data_tracks,L_x,L_y, n, n_sats_max, filtered)
                else:
                    input_ssh = np.zeros((n,n))
                    output_ssh = np.zeros((1,3))
                input_data_final[sample,t_loop,:,:,0] = input_ssh
                input_data_final[sample,t_loop,:,:,1] = sst_loop
                output_data_final.append(output_ssh)

            input_data_final[sample,-1,:,:,0] = bathymetry
            input_data_final[sample,-1,:,:,1] = mdt
            lengths = []
            for i in range(len(output_data_final)):
                lengths.append(output_data_final[i].shape[0])
            for i in range(N_t):
                if lengths[i]<n_obs_max:
                    output_npy[sample,i,:lengths[i],:] = output_data_final[i]
                else:
                    output_npy[sample,i,:,:] = output_data_final[i][:n_obs_max,:]
            sst_total = input_data_final[sample,:-1,:,:,1]
            # condition to exclude examples with extreme sea ice cover:
            if (np.size(sst_total[sst_total==0])<0.9*np.size(sst_total)) or (np.sum(n_tot)/N_t>1):
                trying = False

            regions[sample] = int(r)
    if save_regions:
        np.save(save_dir+'_regions'+f'/batch_{batch}_regions.npy',regions)

    

    writer = tf.io.TFRecordWriter(filename)
    serialized_example = serialize_example(input_data_final, output_npy)
    writer.write(serialized_example)

    
def worker(lock, batches, seed):
    np.random.seed(seed)
    while True:
        #acquire lock to check and update the directories list
        with lock:
            if not batches:
                break  

            batch = batches.pop(0)  
            print(f"Worker {multiprocessing.current_process().name} processing batch: {batch}")

        save_batch(batch)

def create_sublists(large_list, n):
    sublists = [[] for _ in range(n)]

    for i, element in enumerate(large_list):
        sublist_index = i % n
        sublists[sublist_index].append(element)

    return sublists

if __name__ == '__main__':
    centers = [i for i in range(4000)] # number of batches to process
    
    lock = multiprocessing.Lock()
    num_workers = 8 # number of CPUs to parallelise across
    batches_split = create_sublists(centers, num_workers)
    
    random_seeds = [np.random.randint(0, 100000) for _ in range(num_workers)]
    
    processes = []
    
    for i in range(num_workers):
        worker_batches = batches_split[i]
        random_seed = random_seeds[i]
        
        process = multiprocessing.Process(target=worker, args=(lock, worker_batches, random_seed))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()    

    
    
    
        
        
        
