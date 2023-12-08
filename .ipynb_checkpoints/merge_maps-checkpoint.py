import numpy as np
import datetime
import multiprocessing
from src.merging import *
import gc
import os

def worker(lock, batches):
    while True:
        # Acquire the lock to check and update the directories list
        with lock:
            if not batches:
                break  # No more directories to process

            batch = batches.pop(0)  # Get the next directory
        available_regions=np.array([i for i in range(5615)])#np.load('regions_south.npy')
        print(np.size(available_regions))
        merge_maps_and_save(pred_dir, pred_file_pattern, batch, output_nc_dir, mask_filename, dist_filename, mdt_filename, network_name, available_regions,L=250e3, crop_pixels=9, dx=7.5e3, with_grads=True, mask_coast_dist=0, lon_min=-180 ,lon_max=180, lat_min=-70, lat_max=80, res=1/10, progress=False)
        gc.collect()

def create_sublists(large_list, n):
    sublists = [[] for _ in range(n)]

    for i, element in enumerate(large_list):
        sublist_index = i % n
        sublists[sublist_index].append(element)

    return sublists

if __name__ == '__main__':
    pred_dir = './preds_refactored/'
    pred_file_pattern = 'simvp_ssh_sst_ns1000000global_pred_nsats6_'
    pred_dates = [datetime.date(2019,1,1)+datetime.timedelta(days=t) for t in range(365)] 
    output_nc_dir = './SimVP SSH-SST 1M 6sat grads/'
    os.system("mkdir "+output_nc_dir)
    mask_filename = './land_water_mask_10grid.nc' # find in 2023a Ocean Data Challenge
    dist_filename = './distance_to_nearest_coastlines_10grid.nc' # find in 2023a Ocean Data Challenge
    mdt_filename = './mdt_hybrid_cnes_cls18_cmems2020_global.nc' # Chosen MDT, available from AVISO+/CMEMS
    network_name = 'SimVP_SSH_SST_1M_global'
    N_workers = 6 #number of cpus to parallelise across
    
    centers = pred_dates
    
    lock = multiprocessing.Lock()
    num_workers = N_workers
    batches_split = create_sublists(centers, num_workers)
   
    processes = []
    
    for i in range(num_workers):
        worker_batches = batches_split[i]

        process = multiprocessing.Process(target=worker, args=(lock, worker_batches))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()    
