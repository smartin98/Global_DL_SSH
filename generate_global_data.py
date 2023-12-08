# 2023-04-07 Scott Martin
# Revised and optimised data generation code for working on global SSH product

# this code defines a fixed grid of lon/lat points which are approximately equispaced by a distance of L km. These grid points will be the centers of the local patches used to create the global product. The code interpolates netcdf satellite datasets to .npy files containing data on local orthonormal projection grids for every day for the full record considered. These data will later be split for training-validation-testing purposes.

# variables to be interpolated:
    # CMEMS L3 SLA observations (un-gridded, time dependent)
    # CMEMS MDT (gridded, constant in t, lower res than target grid so INTERPOLATE)
    # GEBCO Bathymetry (gridded, constant in t,higher res so BIN AVERAGE)
    # GHRSST MUR L4 SST (gridded, time-dependent, higher res so BIN AVERAGE)
    

import numpy as np
from numpy.random import randint
import pyproj
import scipy.spatial.transform 
import scipy.stats as stats
from scipy import interpolate
import matplotlib.path as mpltPath
import xarray as xr 
import time
from datetime import date, timedelta, datetime
import os
import multiprocessing
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from random import shuffle
import copy
from global_land_mask import globe

from global_data_utils import *

############ DEFINITIONS ######################

# Define the pyproj transformer objects used to transform coordinates between (lat,long,alt) and ECEF in both directions
transformer_ll2xyz = pyproj.Transformer.from_crs(
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        )
transformer_xyz2ll = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        )


# generate the lat/lon coords to center the local patches on (RUN FIRST TIME ONLY AND SAVE COORDS FOR FUTURE USE):

L = 250 # spacing of reconstruction patches in km
R = 6378 # radius of earth in km
# lat-lon extent of global grid of reconstruction patches:
lat_min = -70
lat_max = 65
lon_min = -180
lon_max = 180
# create mesh of roughly equally spaced reconstruction patch centers:
dtheta = np.rad2deg(L/R)
lat = np.linspace(lat_min,lat_max,int((lat_max-lat_min)/dtheta))
coords = np.empty((int(1e5),2))
count = 0
dphis = np.zeros_like(lat)
for i in range(lat.shape[0]):
    dphi = np.rad2deg(L/(R*np.abs(np.cos(np.deg2rad(lat[i])))))
    lon_loop = lon_min + dphi/2
    while lon_loop<lon_max:
        coords[count,0] = lon_loop
        coords[count,1] = lat[i]
        count+=1
        lon_loop+=dphi
    if lon_loop-lon_max<=0.5*dphi:
        lon_loop = lon_max-dphi/2
        coords[count,0] = lon_loop
        coords[count,1] = lat[i]
        count+=1

coords = coords[:count,:]

# PART 2 SINCE WENT BACK AND ADDED HIGHER LATITUDES LATER
# lat-lon extent of global grid of reconstruction patches:
lat_min = 65
lat_max = 80
lon_min = -180
lon_max = 180
# create mesh of roughly equally spaced reconstruction patch centers:
dtheta = np.rad2deg(L/R)
lat = np.linspace(lat_min,lat_max,int((lat_max-lat_min)/dtheta))
coords2 = np.empty((int(1e5),2))
count = 0
dphis = np.zeros_like(lat)
for i in range(lat.shape[0]):
    dphi = np.rad2deg(L/(R*np.abs(np.cos(np.deg2rad(lat[i])))))
    lon_loop = lon_min + dphi/2
    while lon_loop<lon_max:
        coords2[count,0] = lon_loop
        coords2[count,1] = lat[i]
        count+=1
        lon_loop+=dphi
    if lon_loop-lon_max<=0.5*dphi:
        lon_loop = lon_max-dphi/2
        coords2[count,0] = lon_loop
        coords2[count,1] = lat[i]
        count+=1

coords2 = coords2[:count,:]


coords = np.concatenate([coords,coords2],axis=0)
# remove land points:
idx_ocean = []
for i in range(count):
    if ~globe.is_land(coords[i,1], coords[i,0]):
        idx_ocean.append(i)
ocean_coords = np.zeros((len(idx_ocean),2))
for i in range(len(idx_ocean)):
    ocean_coords[i,:] = coords[idx_ocean[i],:]
np.save('./coord_grids.npy',ocean_coords)


regions = np.load('./coord_grids.npy')
lon0 = 0.25*(regions[:,63,63,0]+regions[:,64,63,0]+regions[:,63,64,0]+regions[:,64,64,0])
lat0 = 0.25*(regions[:,63,63,1]+regions[:,64,63,1]+regions[:,63,64,1]+regions[:,64,64,1])
ocean_coords = np.stack((lon0,lat0),axis=-1)


date_start = date(2023,3,20)
date_end = date(2023,5,20)
n_days = (date_end-date_start).days
# n_centers = len(idx_ocean)
n = 128 # pixels in nxn local grids
L_x = 960e3 # size of local grid in m
L_y = 960e3 # size of local grid in m


data_bath = xr.open_dataset(os.path.expanduser('~')+'./gebco_bathymetry_4x_coarsened.nc') # NOT INCLUDED IN REPO, DOWNLOAD GEBCO BATHYMETRY FROM https://www.gebco.net/data_and_products/gridded_bathymetry_data/#global (this dataset is much higher res than needed so this code runs faster if you first coarsen by a factor of 4 in both lon and lat and save for future use.)
data_duacs = xr.open_dataset('./cnes_obs-sl_glo_phy-mdt_my_0.125deg_P20Y_1681506488705.nc') # NOT INCLUDED IN REPO BUT AVAILABLE FROM AVISO+/CMEMS


sst_hr_dir = './sst high res/' # PATH TO DOWNLOADED L4 MUR 0.01 DEGREE SST NETCDF FILES (AVAILABLE ON PODAAC)

files_sst_hr = GetListOfFiles(sst_hr_dir)


# pre-process function, works on 1 patch center at a time for parallelisation
def save_files(center):
    
    save_dir = f'./raw/{center}/' # data to save pre-processed data subsets (~1-10 TB) depending on number of years
     
    print(f'STARTING REGION {center}')
    lon0 = ocean_coords[center,0]
    lat0 = ocean_coords[center,1]
    coord_grid = grid_coords(data_bath, n, L_x, L_y, lon0, lat0)

    for t in range(n_days):
        # start = time.time() 
        date_loop = date_start + timedelta(days=t)

        if date_loop>date(2020,12,31):
            nrt=True
        else:
            nrt = False

        if nrt == False:
            satellites = ['alg','tpn','tp','s3b','s3a','j3','j2n','j2g','j2','j1n','j1g','j1','h2b','h2ag','h2a','g2','enn','en','e2','e1g','al','c2','c2n']
            sat_dir = './l3 sla data/' #PATH TO DIRECTORY WHERE L3 SLA DATA DOWNLOADED: https://doi.org/10.48670/moi-00146
        else:
            satellites = ['s3a','s3b','s6a','j3','j3n','al','c2n','h2b']
            sat_dir = './l3 sla data nrt/' #PATH TO DIRECTORY WHERE L3 SLA DATA FOR YEARS COVERED BY NRT PRODUCT DOWNLOADED: https://doi.org/10.48670/moi-00147
        print(date_loop)

        # extract MDT
        if t==0:
            tri_mdt = mdt_delaunay(data_duacs, n, L_x, L_y, lon0, lat0)
            mdt = grid_mdt(data_duacs, 128, L_x, L_y, lon0, lat0,tri_mdt)

            np.save(save_dir+'mdt.npy',mdt)

        # extract along-track SSH obs:
        for s in range(len(satellites)):
            files_tracked = GetListOfFiles(sat_dir+satellites[s])
            if nrt==False:
                file = [f for f in files_tracked if f'_{date_loop}_'.replace('-','') in f]
            else:
                file = [f for f in files_tracked if f'_{date_loop}'.replace('-','') in f]
            if len(file)>0:
                data_tracked = xr.open_dataset(file[0])
                tracks = extract_tracked(data_tracked, L_x, L_y, lon0, lat0, transformer_ll2xyz, nrt)
                if tracks.shape[0]>5: # discard really short tracks
                    np.save(save_dir+'ssh_tracks_'+satellites[s]+f'_{date_loop}.npy',tracks)
            elif len(file)>1:
                raise Exception("len(sla file)>1")

        # grid high res SST:
        file_sst_hr = [f for f in files_sst_hr if f'/{date_loop}'.replace('-','') in f]
        if len(file_sst_hr)==1:
            data_sst_hr = xr.open_dataset(file_sst_hr[0])
            sst_hr = grid_sst_hr(data_sst_hr, n, L_x, L_y, lon0, lat0, coord_grid)
        elif len(file_sst_hr)>1:
            print(file_sst_hr)
            raise Exception("len(file_sst_hr)>1") 
        else:
            sst_hr = np.zeros((n,n))

        np.save(save_dir+'sst_hr_'+f'{date_loop}.npy',sst_hr)

        
    print(f'FINISHED REGION {center}')


################
# helper functions to apply the pre-processing function in parallel across available CPUs since this is the slowest part of the workflow, only needs to be done once though... 


def worker(lock, centers):
    while True:
        # acquire lock to check and update the directories list
        with lock:
            if not centers:
                break  # no more directories to process

            center = centers.pop(0)  # get next directory
            print(f"Worker {multiprocessing.current_process().name} processing center: {center}")

        save_files(center)

def create_sublists(large_list, n):
    sublists = [[] for _ in range(n)]

    for i, element in enumerate(large_list):
        sublist_index = i % n
        sublists[sublist_index].append(element)

    return sublists

if __name__ == '__main__':
    centers = [i for i in range(5615)]
    
    lock = multiprocessing.Lock()
    num_workers = 16
    centers_split = create_sublists(centers, num_workers)
    
    processes = []
    
    for i in range(num_workers):
        worker_centers = centers_split[i]

        process = multiprocessing.Process(target=worker, args=(lock, worker_centers))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
    
                
                
            
                    
                    
