import numpy as np
from src.global_data_utils import *
from global_land_mask import globe

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
count2 = 0
dphis = np.zeros_like(lat)
for i in range(lat.shape[0]):
    dphi = np.rad2deg(L/(R*np.abs(np.cos(np.deg2rad(lat[i])))))
    lon_loop = lon_min + dphi/2
    while lon_loop<lon_max:
        coords2[count2,0] = lon_loop
        coords2[count2,1] = lat[i]
        count2+=1
        lon_loop+=dphi
    if lon_loop-lon_max<=0.5*dphi:
        lon_loop = lon_max-dphi/2
        coords2[count2,0] = lon_loop
        coords2[count2,1] = lat[i]
        count2+=1

coords2 = coords2[:count2,:]


count = count + count2

coords = np.concatenate([coords,coords2],axis=0)
# remove land points:
idx_ocean = []
for i in range(coords.shape[0]):
    if ~globe.is_land(coords[i,1], coords[i,0]):
        idx_ocean.append(i)
ocean_coords = np.zeros((len(idx_ocean),2))
for i in range(len(idx_ocean)):
    ocean_coords[i,:] = coords[idx_ocean[i],:]

n = 128 # pixels in nxn local grids
L_x = 960e3 # size of local grid in m
L_y = 960e3 # size of local grid in m


data_bath = xr.open_dataset(os.path.expanduser('~')+'./gebco_bathymetry_4x_coarsened.nc') # NOT INCLUDED IN REPO, DOWNLOAD GEBCO BATHYMETRY FROM https://www.gebco.net/data_and_products/gridded_bathymetry_data/#global (this dataset is much higher res than needed so this code runs faster if you first coarsen by a factor of 4 in both lon and lat and save for future use.)

coords_data = np.zeros((ocean_coords.shape[0], n, n, 2))

for r in range(ocean_coords.shape[0]):
    lon0, lat0 = ocean_coords[r,0], ocean_coords[r,1]
    
    coord_grid = grid_coords(data_bath, n, L_x, L_y, lon0, lat0)
    
    coords_data[r,] = coord_grid
    
np.save('./coord_grids.npy', coords_data)