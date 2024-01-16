import numpy as np
import os
import xarray as xr
from glob import glob

input_dir = 'path to directory containing SSH maps created using with_grads = True'
output_dir = 'path to directory to store SSH maps with surface currents, vorticity, and strain'
ice_path = 'path to sea ice concentration NetCDF available in the Harvard Dataverse repo'

g = 9.81
om = 2*np.pi/86164

ds_ice = xr.open_dataset(ice_path)
ds_ice = ds_ice.sel(latitude=slice(-70,80),drop=True)

files = sorted(glob(input_dir+'*.nc'))
files_short = [f[len(input_dir):] for f in files]

for i, file in enumerate(files_short):
    ice = ds_ice.isel(time = i)
    print(file)
    ds = xr.open_dataset(input_dir+file)
    ice = ice.interp_like(ds.isel(time=0))

    f = 2*om*np.sin(np.deg2rad(ds['latitude']))
    ds['ugosa'] = (-g/f)*ds['dSLA_dy']
    ds['vgosa'] = (g/f)*ds['dSLA_dx']
    ds['vorticity'] = (g/f)*(ds['d2SLA_dx2']+ds['d2SLA_dy2'])
    ds['strain'] = (g/np.abs(f))*np.sqrt(4*ds['d2SLA_dxy']**2+(ds['d2SLA_dx2']-ds['d2SLA_dy2'])**2)
    equator_mask = (ds['sla'].latitude >= -5) & (ds['sla'].latitude <= 5)
    ds['ugosa'] = (ds['ugosa']).where(~equator_mask)
    ds['vgosa'] = (ds['vgosa']).where(~equator_mask)
    ds['vorticity'] = (ds['vorticity']).where(~equator_mask)
    ds['strain'] = (ds['strain']).where(~equator_mask)
    
    ds = ds.drop(['dSLA_dx','dSLA_dy','d2SLA_dx2','d2SLA_dy2','d2SLA_dxy'])
    ice_arr = np.array(ice['ice_conc'])
    ice_arr[np.isnan(ice_arr)] = 0 # just avoids land mask growing when applying the ice mask since sea ice data is lower resolution than the SSH maps
    ice['ice_conc'] = (['latitude','longitude'],ice_arr)
    ds = ds.where(ice['ice_conc']<1) # keep only pixels where ice concentration <1%
    ds.to_netcdf(output_dir+file[:-11]+'20240115.nc')

