import numpy as np
import xarray as xr
from glob import glob
from pyproj import Proj
from scipy import interpolate
import pandas as pd

mdt_path = 'path to netcdf containing MDT velocities (download from dataverse)'
simvp_path = 'path to directory containing SimVP maps (download from dataverse)'
duacs_path = 'path to directory containing DUACS maps (download from https://github.com/ocean-data-challenges/2023a_SSH_mapping_OSE)'
ice_path = 'path to NetCDF containing sea ice mask for 2019 (download from dataverse)'
save_dir = 'path to directory to save coarse graining subsets'


ds_mdt = xr.open_dataset(mdt_path)


time_min = '2019-01-01'
time_max = '2019-12-31'
list_of_files = sorted(glob(simvp_path + '/*.nc'))
ds = xr.open_mfdataset(list_of_files, combine='by_coords')
ds = ds.sel(time=slice(time_min,time_max), drop = True)
ds = ds.sel(latitude=slice(-70,80),drop=True)


list_of_files = sorted(glob(duacs_path+'/*.nc'))
ds_duacs = xr.open_mfdataset(list_of_files, combine='by_coords')
ds_duacs = ds_duacs.sel(latitude=slice(-70,80),drop=True)
ds_duacs['longitude'] = ds_duacs['longitude'].assign_attrs({'units':'degrees_east','_CoordinateAxisType':'Lon'})
ds_ice = xr.open_dataset(ice_path)
ds_ice = ds_ice.sel(latitude=slice(-70,80),drop=True)
ds_ice = ds_ice.reindex(time=ds_duacs['time'], method='nearest')
ds_duacs = ds_duacs.where(ds_ice['ice_conc']<1) # Make sea ice mask same as used for SimVP


def subset(ds, ds_mdt, lon0, lat0, domain_name, duacs):
    print(domain_name)
    padding = 20
    lon_min = lon0 - padding
    lon_max = lon0 + padding
    lat_min = lat0 - padding
    lat_max = lat0 + padding
    
    ds_crop = ds.sel(longitude=slice(lon_min,lon_max),drop=True).sel(latitude=slice(lat_min,lat_max),drop=True)
    ds_mdt_crop = ds_mdt.sel(longitude=slice(lon_min,lon_max),drop=True).sel(latitude=slice(lat_min,lat_max),drop=True)
    
    lon = ds_crop['longitude']
    lat = ds_crop['latitude']
    lon,lat = np.meshgrid(lon, lat)
    lon = lon.flatten()
    lat = lat.flatten()
    proj = Proj(proj='laea', lat_0=lat0, lon_0=lon0, units='m', datum='WGS84')
    x, y = proj(lon, lat)
    ugosa = np.array(ds_crop['ugosa'].load())
    vgosa = np.array(ds_crop['vgosa'].load())
    if not duacs:
        ugos = ugosa + np.array(ds_mdt_crop['u'])
        vgos = vgosa + np.array(ds_mdt_crop['v'])
    else:
        ugos = np.array(ds_crop['ugos'].load())
        vgos = np.array(ds_crop['vgos'].load())

    if duacs:
        ugos = np.moveaxis(ugos, [0,1,2], [2,0,1])
        vgos = np.moveaxis(vgos, [0,1,2], [2,0,1])
    
    x_grid = np.arange(-128*10*1e3,128*10*1e3,10*1e3).astype('float')
    y_grid = np.arange(-128*10*1e3,128*10*1e3,10*1e3).astype('float')
    x_grid,y_grid = np.meshgrid(x_grid,y_grid)
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()

    ugos_interp = np.zeros((256,256,365))
    vgos_interp = np.zeros((256,256,365))
    
    for t in range(365):
        if t%10==0:
            print(t)
        arrs = [ugos[:,:,t],vgos[:,:,t]]
        for i, arr in enumerate(arrs):
            x_loop = x.copy()
            y_loop = y.copy()
            var = arr.flatten()
            var = var[x_loop>-1280e3-30e3]
            y_loop = y_loop[x_loop>-1280e3-30e3]
            x_loop = x_loop[x_loop>-1280e3-30e3]

            var = var[x_loop<1280e3+30e3]
            y_loop = y_loop[x_loop<1280e3+30e3]
            x_loop = x_loop[x_loop<1280e3+30e3]

            var = var[y_loop<1280e3+30e3]
            x_loop = x_loop[y_loop<1280e3+30e3]
            y_loop = y_loop[y_loop<1280e3+30e3]

            var = var[y_loop>-1280e3-30e3]
            x_loop = x_loop[y_loop>-1280e3-30e3]
            y_loop = y_loop[y_loop>-1280e3-30e3]

            interpolator = interpolate.LinearNDInterpolator(np.stack((x_loop,y_loop),axis=-1),var)

            var_interp = interpolator(x_grid, y_grid)
            var_interp = var_interp.reshape(256,256)

            if i==0:
                ugos_interp[:,:,t] = var_interp
            elif i==1:
                vgos_interp[:,:,t] = var_interp
                
    ugos_interp[np.isnan(ugos_interp)] = 0
    vgos_interp[np.isnan(vgos_interp)] = 0
        
    start_date = pd.Timestamp("2019-01-01")
    end_date = pd.Timestamp("2019-12-31")
    time = pd.date_range(start=start_date, end=end_date)
    
    da_ugos = xr.DataArray(np.expand_dims(np.moveaxis(ugos_interp, [0,1,2], [1,2,0]),axis=1), dims=("time","depth","latitude", "longitude"), coords={"time": time, "depth":np.zeros(1),"latitude": np.arange(-128*10*1e3,128*10*1e3,10*1e3).astype('float'), "longitude": np.arange(-128*10*1e3,128*10*1e3,10*1e3).astype('float')})
    da_vgos = xr.DataArray(np.expand_dims(np.moveaxis(vgos_interp, [0,1,2], [1,2,0]),axis=1), dims=("time","depth","latitude", "longitude"), coords={"time": time, "depth":np.zeros(1),"latitude": np.arange(-128*10*1e3,128*10*1e3,10*1e3).astype('float'), "longitude": np.arange(-128*10*1e3,128*10*1e3,10*1e3).astype('float')})
    
    ds_gos = xr.Dataset({"uo":da_ugos,"vo":da_vgos})
    
    if duacs:
        ds_gos.to_netcdf(save_dir+domain_name+f'_duacs_{lon0}_{lat0}.nc')
    else:
        ds_gos.to_netcdf(save_dir+domain_name+f'_simvp_{lon0}_{lat0}.nc')
    

## N ATLANTIC
lon0 = 330
lat0 = 45
subset(ds ,ds_mdt , lon0, lat0, domain_name='natlantic', duacs=False)
subset(ds_duacs, ds_mdt, lon0, lat0, domain_name='natlantic', duacs=True)

## STCC
lon0 = 155
lat0 = 25
subset(ds_duacs, ds_mdt, lon0, lat0, domain_name='stcc', duacs=True)
subset(ds ,ds_mdt , lon0, lat0, domain_name='stcc', duacs=False)

## S PACIFIC

lon0 = 200
lat0 = -25
subset(ds ,ds_mdt , lon0, lat0, domain_name='spacific', duacs=False)
subset(ds_duacs, ds_mdt, lon0, lat0, domain_name='spacific', duacs=True)

## KUROSHIO

lon0 = 160
lat0 = 35
subset(ds_duacs, ds_mdt, lon0, lat0, domain_name='kuroshio', duacs=True)
subset(ds ,ds_mdt , lon0, lat0, domain_name='kuroshio', duacs=False)
