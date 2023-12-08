import xarray as xr
import numpy as np
from scipy import interpolate

def add_ghost_points(ds,lon_name='longitude',lon_bounds=[0,360]):
    
    """
    Add extra columns to xarray dataset at the edges of the longitude domain that are the average of the two edges of the dataset to ensure it wraps fully and NaNs don't appear at the edges when interpolating to a finer grid

    Args:
        ds: regular gridded xarray dataset with a longitude dimension that is intended to be periodic
        lon_name: string, key for the longitude variable
        lon_bounds: list of 2 floats, indicates whether range is [-180,180] or [0, 360]

    Returns:
        ds: xarray dataset with extra ghost point added at each edge of the dataset
    """
    
    if ds[lon_name][-1] != lon_bounds[-1]:
        ds = xr.concat([ds, ds.isel({lon_name: 0})], dim=lon_name)

    if ds[lon_name][0] != lon_bounds[0]:
        ds = xr.concat([ds.isel({lon_name: -1}), ds], dim=lon_name)

    for var in ds.data_vars:
        ds[var][{ds[lon_name].name: 0}] = ds[var][{ds[lon_name].name: -1}] = (
        ds[var][{ds[lon_name].name: 0}] + ds[var][{ds[lon_name].name: -1}]
        ) / 2.0
    lon = np.array(ds[lon_name])
    lon[0] = lon_bounds[0]
    lon[-1] = lon_bounds[-1]

    ds[lon_name] = lon
    return ds

def spectral_slope(k,s,n=200):
    
    """
    Calculate spectral slope of 1D KE spectrum

    Args:
        k: 1D numpy array, wavenumbers can be irregularly spaced
        s: 1D numpy array, corresponding KE spectrum values
        n: integer, number of points to resample to in log space

    Returns:
        k_uniform: 1D numpy array of size n, unformly spaced wavenumbers
        ds_dk: 1D numpy array of size n, spectral slope
    """
    
    log_k = np.log(k)
    log_k_uniform = np.linspace(log_k.min(),log_k.max(),n)
    f = interpolate.interp1d(log_k, s)
    
    s_new = f(log_k_uniform)
    ds_dk = np.gradient(np.log(s_new),log_k_uniform[1]-log_k_uniform[0])
    
    return np.exp(log_k_uniform), ds_dk

def azimuthal_average(data):
    center = (np.array(data.shape) - 1) / 2.0
    y, x = np.indices(data.shape)
    r_f = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r_f.astype(np.int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def kinetic_energy_spectrum_1d(u, v, dx, dy):
    
    fft_u = np.fft.fft2(u)
    fft_u_shifted = np.abs(np.fft.fftshift(fft_u))
    
    fft_v = np.fft.fft2(v)
    fft_v_shifted = np.abs(np.fft.fftshift(fft_v))
    
    fft_magnitude_shifted = 0.5*(fft_u_shifted**2+fft_v_shifted**2)
    
    Nx, Ny = u.shape
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
    dk = np.abs(kx[1]-kx[0])
    
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    
    k = np.fft.fftshift(np.sqrt(kx_grid**2 + ky_grid**2))
        
    energy_spectrum = azimuthal_average(fft_magnitude_shifted)
    k_spectrum = azimuthal_average(k)
    energy_spectrum = energy_spectrum*2*np.pi*k_spectrum
    
    return k_spectrum, energy_spectrum