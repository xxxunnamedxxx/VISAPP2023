


import xarray as xr
import numpy
import pyinterp
import pyinterp.fill
import logging 
import matplotlib.pylab as plt
from matplotlib.ticker import ScalarFormatter
import xarray as xr
import numpy
import logging
import xrft
from dask.diagnostics import ProgressBar

def rmse_based_scores(ds_oi, ds_ref):
    
    logging.info('     Compute RMSE-based scores...')
    
    # RMSE(t) based score
    rmse_t = 1.0 - (((ds_oi['sossheig'] - ds_ref['sossheig'])**2).mean(dim=('lon', 'lat')))**0.5/(((ds_ref['sossheig'])**2).mean(dim=('lon', 'lat')))**0.5
    # RMSE(x, y) based score
    # rmse_xy = 1.0 - (((ds_oi['sossheig'] - ds_ref['sossheig'])**2).mean(dim=('time')))**0.5/(((ds_ref['sossheig'])**2).mean(dim=('time')))**0.5
    rmse_xy = (((ds_oi['sossheig'] - ds_ref['sossheig'])**2).mean(dim=('time')))**0.5
    
    rmse_t = rmse_t.rename('rmse_t')
    rmse_xy = rmse_xy.rename('rmse_xy')

    # Temporal stability of the error
    reconstruction_error_stability_metric = rmse_t.std().values

    # Show leaderboard SSH-RMSE metric (spatially and time averaged normalized RMSE)
    leaderboard_rmse = 1.0 - (((ds_oi['sossheig'] - ds_ref['sossheig']) ** 2).mean()) ** 0.5 / (
        ((ds_ref['sossheig']) ** 2).mean()) ** 0.5

    logging.info('          => Leaderboard SSH RMSE score = %s', numpy.round(leaderboard_rmse.values, 4))
    logging.info('          Error variability = %s (temporal stability of the mapping error)', numpy.round(reconstruction_error_stability_metric, 4))
    
    return rmse_t, rmse_xy, numpy.round(leaderboard_rmse.values, 4), numpy.round(reconstruction_error_stability_metric, 4)


def psd_based_scores(ds_oi, ds_ref):
    
    logging.info('     Compute PSD-based scores...')
    
    with ProgressBar():
        
        # Compute error = SSH_reconstruction - SSH_true
        err = (ds_oi['sossheig'] - ds_ref['sossheig'])
        err = err.chunk({"lat":1, 'time': err['time'].size, 'lon': err['lon'].size})
        # make time vector in days units 
        err['time'] = (err.time - err.time[0]) / numpy.timedelta64(1, 'D')
        
        # Rechunk SSH_true
        signal = ds_ref['sossheig'].chunk({"lat":1, 'time': ds_ref['time'].size, 'lon': ds_ref['lon'].size})
        # make time vector in days units
        signal['time'] = (signal.time - signal.time[0]) / numpy.timedelta64(1, 'D')
    
        # Compute PSD_err and PSD_signal
        psd_err = xrft.power_spectrum(err, dim=['time', 'lon'], detrend='constant', window=True).compute()
        psd_signal = xrft.power_spectrum(signal, dim=['time', 'lon'], detrend='constant', window=True).compute()
        
        # Averaged over latitude
        mean_psd_signal = psd_signal.mean(dim='lat').where((psd_signal.freq_lon > 0.) & (psd_signal.freq_time > 0), drop=True)
        mean_psd_err = psd_err.mean(dim='lat').where((psd_err.freq_lon > 0.) & (psd_err.freq_time > 0), drop=True)
        
        # return PSD-based score
        psd_based_score = (1.0 - mean_psd_err/mean_psd_signal)

        # Find the key metrics: shortest temporal & spatial scales resolved based on the 0.5 contour criterion of the PSD_score

        

        level = [0.5]
        cs = plt.contour(1./psd_based_score.freq_lon.values,1./psd_based_score.freq_time.values, psd_based_score, level)
        x05, y05 = cs.collections[0].get_paths()[0].vertices.T
        plt.close()
        
        shortest_spatial_wavelength_resolved = numpy.min(x05)
        shortest_temporal_wavelength_resolved = numpy.min(y05)

        logging.info('          => Leaderboard Spectral score = %s (degree lon)',
                     numpy.round(shortest_spatial_wavelength_resolved, 2))
        logging.info('          => shortest temporal wavelength resolved = %s (days)',
                     numpy.round(shortest_temporal_wavelength_resolved, 2))

        return (1.0 - mean_psd_err/mean_psd_signal), numpy.round(shortest_spatial_wavelength_resolved, 4), numpy.round(shortest_temporal_wavelength_resolved, 4)
def read_obs(input_file, oi_grid, oi_param, simu_start_date, coarsening):
    
    logging.info('     Reading observations...')
    
    def preprocess(ds):
        return ds.coarsen(coarsening, boundary="trim").mean()
    
    ds_obs = xr.open_mfdataset(input_file, combine='nested', concat_dim='time', parallel=True, preprocess=preprocess) #.sortby('time')
    #ds_obs = ds_obs.coarsen(coarsening, boundary="trim").mean().sortby('time')
    ds_obs = ds_obs.sortby('time')
    
    lon_min = oi_grid.lon.min().values
    lon_max = oi_grid.lon.max().values
    lat_min = oi_grid.lat.min().values
    lat_max = oi_grid.lat.max().values
    time_min = oi_grid.time.min().values
    time_max = oi_grid.time.max().values
    
    ds_obs = ds_obs.sel(time=slice(time_min - numpy.timedelta64(int(2*oi_param.Lt.values), 'D'), 
                                   time_max + numpy.timedelta64(int(2*oi_param.Lt.values), 'D')), drop=True)
    
    # correct lon if domain is between [-180:180]
    if lon_min < 0:
        ds_obs['lon'] = xr.where(ds_obs['lon'] >= 180., ds_obs['lon']-360., ds_obs['lon'])
        
    ds_obs = ds_obs.where((ds_obs['lon'] >= lon_min - oi_param.Lx.values) & 
                          (ds_obs['lon'] <= lon_max + oi_param.Lx.values) &
                          (ds_obs['lat'] >= lat_min - oi_param.Ly.values) &
                          (ds_obs['lat'] <= lat_max + oi_param.Ly.values) , drop=True)
    
    vtime = (ds_obs['time'].values - numpy.datetime64(simu_start_date)) / numpy.timedelta64(1, 'D')
    ds_obs = ds_obs.assign_coords({'time': vtime})
    
    return ds_obs
def oi_regrid(ds_source, ds_target):
    
    logging.info('     Regridding...')
    
    # Define source grid
    x_source_axis = pyinterp.Axis(ds_source["lon"][:].values, is_circle=False)
    y_source_axis = pyinterp.Axis(ds_source["lat"][:].values)
    z_source_axis = pyinterp.TemporalAxis(ds_source["time"][:].values)
    ssh_source = ds_source["gssh"][:].T
    grid_source = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis, ssh_source.data)
    
    # Define target grid
    mx_target, my_target, mz_target = numpy.meshgrid(ds_target['lon'].values,
                                                     ds_target['lat'].values,
                                                     z_source_axis.safe_cast(ds_target['time'].values),
                                                     indexing="ij")
    # Spatio-temporal Interpolation
    ssh_interp = pyinterp.trivariate(grid_source,
                                     mx_target.flatten(),
                                     my_target.flatten(),
                                     mz_target.flatten(),
                                     bounds_error=False).reshape(mx_target.shape).T
    
    # MB add extrapolation in NaN values if needed
    if numpy.isnan(ssh_interp).any():
        logging.info('     NaN found in ssh_interp, starting extrapolation...')
        x_source_axis = pyinterp.Axis(ds_target['lon'].values, is_circle=False)
        y_source_axis = pyinterp.Axis(ds_target['lat'].values)
        z_source_axis = pyinterp.TemporalAxis(ds_target["time"][:].values)
        grid = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis,  ssh_interp.T)
        has_converged, filled = pyinterp.fill.gauss_seidel(grid)
    else:
        filled = ssh_interp.T
    
    # Save to dataset
    ds_ssh_interp = xr.Dataset({'sossheig' : (('time', 'lat', 'lon'), filled.T)},
                               coords={'time': ds_target['time'].values,
                                       'lon': ds_target['lon'].values, 
                                       'lat': ds_target['lat'].values, 
                                       })
    
    return ds_ssh_interp