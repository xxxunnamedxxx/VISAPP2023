
import numpy as np
import pyinterp
import logging
import xarray as xr


def compute_segment_alongtrack(time_alongtrack, 
                               lat_alongtrack, 
                               lon_alongtrack, 
                               ssh_alongtrack, 
                               ssh_map_interp, 
                               lenght_scale,
                               delta_x,
                               delta_t):

    segment_overlapping = 0.25
    max_delta_t_gap = 4 * np.timedelta64(1, 's')  # max delta t of 4 seconds to cut tracks

    list_lat_segment = []
    list_lon_segment = []
    list_ssh_alongtrack_segment = []
    list_ssh_map_interp_segment = []

    # Get number of point to consider for resolution = lenghtscale in km
    delta_t_jd = delta_t / (3600 * 24)
    npt = int(lenght_scale / delta_x)

    # cut track when diff time longer than 4*delta_t
    indi = np.where((np.diff(time_alongtrack) > max_delta_t_gap))[0]
    track_segment_lenght = np.insert(np.diff(indi), [0], indi[0])

    # Long track >= npt
    selected_track_segment = np.where(track_segment_lenght >= npt)[0]

    if selected_track_segment.size > 0:

        for track in selected_track_segment:

            if track-1 >= 0:
                index_start_selected_track = indi[track-1]
                index_end_selected_track = indi[track]
            else:
                index_start_selected_track = 0
                index_end_selected_track = indi[track]

            start_point = index_start_selected_track
            end_point = index_end_selected_track

            for sub_segment_point in range(start_point, end_point - npt, int(npt*segment_overlapping)):

                # Near Greenwhich case
                if ((lon_alongtrack[sub_segment_point + npt - 1] < 50.)
                    and (lon_alongtrack[sub_segment_point] > 320.)) \
                        or ((lon_alongtrack[sub_segment_point + npt - 1] > 320.)
                            and (lon_alongtrack[sub_segment_point] < 50.)):

                    tmp_lon = np.where(lon_alongtrack[sub_segment_point:sub_segment_point + npt] > 180,
                                       lon_alongtrack[sub_segment_point:sub_segment_point + npt] - 360,
                                       lon_alongtrack[sub_segment_point:sub_segment_point + npt])
                    mean_lon_sub_segment = np.median(tmp_lon)

                    if mean_lon_sub_segment < 0:
                        mean_lon_sub_segment = mean_lon_sub_segment + 360.
                else:

                    mean_lon_sub_segment = np.median(lon_alongtrack[sub_segment_point:sub_segment_point + npt])

                mean_lat_sub_segment = np.median(lat_alongtrack[sub_segment_point:sub_segment_point + npt])

                ssh_alongtrack_segment = np.ma.masked_invalid(ssh_alongtrack[sub_segment_point:sub_segment_point + npt])

                ssh_map_interp_segment = []
                ssh_map_interp_segment = np.ma.masked_invalid(ssh_map_interp[sub_segment_point:sub_segment_point + npt])
                if np.ma.is_masked(ssh_map_interp_segment):
                    ssh_alongtrack_segment = np.ma.compressed(np.ma.masked_where(np.ma.is_masked(ssh_map_interp_segment), ssh_alongtrack_segment))
                    ssh_map_interp_segment = np.ma.compressed(ssh_map_interp_segment)

                if ssh_alongtrack_segment.size > 0:
                    list_ssh_alongtrack_segment.append(ssh_alongtrack_segment)
                    list_lon_segment.append(mean_lon_sub_segment)
                    list_lat_segment.append(mean_lat_sub_segment)
                    list_ssh_map_interp_segment.append(ssh_map_interp_segment)


    return list_lon_segment, list_lat_segment, list_ssh_alongtrack_segment, list_ssh_map_interp_segment, npt 




def compute_spectral_scores(time_alongtrack, 
                            lat_alongtrack, 
                            lon_alongtrack, 
                            ssh_alongtrack, 
                            ssh_map_interp, 
                            lenght_scale,
                            delta_x,
                            delta_t,
                            output_filename):
    
    # make time vector as days since 1950-01-01
    #time_alongtrack = (time_alongtrack - np.datetime64('1950-01-01T00:00:00Z')) / np.timedelta64(1, 'D')
    
    # compute segments
    lon_segment, lat_segment, ref_segment, study_segment, npt  = compute_segment_alongtrack(time_alongtrack, 
                                                                                            lat_alongtrack, 
                                                                                            lon_alongtrack, 
                                                                                            ssh_alongtrack, 
                                                                                            ssh_map_interp, 
                                                                                            lenght_scale,
                                                                                            delta_x,
                                                                                            delta_t)
    
    # Power spectrum density reference field
    global_wavenumber, global_psd_ref = scipy.signal.welch(np.asarray(ref_segment).flatten(),
                                                           fs=1.0 / delta_x,
                                                           nperseg=npt,
                                                           scaling='density',
                                                           noverlap=0)

    # Power spectrum density study field
    _, global_psd_study = scipy.signal.welch(np.asarray(study_segment).flatten(),
                                             fs=1.0 / delta_x,
                                             nperseg=npt,
                                             scaling='density',
                                             noverlap=0)

    # Power spectrum density study field
    _, global_psd_diff = scipy.signal.welch(np.asarray(study_segment).flatten()-np.asarray(ref_segment).flatten(),
                                            fs=1.0 / delta_x,
                                            nperseg=npt,
                                            scaling='density',
                                            noverlap=0)
    
    # Save psd in netcdf file
    ds = xr.Dataset({"psd_ref": (["wavenumber"], global_psd_ref),
                     "psd_study": (["wavenumber"], global_psd_study),
                     "psd_diff": (["wavenumber"], global_psd_diff),
                    },
                    coords={"wavenumber": (["wavenumber"], global_wavenumber)},
                   )
    
    ds.to_netcdf(output_filename)
    logging.info(f'  Results saved in: {output_filename}')

def read_l3_dataset(file,
                    lon_min=0., 
                    lon_max=360., 
                    lat_min=-90, 
                    lat_max=90., 
                    time_min='1900-10-01', 
                    time_max='2100-01-01'):
    
    ds = xr.open_dataset(file)
    ds = ds.sel(time=slice(time_min, time_max), drop=True)
    ds = ds.where((ds["latitude"] >= lat_min) & (ds["latitude"] <= lat_max), drop=True)
    ds = ds.where((ds["longitude"] >= lon_min%360.) & (ds["longitude"] <= lon_max%360.), drop=True)
    
    return ds

def read_l3_dataset_from_aviso(url_dataset, 
                               my_aviso_session,
                               lon_min=0., 
                               lon_max=360., 
                               lat_min=-90, 
                               lat_max=90., 
                               time_min='1900-10-01', 
                               time_max='2100-01-01'):
    
    # disable logger for library
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("pydap").setLevel(logging.WARNING)

    store = xr.backends.PydapDataStore.open(url_dataset, session=my_aviso_session)
    ds = xr.open_dataset(store)
    ds = ds.sel(time=slice(time_min, time_max), drop=True)
    ds = ds.where((ds["latitude"] >= lat_min) & (ds["latitude"] <= lat_max), drop=True)
    ds = ds.where((ds["longitude"] >= lon_min%360.) & (ds["longitude"] <= lon_max%360.), drop=True)
    
    return ds

def read_l4_dataset(list_of_file, 
                    lon_min=0., 
                    lon_max=360., 
                    lat_min=-90, 
                    lat_max=90., 
                    time_min='1900-10-01', 
                    time_max='2100-01-01', 
                    is_circle=True):
    
    if isinstance(list_of_file, str):
        ds = xr.open_mfdataset(list_of_file)
    else :
        ds=list_of_file
    # print(ds)
    
    # ds= ds.rename({"longitude":"lon","latitude":"lat"})
    ds = ds.sel(time=slice(time_min, time_max), drop=True)
    ds = ds.where((ds["lon"]%360. >= lon_min) & (ds["lon"]%360. <= lon_max), drop=True)
    ds = ds.where((ds["lat"] >= lat_min) & (ds["lat"] <= lat_max), drop=True)
    
    x_axis = pyinterp.Axis(ds["lon"][:]%360., is_circle=is_circle)
    y_axis = pyinterp.Axis(ds["lat"][:])
    z_axis = pyinterp.TemporalAxis(np.array(ds["time"][:]))
    
    var = ds['ssh'][:]
    var = var.transpose('lon', 'lat', 'time')

    # The undefined values must be set to nan.
    try:
        var[var.mask] = float("nan")
    except AttributeError:
        pass
    
    grid = pyinterp.Grid3D(x_axis, y_axis, z_axis, var.data)
    
    del ds
    
    return x_axis, y_axis, z_axis, grid

def interp_on_alongtrack(gridded_dataset, 
                         ds_alongtrack,
                         lon_min=0., 
                         lon_max=360., 
                         lat_min=-90, 
                         lat_max=90., 
                         time_min='1900-10-01', 
                         time_max='2100-01-01',
                         is_circle=True):
    
    # Interpolate maps onto alongtrack dataset
    if isinstance(gridded_dataset, str):
        x_axis, y_axis, z_axis, grid = read_l4_dataset(gridded_dataset,
                                                       lon_min=lon_min,
                                                       lon_max=lon_max, 
                                                       lat_min=lat_min,
                                                       lat_max=lat_max, 
                                                       time_min=time_min,
                                                       time_max=time_max,
                                                       is_circle=is_circle)
    elif isinstance(gridded_dataset, list):
        
        x_axis, y_axis, z_axis, grid = read_l4_dataset_from_aviso(gridded_dataset[0],
                                                                  gridded_dataset[1],
                                                                  lon_min=lon_min,
                                                                  lon_max=lon_max, 
                                                                  lat_min=lat_min,
                                                                  lat_max=lat_max, 
                                                                  time_min=time_min,
                                                                  time_max=time_max,
                                                                  is_circle=is_circle)
    elif type(gridded_dataset)==xr.core.dataset.Dataset:
        x_axis, y_axis, z_axis, grid = read_l4_dataset(gridded_dataset,
                                                   lon_min=lon_min,
                                                   lon_max=lon_max, 
                                                   lat_min=lat_min,
                                                   lat_max=lat_max, 
                                                   time_min=time_min,
                                                   time_max=time_max,
                                                   is_circle=is_circle)
    
    ssh_map_interp = pyinterp.trivariate(grid, 
                                         ds_alongtrack["longitude"].values, 
                                         ds_alongtrack["latitude"].values,
                                         z_axis.safe_cast(ds_alongtrack.time.values),
                                         bounds_error=False).reshape(ds_alongtrack["longitude"].values.shape)
    
    ssh_alongtrack = (ds_alongtrack["sla_unfiltered"] + ds_alongtrack["mdt"] - ds_alongtrack["lwe"]).values

    lon_alongtrack = ds_alongtrack["longitude"].values
    lat_alongtrack = ds_alongtrack["latitude"].values
    time_alongtrack = ds_alongtrack["time"].values
    
    # get and apply mask from map_interp & alongtrack on each dataset
    msk1 = np.ma.masked_invalid(ssh_alongtrack).mask
    msk2 = np.ma.masked_invalid(ssh_map_interp).mask
    msk = msk1 + msk2
    
    ssh_alongtrack = np.ma.masked_where(msk, ssh_alongtrack).compressed()
    lon_alongtrack = np.ma.masked_where(msk, lon_alongtrack).compressed()
    lat_alongtrack = np.ma.masked_where(msk, lat_alongtrack).compressed()
    time_alongtrack = np.ma.masked_where(msk, time_alongtrack).compressed()
    ssh_map_interp = np.ma.masked_where(msk, ssh_map_interp).compressed()
    
    # select inside value (this is done to insure similar number of point in statistical comparison between methods)
    indices = np.where((lon_alongtrack >= lon_min+0.25) & (lon_alongtrack <= lon_max-0.25) &
                       (lat_alongtrack >= lat_min+0.25) & (lat_alongtrack <= lat_max-0.25))[0]
    
    return time_alongtrack[indices], lat_alongtrack[indices], lon_alongtrack[indices], ssh_alongtrack[indices], ssh_map_interp[indices]
    

def write_timeserie_stat(ssh_alongtrack, ssh_map_interp, time_vector, freq):
    
    
    diff = ssh_alongtrack - ssh_map_interp
    # convert data vector and time vector into xarray.Dataarray
    da = xr.DataArray(diff, coords=[time_vector], dims="time")
    
    # resample 
    da_resample = da.resample(time=freq)
    
    # compute stats
    vmean = da_resample.mean()
    vminimum = da_resample.min()
    vmaximum = da_resample.max()
    vcount = da_resample.count()
    vvariance = da_resample.var()
    vmedian = da_resample.median()
    vrms = np.sqrt(np.square(da).resample(time=freq).mean())
    
    rmse = np.copy(vrms)
    
    # save stat to dataset
    # ds = xr.Dataset(
    #     {
    #         "mean": (("time"), vmean.values),
    #         "min": (("time"), vminimum.values),
    #         "max": (("time"), vmaximum.values),
    #         "count": (("time"), vcount.values),
    #         "variance": (("time"), vvariance.values),
    #         "median": (("time"), vmedian.values),
    #         "rms": (("time"), vrms.values),            
    #     },
    #     {"time": vmean['time']},
    # )
    
    # ds.to_netcdf(output_filename, group='diff')
    
    
    # convert data vector and time vector into xarray.Dataarray
    da = xr.DataArray(ssh_alongtrack, coords=[time_vector], dims="time")
    
    # resample 
    da_resample = da.resample(time=freq)
    
    # compute stats
    vmean = da_resample.mean()
    vminimum = da_resample.min()
    vmaximum = da_resample.max()
    vcount = da_resample.count()
    vvariance = da_resample.var()
    vmedian = da_resample.median()
    vrms = np.sqrt(np.square(da).resample(time=freq).mean())
    
    rms_alongtrack = np.copy(vrms)
    
    # save stat to dataset
    # ds = xr.Dataset(
    #     {
    #         "mean": (("time"), vmean.values),
    #         "min": (("time"), vminimum.values),
    #         "max": (("time"), vmaximum.values),
    #         "count": (("time"), vcount.values),
    #         "variance": (("time"), vvariance.values),
    #         "median": (("time"), vmedian.values),
    #         "rms": (("time"), vrms.values),            
    #     },
    #     {"time": vmean['time']},
    # )
    
    # ds.to_netcdf(output_filename, group='alongtrack', mode='a')
    
    
    # convert data vector and time vector into xarray.Dataarray
    da = xr.DataArray(ssh_map_interp, coords=[time_vector], dims="time")
    
    # resample 
    da_resample = da.resample(time=freq)
    
    # compute stats
    vmean = da_resample.mean()
    vminimum = da_resample.min()
    vmaximum = da_resample.max()
    vcount = da_resample.count()
    vvariance = da_resample.var()
    vmedian = da_resample.median()
    vrms = np.sqrt(np.square(da).resample(time=freq).mean())
    
    # save stat to dataset
    # ds = xr.Dataset(
    #     {
    #         "mean": (("time"), vmean.values),
    #         "min": (("time"), vminimum.values),
    #         "max": (("time"), vmaximum.values),
    #         "count": (("time"), vcount.values),
    #         "variance": (("time"), vvariance.values),
    #         "median": (("time"), vmedian.values),
    #         "rms": (("time"), vrms.values),            
    #     },
    #     {"time": vmean['time']},
    # )
    
    # ds.to_netcdf(output_filename, group='maps', mode='a')
    
    # logging.info(' ')
    # logging.info(f'  Results saved in: {output_filename}')
    
    rmse_score = 1. - rmse/rms_alongtrack
    # mask score if nb obs < nb_min_obs
    nb_min_obs = 10
    rmse_score = np.ma.masked_where(vcount.values < nb_min_obs, rmse_score)
    
    mean_rmse = np.ma.mean(np.ma.masked_invalid(rmse_score))
    std_rmse = np.ma.std(np.ma.masked_invalid(rmse_score))
    
    # logging.info(' ')
    # logging.info(f'  MEAN RMSE Score = {mean_rmse}')
    # logging.info(' ')
    # logging.info(f'  STD RMSE Score = {std_rmse}')
    
    return mean_rmse, std_rmse

def compute_stats(time_alongtrack, 
                  lat_alongtrack, 
                  lon_alongtrack, 
                  ssh_alongtrack, 
                  ssh_map_interp, 
                  bin_lon_step,
                  bin_lat_step, 
                  bin_time_step,
                  output_filename,
                  output_filename_timeseries):

    # ncfile = netCDF4.Dataset(output_filename,'w')

    # binning = pyinterp.Binning2D(
    #     pyinterp.Axis(np.arange(0, 360, bin_lon_step), is_circle=True),
    #     pyinterp.Axis(np.arange(-90, 90 + bin_lat_step, bin_lat_step)))

    # # binning alongtrack
    # binning.push(lon_alongtrack, lat_alongtrack, ssh_alongtrack, simple=True)
    # write_stat(ncfile, 'alongtrack', binning)
    # binning.clear()

    # # binning map interp
    # binning.push(lon_alongtrack, lat_alongtrack, ssh_map_interp, simple=True)
    # write_stat(ncfile, 'maps', binning)
    # binning.clear()

    # # binning diff sla-msla
    # binning.push(lon_alongtrack, lat_alongtrack, ssh_alongtrack - ssh_map_interp, simple=True)
    # write_stat(ncfile, 'diff', binning)
    # binning.clear()

    # # add rmse
    # diff2 = (ssh_alongtrack - ssh_map_interp)**2
    # binning.push(lon_alongtrack, lat_alongtrack, diff2, simple=True)
    # var = ncfile.groups['diff'].createVariable('rmse', binning.variable('mean').dtype, ('lat','lon'), zlib=True)
    # var[:, :] = np.sqrt(binning.variable('mean')).T  
    
    # ncfile.close()
    
    # logging.info(f'  Results saved in: {output_filename}')

    # write time series statistics
    leaderboard_nrmse, leaderboard_nrmse_std = write_timeserie_stat(ssh_alongtrack, 
                                                                    ssh_map_interp, 
                                                                    time_alongtrack, 
                                                                    bin_time_step, 
                                                                    )
    
    return leaderboard_nrmse, leaderboard_nrmse_std



