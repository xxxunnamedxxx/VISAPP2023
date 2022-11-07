








import matplotlib.pyplot as plt
plt.close("all")
import xarray as xr
import numpy
import os
import sys
sys.path.append("../")
import numpy as np

def define_grid(lat_min=33,lat_max=43,lon_min=-65,lon_max=-55,
            t_min=np.datetime64('2012-10-01'),t_max=np.datetime64('2013-10-01'),
            dx=0.05,dt=1):
    LAT=np.arange(lat_min, lat_max,dx)
    LON=np.arange(lon_min, lon_max,dx)
    TIME=np.arange(t_min, t_max,dt)
    return LAT,LON,TIME

PATH="Data/"
nc_files=os.listdir(PATH+"raw")
nc_files.sort()
print("FILES :")
print("-----------")

for file in nc_files:
    print(file)
print("-----------")
print("")


#___________GRID PARAM________________
lat_min=33
lat_max=43
lon_min=-65
lon_max=-55
t_min=np.datetime64('2012-10-01')
t_max=np.datetime64('2013-10-01')
dx=0.05
dt=1
LAT,LON,TIME=define_grid(lat_min,lat_max,lon_min,lon_max,t_min,t_max,dx,dt)


satellite="_swotnadir.nc"
satellite="_nadir.nc"
satellite="_swot.nc"


dic={}

for file in nc_files:
    if "REF_ssh" in file:
        ds=xr.open_mfdataset(PATH+"raw/"+file)
        ds=ds.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min-dx,lon_max-dx))
        dic["SSH"]=(("time",'lat', 'lon'), ds.ssh.data)
        
    elif "REF_sst" in file:
        ds=xr.open_mfdataset(PATH+"raw/"+file)
        ds=ds.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min-dx,lon_max-dx))
        dic["SST"]=(("time",'lat', 'lon'), ds.sst.data)

    elif satellite in file:
        if "OBS" in file:
            ds=xr.open_mfdataset(PATH+"raw/"+file)
            ds=ds.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min-dx,lon_max-dx))
            dic["OBS"]=(("time",'lat', 'lon'), ds.ssh_obs.data)
            dic["OBS_MOD"]=(("time",'lat', 'lon'), ds.ssh_mod.data)
            
        if "DUACS" in file:
            ds=xr.open_mfdataset(PATH+"raw/"+file)
            ds=ds.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min-dx,lon_max-dx))
            dic["DUACS"]=(("time",'lat', 'lon'), ds.ssh_obs.data)
            dic["DUACS_MOD"]=(("time",'lat', 'lon'), ds.ssh_mod.data)
    

ds_final = xr.Dataset(dic,
                      coords={'time': TIME,
                                      'lon': LON, 
                                      'lat': LAT,})

ds_final.to_netcdf(PATH+"/processed/"+"ds"+satellite)














