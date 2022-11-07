

#preprocess data

import xarray as xr
import requests as rq
import sys
import numpy
import hvplot.xarray
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as T
import torch
from utils import *
plt.close("all")


#_________________________________SSH SATELLITE_____________________________


my_aviso_session = rq.Session()
my_aviso_session.auth = ("<AVISO_LOGIN>", "<AVISO_PWD>")
url_alongtrack = 'https://tds.aviso.altimetry.fr/thredds/dodsC/2021a-SSH-mapping-OSE-along-track-data'
url_map = 'https://tds.aviso.altimetry.fr/thredds/dodsC/2021a-SSH-mapping-OSE-grid-data'

def read_l3_dataset_from_aviso(url_dataset, 
                               my_aviso_session,
                               lon_min=0., 
                               lon_max=360., 
                               lat_min=-90, 
                               lat_max=90., 
                               time_min='1900-10-01', 
                               time_max='2100-01-01'):
    
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("pydap").setLevel(logging.WARNING)

    store = xr.backends.PydapDataStore.open(url_dataset, session=my_aviso_session)
    ds = xr.open_dataset(store)
    ds = ds.sel(time=slice(time_min, time_max), drop=True)
    ds = ds.where((ds["latitude"] >= lat_min) & (ds["latitude"] <= lat_max), drop=True)
    ds = ds.where((ds["longitude"] >= lon_min%360.) & (ds["longitude"] <= lon_max%360.), drop=True)
    
    return ds



inputs = [f'{url_alongtrack}/dt_gulfstream_alg_phy_l3_20161201-20180131_285-315_23-53.nc', 
          f'{url_alongtrack}/dt_gulfstream_j3_phy_l3_20161201-20180131_285-315_23-53.nc', 
          f'{url_alongtrack}/dt_gulfstream_s3a_phy_l3_20161201-20180131_285-315_23-53.nc',
          f'{url_alongtrack}/dt_gulfstream_h2g_phy_l3_20161201-20180131_285-315_23-53.nc',
          f'{url_alongtrack}/dt_gulfstream_j2g_phy_l3_20161201-20180131_285-315_23-53.nc',
          f'{url_alongtrack}/dt_gulfstream_j2n_phy_l3_20161201-20180131_285-315_23-53.nc',
          f'{url_alongtrack}/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc']
 
lon_lat=(295., 
305.0, 
33, 
43.0, 
'1900-10-01', 
'2100-01-01')




ds_alg = read_l3_dataset_from_aviso(inputs[0], my_aviso_session,*lon_lat)
print(ds_alg)
# ds_alg.to_netcdf("Data/raw/ds_alg2.nc")

ds_j3  = read_l3_dataset_from_aviso(inputs[1], my_aviso_session,*lon_lat)
print(ds_j3)
# ds_j3.to_netcdf("Data/raw/ds_j3.nc")

ds_s3a = read_l3_dataset_from_aviso(inputs[2], my_aviso_session,*lon_lat)
print(ds_s3a)
# ds_s3a.to_netcdf("Data/raw/ds_s3a.nc")

ds_h2g = read_l3_dataset_from_aviso(inputs[3], my_aviso_session,*lon_lat)
print(ds_h2g)
# ds_h2g.to_netcdf("Data/raw/ds_h2g.nc")

ds_j2g = read_l3_dataset_from_aviso(inputs[4], my_aviso_session,*lon_lat)
print(ds_j2g)
# ds_j2g.to_netcdf("Data/raw/ds_j2g.nc")

ds_j2n = read_l3_dataset_from_aviso(inputs[5], my_aviso_session,*lon_lat)
print(ds_j2n)
# ds_j2n.to_netcdf("Data/raw/ds_j2n.nc")

ds_c2  = read_l3_dataset_from_aviso(inputs[6], my_aviso_session,*lon_lat)
print(ds_c2)
# ds_c2.to_netcdf("Data/raw/ds_c2.nc")


def grid_dayly_nadir(ds_source):
    
    if len(ds_source)>0:
        ds_concat_nadirs = xr.concat(ds_source, dim='time')
        ds_concat_nadirs = ds_concat_nadirs.sortby(ds_concat_nadirs.time)
        ds_concat_nadirs = ds_concat_nadirs.assign_coords({'longitude': ds_concat_nadirs.longitude, 'latitude': ds_concat_nadirs.latitude})
    else :
        ds_concat_nadirs=ds_source[0]


    #CALCULATE SSH 
    # if add_mdt:
    SSH_u_linear= (ds_concat_nadirs["sla_unfiltered"] + ds_concat_nadirs["mdt"] - ds_concat_nadirs["lwe"])
    SSH_f_linear=(ds_concat_nadirs["sla_filtered"] + ds_concat_nadirs["mdt"] - ds_concat_nadirs["lwe"])
    # else:
    SLA_u_linear= (ds_concat_nadirs["sla_unfiltered"] - ds_concat_nadirs["lwe"])
    SLA_f_linear=(ds_concat_nadirs["sla_filtered"] - ds_concat_nadirs["lwe"])
    
    # SSH_u_linear= (ds_concat_nadirs["sla_unfiltered"]  - ds_concat_nadirs["lwe"])
    # SSH_f_linear=(ds_concat_nadirs["sla_filtered"]  - ds_concat_nadirs["lwe"])
    
    #DEFINE TIME SERIES
    # t_min,t_max=np.datetime64(SSH_u_linear.time.values[0],"D"),np.datetime64(SSH_u_linear.time.values[-1],"D")+1
    t_min,t_max=np.datetime64("2016-12-01"),np.datetime64("2018-01-31")+1

    TIME=np.arange(t_min,t_max)
    Nt=len(TIME)
    
    #DEFINE GRID
    Nx=200
    Ny=200
    lat_min=33.
    lat_max=43.
    lon_min=-65.+360
    lon_max=-55.+360
    LAT=np.arange(lat_min,lat_max,(lat_max-lat_min)/Ny)
    LON=np.arange(lon_min,lon_max,(lon_max-lon_min)/Nx)
    
    # DEFINE DX, DY, DT
    dx=0.05
    dy=0.05
    dt=np.timedelta64(1,"D")
    
    #DEFINE EMPTY GRID 
    ssh_u=np.empty((Nt,Nx,Ny))
    ssh_u[:]=[np.NaN]
    ssh_f=np.empty((Nt,Nx,Ny))
    ssh_f[:]=[np.NaN]
    sla_u=np.empty((Nt,Nx,Ny))
    sla_u[:]=[np.NaN]
    sla_f=np.empty((Nt,Nx,Ny))
    sla_f[:]=[np.NaN]
    nb_points=np.zeros((Nt,Nx,Ny))
    L=SSH_u_linear.shape[0]
    
    
    
    for n in range(SSH_u_linear.shape[0]):
        val_u_ssh,val_f_ssh,val_u_sla,val_f_sla=SSH_u_linear[n] ,SSH_f_linear[n] ,SLA_u_linear[n] ,SLA_f_linear[n] 
        # print(val_u)
        if n%5000==0:
            print(np.round(100*n/L,2), " % points done")
        time=(val_u_ssh["time"].values-t_min)//dt
        lat=int((val_u_ssh["latitude"].values-lat_min)//dy)
        lon=int((val_u_ssh["longitude"].values-lon_min)//dx)
 
        # print(val_u_ssh)
        i=nb_points[time,lat,lon]
        if i==0:
            ssh_u[time,lat,lon]=val_u_ssh.values
            ssh_f[time,lat,lon]=val_f_ssh.values
            sla_u[time,lat,lon]=val_u_sla.values
            sla_f[time,lat,lon]=val_f_sla.values
            nb_points[time,lat,lon]+=1
    
        else :
            ssh_u[time,lat,lon]=(val_u_ssh.values+ssh_u[time,lat,lon]*i)/(i+1)
            ssh_f[time,lat,lon]=(val_f_ssh.values+ssh_f[time,lat,lon]*i)/(i+1)
            sla_u[time,lat,lon]=(val_u_sla.values+sla_u[time,lat,lon]*i)/(i+1)
            sla_f[time,lat,lon]=(val_f_sla.values+sla_f[time,lat,lon]*i)/(i+1)
            nb_points[time,lat,lon]+=1
    
    ssh_u=xr.DataArray(ssh_u)
    ssh_u=ssh_u.to_masked_array()
    ssh_f=xr.DataArray(ssh_f)
    ssh_f=ssh_f.to_masked_array()
    sla_u=xr.DataArray(sla_u)
    sla_u=sla_u.to_masked_array()
    sla_f=xr.DataArray(sla_f)
    sla_f=sla_f.to_masked_array()
    ds_return = xr.Dataset({'ssh_unfiltered': (("time",'lat', 'lon'), ssh_u),'ssh_filtered': (("time",'lat', 'lon'), ssh_f),
                            'sla_unfiltered': (("time",'lat', 'lon'), sla_u),'sla_filtered': (("time",'lat', 'lon'), sla_f)},
                          coords={'time': TIME ,
                                  'lon': LON ,
                                  'lat': LAT,
                                  })
    
    return ds_return

# GRID NADIR
ds_alg_grid=grid_dayly_nadir([ds_alg])
ds_j3_grid=grid_dayly_nadir([ds_j3])
ds_s3a_grid=grid_dayly_nadir([ds_s3a])
ds_h2g_grid=grid_dayly_nadir([ds_h2g])
ds_j2g_grid=grid_dayly_nadir([ds_j2g])
ds_j2n_grid=grid_dayly_nadir([ds_j2n])
ds_c2_grid=grid_dayly_nadir([ds_c2])


LDS=[ds_alg_grid,ds_j3_grid,ds_s3a_grid,ds_h2g_grid,ds_j2g_grid,ds_j2n_grid,ds_c2_grid]
legend=["alg","j3","s3a","h2g","j2g","j2n","c2"]
satelittes=[]
dt=20
for ds in LDS:
    nb_points=[]
    for t in range(0,ds.ssh_unfiltered.shape[0],dt):
        print(ds.time[t].values)
        ssh=ds.ssh_unfiltered[t:t+dt]
        nb_points.append(np.count_nonzero(~np.isnan(ssh)))
    satelittes.append(nb_points)

satelittes=np.array(satelittes).T
plt.figure()
plt.plot(satelittes)
plt.legend(legend)

#CONCATENATE TRAINING NADIR:
NADIR=[ds_alg_grid,ds_j3_grid,ds_s3a_grid,ds_h2g_grid,ds_j2n_grid] #without c2 and J2G
ssh_filtered=[]
ssh_unfiltered=[]
sla_filtered=[]
sla_unfiltered=[]
for nad in NADIR:
    ssh_filtered.append(nad.ssh_filtered.values)
    ssh_unfiltered.append(nad.ssh_unfiltered.values)
    sla_filtered.append(nad.sla_filtered.values)
    sla_unfiltered.append(nad.sla_unfiltered.values)
ssh_filtered=np.nanmean(np.array(ssh_filtered),axis=0)
ssh_unfiltered=np.nanmean(np.array(ssh_unfiltered),axis=0)
sla_filtered=np.nanmean(np.array(sla_filtered),axis=0)
sla_unfiltered=np.nanmean(np.array(sla_unfiltered),axis=0)

mdt=torch.Tensor(xr.open_mfdataset("Data/raw/mdt.nc").sel(lat=slice(33,43),lon=slice(-65+360,-55+360)).mdt.values).unsqueeze(0)
resize=T.Resize(200)
mdt=resize(mdt).squeeze(0).numpy()

x=ds_c2_grid +ds_alg_grid+ds_j3_grid+ds_h2g_grid
dic={'nadir_ssh_unfiltered': (("time",'lat', 'lon'), ssh_unfiltered),
     'nadir_ssh_filtered': (("time",'lat', 'lon'), ssh_filtered),
     'c2_ssh_unfiltered': (("time",'lat', 'lon'), ds_c2_grid.ssh_unfiltered.values),
     'c2_ssh_filtered': (("time",'lat', 'lon'), ds_c2_grid.ssh_filtered.values),
     'j2g_ssh_unfiltered': (("time",'lat', 'lon'), ds_j2g_grid.ssh_unfiltered.values),
     'j2g_ssh_filtered': (("time",'lat', 'lon'), ds_j2g_grid.ssh_filtered.values),
     
     'nadir_sla_unfiltered': (("time",'lat', 'lon'), sla_unfiltered),
     'nadir_sla_filtered': (("time",'lat', 'lon'), sla_filtered),
     'c2_sla_unfiltered': (("time",'lat', 'lon'), ds_c2_grid.sla_unfiltered.values),
     'c2_sla_filtered': (("time",'lat', 'lon'), ds_c2_grid.sla_filtered.values),
     'j2g_sla_unfiltered': (("time",'lat', 'lon'), ds_j2g_grid.sla_unfiltered.values),
     'j2g_sla_filtered': (("time",'lat', 'lon'), ds_j2g_grid.sla_filtered.values),
    
    "mdt":(('lat', 'lon'),mdt)}



if True :
    
    #SST FROM NASA

    list_of_file=os.listdir(path)    
    list_of_file=["Data/raw/SST/SST_GHRSST_MUR/"+i for i  in list_of_file if i[-3:len(i)]==".nc"]
    list_of_file.sort()
    resize=T.Resize(200)
    i=list_of_file[0]
    L=[]
    for i in list_of_file:
        print("FILE: ", i)
        x=torch.Tensor(xr.open_mfdataset(i).sel(lat=slice(33,43),lon=slice(-65,-55))["analysed_sst"].values[:,:,:])-273.15  
        xx=resize(x)[0]
        L.append(xx.numpy())
       
    tab_sst=np.array(L)
    np.save("Data/raw/SST/SST_GHRSST_MUR/array.npy",tab_sst)
    # plt.figure()
if True:#SST comparison
    #SST GLO NRT http://www.ghrsst.org
    sst_glo_nrt=xr.open_mfdataset("Data/raw/SST/METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2_1662477276483.nc").sel(lat=slice(33,43),lon=slice(-65,-55))
    sst_glo_nrt=sst_glo_nrt.rename({"analysed_sst":"sst"})-273.15
    
    #SST MUR DOI :10.5067/GHGMR-4FJ04
    arr=np.load("Data/raw/SST/SST_GHRSST_MUR/array.npy")
    sst_mur= xr.Dataset({ 'sst': (("time",'lat', 'lon'), arr),},
                             coords={'time': sst_glo_nrt["time"]  ,
                                     'lon':  sst_glo_nrt["lon"] ,
                                     'lat':  sst_glo_nrt["lat"]  })
    
    #SST esa  dx.doi.org/10.5285/62c0f97b1eac4e0197a674870afe1ee6
    sst_esa=  xr.open_mfdataset("Data/raw/SST/SST_PRODUCT_010_024/ds.nc").sel(lat=slice(33,43),lon=slice(-65,-55))
    sst_esa=sst_esa.rename({"analysed_sst":"sst"})-273.15
    
    #glorys
    sst_glorys=  xr.open_mfdataset("Data/raw/SST/GLORYS12V1_PRODUCT_001_030/ds.nc").sel(lat=slice(33,43),lon=slice(-65,-55))
    sst_glorys=sst_glorys
    
    n=100

    l_ds=[sst_glo_nrt,sst_mur,sst_esa,sst_glorys]
    Titres=["SST sat NRT","SST sat DT","SST esa","SST GLORYS"]
    Images=[ds["sst"].values[n,:,:] for ds in l_ds]
    

    # Images=[sst_glo_nrt["sst"].values[n,:,:],sst_mur["sst"].values[n,:,:]]
    plot_n_lines(Images,Titres,cmap="seismic",save_name="here",label="SST(°C)",shrink=0.3,center_colormap=False,fig_width=15)
    
    
    dic["sst_sat_nrt"]=(("time",'lat', 'lon'), sst_glo_nrt["sst"].values)
    dic["sst_sat_mur"]=(("time",'lat', 'lon'), sst_mur["sst"].values)
    dic["sst_sat_esa"]=(("time",'lat', 'lon'), sst_esa["sst"].values)
    dic["sst_mercator"]=(("time",'lat', 'lon'), sst_glorys["sst"].values)

    
    ds_challenge= xr.Dataset(dic,
                             coords={'time': ds_alg_grid["time"]  ,
                                     'lon':  ds_alg_grid["lon"] ,
                                     'lat':  ds_alg_grid["lat"]  })
    ds_challenge.to_netcdf("Data/processed/dsi.nc")


if False:
    path="Data/raw/METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2_1662477276483.nc"
    ds_sst=xr.open_mfdataset(path,engine='netcdf4').sel(lat=slice(33,43),lon=slice(-65,-55))
    tab_sst=ds_sst.analysed_sst.values-273.15
    dic["sst"]=(("time",'lat', 'lon'), tab_sst)




if False:
    
    path="Data/raw/SST/GLORYS12V1_PRODUCT_001_030/"
    l=os.listdir(path)
    l.sort()
    l=[i for i in l if "2016-12" in i]+[i for i in l if "2017" in i]+[i for i in l if "2018-01" in i]
    l=[path+"/"+i for i in l]
    ds=xr.open_mfdataset(l).sortby("time")
    ds=ds.rename({"latitude":"lat","longitude":"lon"})
    ds=ds.sel(lat=slice(33,43),lon=slice(-65,-55))
    
    resize=T.Resize((200,200))

    x=torch.Tensor(ds.sst.values) 
    xx=resize(x).numpy()
    lat=np.arange(33,43,0.05)
    lon=np.arange(-65,-55,0.05)
    ds2=xr.Dataset({"sst":(("time","lat","lon"),xx)},
                   coords={'time': ds["time"]  ,
                           'lon':  lon ,
                           'lat':  lat  }
        )
    ds2.to_netcdf("Data/raw/SST/GLORYS12V1_PRODUCT_001_030/ds.nc")
    
    
    
    # ds.to_netcdf("Data/raw/SST/SST_PRODUCT_010_024/ds.nc")
    
    
if False:
    path="Data/raw/SST/SST_PRODUCT_010_024"
    l=os.listdir(path)
    l.sort()
    l=[i for i in l if "2016-12" in i]+[i for i in l if "2017" in i]+[i for i in l if "2018-01" in i]
    l=[path+"/"+i for i in l]
    ds=xr.open_mfdataset(l).sortby("time")
    ds=ds.rename({"latitude":"lat","longitude":"lon"})
    ds=ds.sel(lat=slice(33,43),lon=slice(-65,-55))
    ds.to_netcdf("Data/raw/SST/SST_PRODUCT_010_024/ds.nc")


if False :
    #_________________________________SST SATELLITE_____________________________
    DS_SST=[]
    time=ds_alg_grid.time.values
    
    for t in time:
        y,m,d=str(t)[0:4],str(t)[5:7],str(t)[8:10]
        
        if int(y)>=2017:
            file="Data/raw/sst/"+y+m+d+"120000-C3S-L4_GHRSST-SSTdepth-OSTIA-GLOB_ICDR2.1-v02.0-fv01.0.nc"
        else:        
            file="Data/raw/sst/"+y+m+d+"120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc"
    
        sst = xr.open_mfdataset(file).sel(lat=slice(33,43),lon=slice(-65,-55))
        DS_SST.append(sst["analysed_sst"][:,:,:])
    
    ds_sst = xr.concat(DS_SST,dim="time")
    tab_sst=ds_sst.values-273.15
    
    dic["sst"]=(("time",'lat', 'lon'), tab_sst)


    


    # plt.subplot(121)
    # plt.imshow(x[0])
    # plt.subplot(122)
    # plt.imshow(xx[0])
    # list_of_file=["Data/raw/SST/SST_GHRSST_MUR/"+i for i  in list_of_file if i[-3:len(i)]==".nc" and i!='20170912090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc']
    # ds_list=[xr.open_mfdataset(i).sel(lat=slice(33,43),lon=slice(-65,-55))["analysed_sst"].values for i  in list_of_file]
     
    # ds_concat_sst = xr.concat(ds_list, dim='time')
    # ds_concat_sst = ds_concat_sst.sortby(ds_concat_sst.time)
    # ds_concat_sst = ds_concat_sst.assign_coords({'lon': ds_concat_sst.lon, 'lat': ds_concat_sst.lat})
    # ds_concat_sst=ds_concat_sst-273.15  
    # sst_tab=ds_concat_sst.values
    # x=torch.Tensor(sst_tab)
    # resize=T.Resize(size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None)
    dic["sst"]=(("time",'lat', 'lon'), tab_sst)
    
    


if False:

    
    a=xr.open_mfdataset("/home/pequan2/archambault/src/Ocean_challenge/Ocean_Challenge_2021/Data/raw/sst/20160826120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc").sel(lat=slice(33,43),lon=slice(-65,-55))
    
    x=a.analysed_sst.values
    plt.figure()
    plt.imshow(x[0,:,:])
    
    x=a.analysed_sst_uncertainty.values
    plt.figure()
    plt.imshow(x[0,:,:])
    
    
    path="/home/pequan2/archambault/Téléchargements/dataset-satellite-sea-surface-temperature-5bb3f8f5-b6e7-4d54-ae1c-ab994d609dfe/20170208120000-C3S-L3C_GHRSST-SSTskin-AVHRRMTA-ICDR2.1_night-v02.0-fv01.0.nc"
    a=xr.open_mfdataset(path,engine='netcdf4').sel(lat=slice(33,43),lon=slice(-65,-55))
    
    x=a.sea_surface_temperature.values
    plt.figure()
    plt.imshow(x[0,:,:])
    
    
    
    
    
    
    path="/home/pequan2/archambault/Téléchargements/cmems_mod_glo_phy_anfc_merged-uv_PT1H-i_1662475749700.nc"
    a=xr.open_mfdataset(path,engine='netcdf4').sel()
    
    
    
    path="/home/pequan2/archambault/Téléchargements/CORIOLIS-GLOBAL-NRTOA-OBS_TIME_SERIE_1662476016038.nc"
    a=xr.open_mfdataset(path,engine='netcdf4')
    
    x=a.TEMP.values
    plt.figure()
    plt.imshow(x[0,0,:,:])
        
    path="/home/pequan2/archambault/Téléchargements/METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2_1662476344595.nc"
    a=xr.open_mfdataset(path,engine='netcdf4')
    
    x=a.analysed_sst.values
    plt.figure()
    plt.imshow(x[0,:,:])
    
    
    
    path="/home/pequan2/archambault/src/Ocean_challenge/Ocean_Challenge_2021/Data/raw/METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2_1662477276483.nc"
    a=xr.open_mfdataset(path,engine='netcdf4').sel(lat=slice(33,43),lon=slice(-65,-55))
    
    x=a.analysed_sst.values
    plt.figure()
    plt.imshow(x[350,:,:])
    
    
    
    path="/home/pequan2/archambault/src/Ocean_challenge/Ocean_Challenge_2021/Data/raw/METOFFICE-GLO-SST-L4-NRT-OBS-ANOM-V2_1662549563518.nc"
    a=xr.open_mfdataset(path,engine='netcdf4').sel(lat=slice(33,43),lon=slice(-65,-55))
    
    x=a.analysed_sst.values
    plt.figure()
    plt.imshow(x[350,:,:])
    
    
    path="/home/pequan2/archambault/src/Ocean_challenge/Ocean_Challenge_2021/Data/raw/SST/20180117090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc"
    a=xr.open_mfdataset(path,engine='netcdf4').sel(lat=slice(33,43),lon=slice(-65,-55))
    a=xr.open_mfdataset(path,engine='netcdf4')
    
    x=a.analysis_error.values
    plt.figure()
    plt.imshow(x[0,:,:])
    
    
    
    path="/home/pequan2/archambault/src/Ocean_challenge/Ocean_Challenge_2021/Data/raw/SST/SST_PRODUCT_010_024/sst_sat_product_010_024_2004-04.nc"
    a=xr.open_mfdataset(path,engine='netcdf4').sel(latitude=slice(33,43),longitude=slice(-65,-55))
    
    x=a.analysed_sst.values
    plt.figure()
    plt.imshow(x[0,:,:])
    
    path="/home/pequan2/archambault/src/Ocean_challenge/Ocean_Challenge_2021/Data/raw/SST/glorys12v1_mod_product_001_030_2015-10.nc"
    a=xr.open_mfdataset(path,engine='netcdf4').sel(latitude=slice(33,43),longitude=slice(-65,-55))
    
    x=a.sst.values
    plt.figure()
    plt.imshow(x[0,:,:])
    
    
