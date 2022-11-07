import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import warnings
from torch import masked_select
import os
from utils import *
import xarray as xr

class Minmax_scaler():
    
    def __init__(self, new_min, new_max):
        
        self.new_min=new_min
        self.new_max=new_max
        
    def fit(self, data):
        # nan=data.mean()
        data=torch.nan_to_num(data,) 

        self.data_min=data.min()
        self.data_max=data.max()
        # print(self.data_min)
        # print(self.data_max)


    def transform(self, X):
        
        X_std = (X-self.data_min) / (self.data_max-self.data_min)
        transformed_X = X_std*(self.new_max-self.new_min) + self.new_min
        
        return transformed_X

    def inverse_transform(self, X):
        
        ratio = (self.data_max-self.data_min)/(self.new_max-self.new_min)
        itransformed_X = (X-self.new_min)*ratio + self.data_min
        
        return itransformed_X

   

class Challenge2020Dataset(Dataset):
    
    def __init__(self,path="Data/processed/",mode="all",satellite="swotnadir",noise=True,window=3,new_min=0.1,new_max=1.6,nan=0.75):
        warnings.filterwarnings('ignore')

        self.mode=mode
        self.window=window
        self.satellite=satellite
        self.path=path
        self.ds=xr.open_mfdataset(self.path+"ds_"+self.satellite+".nc",mode="r")
        
        scaler_sst=Minmax_scaler(new_min,new_max)
        scaler_ref=Minmax_scaler(new_min,new_max)
        scaler_tracks=Minmax_scaler(new_min,new_max)
        scaler_duacs=Minmax_scaler(new_min,new_max)
        
        scaler_ref.fit(torch.Tensor(self.ds.SSH.values))
        scaler_sst.fit(torch.Tensor(self.ds.SST.values))
        if noise:
            scaler_tracks.fit(torch.Tensor(self.ds.OBS.values))
            scaler_duacs.fit(torch.Tensor(self.ds.DUACS.values))
        else:
            scaler_tracks.fit(torch.Tensor(self.ds.OBS_MOD.values))
            scaler_duacs.fit(torch.Tensor(self.ds.DUACS_MOD.values))
        
        self.scaler_sst=scaler_sst
        self.scaler_ref=scaler_ref
        self.scaler_tracks=scaler_tracks
        self.scaler_duacs=scaler_duacs
        
        
        if mode=="train":
            t_min=np.datetime64('2013-01-02')
            t_max=np.datetime64('2013-09-30')
     
        if mode=="all":
            t_min=np.datetime64('2012-10-01')
            t_max=np.datetime64('2013-09-30')

             
        self.ds=self.ds.sel(time=slice(t_min,t_max))

        
        REF=torch.Tensor(self.ds.SSH.values)
        SST=torch.Tensor(self.ds.SST.values)
        if noise:
            TRACKS=torch.Tensor(self.ds.OBS.values)
            DUACS=torch.Tensor(self.ds.DUACS.values)
            a=xr.DataArray.to_masked_array(self.ds.OBS).mask
            o=np.ones(a.shape)
            MASK=torch.tensor(o-a).bool()
        else:
            TRACKS=torch.Tensor(self.ds.OBS_MOD.values)
            DUACS=torch.Tensor(self.ds.DUACS_MOD.values)
            a=self.ds.OBS_MOD.mask
            o=np.ones(a.shape)
            MASK=torch.tensor(o-a).bool()

            
        self.rms = np.sqrt(np.mean(REF.numpy()**2))


       
        self.REF=scaler_ref.transform(REF)
        self.SST=scaler_sst.transform(SST)
        self.DUACS=scaler_duacs.transform(DUACS)        
        self.MASK=MASK
        self.TRACKS=torch.nan_to_num(scaler_tracks.transform(TRACKS),nan=nan) 
        self.len=self.TRACKS.shape[0]-2*(self.window//2)
    def __getitem__(self, index):
       
        
        tracks=self.TRACKS[index:index+self.window,:,:]
        sst=self.SST[index:index+self.window,:,:]
        oi=self.DUACS[index:index+self.window,:,:]
        ref=self.REF[index:index+self.window,:,:]
        mask=self.MASK[index:index+self.window,:,:]

        return {"tracks": tracks, "ref": ref,"sst":sst,"duacs":oi,"mask":mask}

    def __len__(self):
        return self.len
    
    def netcdf_dataset(self,tab,inv_norm=False):
        
        if inv_norm:
            tab=self.scaler_tracks.inverse_transform(torch.Tensor(tab)).numpy()
        t_min=np.datetime64('2012-10-01')+self.window//2
        t_max=np.datetime64('2013-09-30')-self.window//2
        
        ds=self.ds.sel(time=slice(t_min,t_max))
        if len(ds.SSH)!=tab.shape[0]:
            raise IndexError("not the good window size")
        dic_return={}
        for k in range(self.window):
            x=tab[:,k,:,:]
            dic_return['gssh_t'+str(k)]= (('time', 'lat', 'lon'), x)
        ds_return = xr.Dataset(dic_return,
                          coords={'time': ds['time'].values,
                                  'lon': ds['lon'].values, 
                                  'lat': ds['lat'].values, 
                                  })
        return ds_return
 
class Challenge2021Dataset(Dataset):
    
    def __init__(self,path="Data/processed/",filtered=True,window=10,new_min=0.1,new_max=1.6,nan=0,add_mdt=True,sst="sst_sat_nrt",all_nadir=False):
        warnings.filterwarnings('ignore')

        self.window=window
        self.path=path
        self.ds=xr.open_mfdataset(self.path+"ds.nc",mode="r")
        self.sst=sst
        self.ds=self.ds.rename({sst:"sst"})
        self.add_mdt=add_mdt
        scaler_sst=Minmax_scaler(new_min,new_max)
        scaler_tracks=Minmax_scaler(new_min,new_max)
        
        scaler_sst.fit(torch.Tensor(self.ds.sst.values))
        self.scaler_sst=scaler_sst
        SST=torch.Tensor(self.ds.sst.values)
        if add_mdt:
            if filtered:
                if all_nadir:
                    NADIR=np.array([self.ds.nadir_ssh_filtered.values,self.ds.j2g_ssh_filtered.values])
                    NADIR=torch.Tensor(np.nanmean(NADIR,axis=0)) 
                else:
                    NADIR=torch.Tensor(self.ds.nadir_ssh_filtered.values)
                    
                C2=torch.Tensor(self.ds.c2_ssh_filtered.values)
                J2G=torch.Tensor(self.ds.j2g_ssh_filtered.values)
            else :
                if all_nadir:
                    NADIR=np.array([self.ds.nadir_ssh_unfiltered.values,self.ds.j2g_ssh_unfiltered.values])
                    NADIR=torch.Tensor(np.nanmean(NADIR,axis=0))
                else:
                    NADIR=torch.Tensor(self.ds.nadir_ssh_unfiltered.values)
                C2=torch.Tensor(self.ds.c2_ssh_unfiltered.values)
                J2G=torch.Tensor(self.ds.j2g_ssh_unfiltered.values)                             
        else:
            if filtered:
                if all_nadir:
                    NADIR=np.array([self.ds.nadir_sla_filtered.values,self.ds.j2g_sla_filtered.values])
                    NADIR=torch.Tensor(np.nanmean(NADIR,axis=0)) 
                else:
                    NADIR=torch.Tensor(self.ds.nadir_sla_filtered.values)
                C2=torch.Tensor(self.ds.c2_sla_filtered.values)
                J2G=torch.Tensor(self.ds.j2g_sla_filtered.values)
            else :
                if all_nadir:
                    NADIR=np.array([self.ds.nadir_sla_unfiltered.values,self.ds.j2g_sla_unfiltered.values])
                    NADIR=torch.Tensor(np.nanmean(NADIR,axis=0))
                else:
                    NADIR=torch.Tensor(self.ds.nadir_sla_unfiltered.values)
                C2=torch.Tensor(self.ds.c2_sla_unfiltered.values)
                J2G=torch.Tensor(self.ds.j2g_sla_unfiltered.values)
                

        scaler_tracks.fit(NADIR)
        self.scaler_tracks=scaler_tracks
      
        mask=np.isnan(NADIR.numpy())
        o=np.ones(mask.shape)
        self.MASK_NADIR=torch.tensor(o-mask).bool()
        
        mask=xr.DataArray.to_masked_array(self.ds.c2_ssh_filtered).mask
        o=np.ones(mask.shape)
        self.MASK_C2=torch.tensor(o-mask).bool()
        
        mask=xr.DataArray.to_masked_array(self.ds.j2g_ssh_filtered).mask
        o=np.ones(mask.shape)
        self.MASK_J2G=torch.tensor(o-mask).bool()
 

        self.mdt=torch.nan_to_num(self.scaler_tracks.transform(torch.Tensor(self.ds.mdt.values)),nan=nan)

        self.rms = np.sqrt(np.nanmean(NADIR.numpy()**2))

        self.SST=self.scaler_sst.transform(SST)
        self.NADIR=torch.nan_to_num(self.scaler_tracks.transform(NADIR),nan=nan)
        self.C2=torch.nan_to_num(self.scaler_tracks.transform(C2),nan=nan)
        self.J2G=torch.nan_to_num(self.scaler_tracks.transform(J2G),nan=nan)
        
       
      
        self.len=self.NADIR.shape[0]-(self.window//2)*2

    def __getitem__(self, index):
       
        
        nadir=self.NADIR[index:index+self.window,:,:]
        mask_nadir=self.MASK_NADIR[index:index+self.window,:,:]
        
        c2=self.C2[index:index+self.window,:,:]
        mask_c2=self.MASK_C2[index:index+self.window,:,:]
        
        j2g=self.J2G[index:index+self.window,:,:]
        mask_j2g=self.MASK_J2G[index:index+self.window,:,:]
        
        sst=self.SST[index:index+self.window,:,:]

        return {"nadir":nadir,"mask_nadir":mask_nadir,
                "c2":c2,"mask_c2":mask_c2,
                "j2g":j2g,"mask_j2g":mask_j2g,
                "sst":sst,
                "mdt":self.mdt}
       
    def __len__(self):
        return self.len
    
