#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:25:11 2022

@author: archambault
"""

#preprocess data
import requests as rq
import numpy as np
import matplotlib.pyplot as plt
import sys
# sys.path=[]
sys.path.append("../")
from dataset import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import masked_select
import os 

from torch.autograd import Variable
from torch.optim import Adam, SGD

from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import _LRScheduler
import json

def save_history(history, path):
    dic=json.dumps(history)
    f = open(path,"w")
    f.write(dic)
    f.close()
    

def load_history(path):
    
    with open(path, "r") as read_file:
        emps = json.load(read_file)
    return emps

    
def train(param):
    
    

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if not os.path.exists(params["path_save"]):
        os.makedirs(params["path_save"])
    
    window=params["window"]
    
   
    in_feature=window
    
    
   
    
    NETWORK=params["architecture"](in_feature,window)
    NETWORK.to(device)
    optimizer = Adam(NETWORK.parameters(),lr=params["lr"])
    
    scheduler = ExponentialLR(optimizer, gamma=params["gamma"])
    
    
    dataloader = DataLoader(
        Challenge2021Dataset(filtered=params["filtered"],window=window,add_mdt=params["add_mdt"],sst=params["sst"],all_nadir=params["all_nadir"]),
        batch_size=params["batch_size"],
        shuffle= params[ 'shuffle'],
        num_workers=params[ 'num_workers'],
    )
    dataloader2 = DataLoader(
        Challenge2021Dataset(filtered=params["filtered"],window=window,add_mdt=params["add_mdt"],sst=params["sst"],all_nadir=params["all_nadir"]),
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params[ 'num_workers'],
    )
    
    loss = MaskedMSELoss()
    

     
    for epoch in range(0, params["nb_epochs"]):
        
        s=0
        
        for i, imgs in enumerate(dataloader):
            optimizer.zero_grad()

            tracks=imgs["nadir"].to(device)
            mask=imgs["mask_nadir"].to(device)
            x=imgs["sst"].to(device)
            
            out=NETWORK(x)
 
            l=loss(out,tracks,mask)

            s=l.item()+s
            l.backward()
            optimizer.step()
        
        plot=False
        if epoch+1>=0:
            scheduler.step()
        if (epoch+1)%10==0:
            plot=True
        if epoch==0:
            
            SCORE=validation_step2021(dataloader ,NETWORK,plot,plot_name=params["path_save"]+"epoch"+str(epoch+1))
            history={"epoch":[0]}
            history["loss"]=[]
            for k in SCORE.keys():
                history[k]=[SCORE[k]]
         
        else:            
            SCORE=validation_step2021(dataloader ,NETWORK,plot,plot_name=params["path_save"]+"epoch"+str(epoch+1))
            for k in SCORE.keys():
                history[k].append(SCORE[k])
            history["epoch"].append(epoch)
        
        plt.close()
       
        history["loss"].append(s/len(dataloader))
        
        
        nrmse_train="%2.2f"%(SCORE["train nrmse t"+str(window//2)]*100)
        nrmse_validation="%2.2f"%(SCORE["validation nrmse t"+str(window//2)]*100)
        rmse_train="%2.2f"%(SCORE["train rmse t"+str(window//2)]*100)
        rmse_validation="%2.2f"%(SCORE["validation rmse t"+str(window//2)]*100)

        print("Epoch "+str(epoch+1)+" : ",
              "NRMSE = "+nrmse_train+"/"+nrmse_validation+"/"+" % ",
              "RMSE = "+rmse_train+"/"+rmse_validation+"/"+" (cm)")

        
        
        

    RESULTS=[]
    for l in range(window//2):
        RESULTS.append(np.zeros((200,200)))
    for i, imgs in enumerate(dataloader2):
        optimizer.zero_grad()
        x=imgs[IN].to(device)
        
        if IN2!=None:
            x2=imgs[IN2].to(device)
            x=torch.cat((x,x2),1)
        out=NETWORK(x)
        if params["add_mdt"]:
            RESULTS.append(out[0,window//2,:,:].detach().cpu().numpy())
        else: 
            RESULTS.append(out[0,window//2,:,:].detach().cpu().numpy())+imgs["mdt"].squeeze(0)

    for l in range(window//2):
        RESULTS.append(np.zeros((200,200)))
    RESULTS=np.array(RESULTS)

        
    RESULTS=dataloader2.dataset.scaler_tracks.inverse_transform(torch.Tensor(RESULTS)).cpu().numpy()
        
    ds_ours = xr.Dataset({'ssh': (("time",'lat', 'lon'), RESULTS)},
                          coords={'time': dataloader2.dataset.ds.time ,
                                  'lon': dataloader2.dataset.ds.lon ,
                                  'lat': dataloader2.dataset.ds.lat,
                                  })
    ds_ours["time"]=ds_ours["time"]+np.timedelta64(12, 'h')
        


                       
    
    ds_ours.to_netcdf(params["path_save"]+"ds.nc")
    validation_step2021(dataloader ,NETWORK,IN,IN2,True,
                    plot_name=params["path_save"]+"final")
    save_history(history, params["path_save"]+"history.json")


if __name__=="__main__":
    from evaluation_utils import *
    from utils import *
    from loss_and_metrics import *
    from network_architectures import *
    w=7
    n=0
    name="Save/test_git"+"/"
    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 1,
              "epoch":0,
              "nb_epochs":52,
              "filtered":False,
              "window":w,
              "metric_on_window":True,
              "path_save":name,
              "architecture":Unet,
              "lr":1e-3,
              "gamma":0.96,
              "add_mdt":True,
              "sst":"sst_sat_mur",
              "all_nadir":True}
    h=train(params)