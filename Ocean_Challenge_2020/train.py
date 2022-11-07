import numpy as np
import matplotlib.pyplot as plt
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
import sys



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
    satellite=params["satellite"]
    
   
    if params["architecture"]==STAE:
        NETWORK=params["architecture"](window,window,256)
    else:
        NETWORK=params["architecture"](window,window)
    NETWORK.to(device)
    optimizer = Adam(NETWORK.parameters(),lr=params["lr"])
    
    scheduler = ExponentialLR(optimizer, gamma=params["gamma"])
    
    
    dataloader = DataLoader(
        Challenge2020Dataset(mode="all",window=window,satellite=satellite),
        batch_size=params["batch_size"],
        shuffle= params[ 'shuffle'],
        num_workers=params[ 'num_workers'],
    )
        
    dataloader_save = DataLoader(
        Challenge2020Dataset(mode="all",window=window,satellite=satellite),
        batch_size=params["batch_size"],
        shuffle= False,
        num_workers=params[ 'num_workers'],
    )
    dataloader_val = DataLoader(
        Challenge2020Dataset(mode="train",window=window,satellite=satellite),
        batch_size=params["batch_size"],
        shuffle= params[ 'shuffle'],
        num_workers=params[ 'num_workers'],
    )

    
    loss =MaskedMSELoss()
    for epoch in range(0, params["nb_epochs"]):
        
        s=0
        
        for i, imgs in enumerate(dataloader):
            optimizer.zero_grad()

            tracks=imgs["tracks"].to(device)
            mask=imgs["mask"].to(device)
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
            plot=False
        
        
        
        if epoch==0:
            SCORE=validation_step2020(dataloader_val ,NETWORK,plot,plot_name=params["path_save"]+"epoch"+str(epoch+1))
            history={"epoch":[0]}
            history["loss"]=[]
            for k in SCORE.keys():
                history[k]=[SCORE[k]]

        else:
            SCORE=validation_step2020(dataloader_val ,NETWORK,plot,plot_name=params["path_save"]+"epoch"+str(epoch+1))
            for k in SCORE.keys():
                history[k].append(SCORE[k])
            history["epoch"].append(epoch)
        plt.close()
        history["loss"].append(s/len(dataloader))
        print("Epoch "+str(epoch+1)+" : NRMSE = %2.2f"%(SCORE["nrmse t"+str(window//2)]*100)+"% \t"+" : RMSE = %2.2f"%(SCORE["rmse t"+str(window//2)]*100),"(cm)\t lr : %1.5f"%(optimizer.param_groups[0]["lr"]*1000) ,"x 1e-3")
    
    
    
    
    save_history(history, params["path_save"]+"history.json")
    
    RESULTS=[]

    for i, imgs in enumerate(dataloader_save):
        optimizer.zero_grad()
        x=imgs["sst"].to(device)
        

        out=NETWORK(x)
        RESULTS.append(out[0,:,:,:].detach().cpu().numpy())
        
    
    ds=dataloader_save.dataset.netcdf_dataset(np.array(RESULTS),True)
    ds.to_netcdf(params["path_save"]+"ds.nc")
    save_history(history, params["path_save"]+"history.json")

if __name__=="__main__":
    from loss_and_metrics import *
    sys.path.insert(0, "../")

    from network_architectures import *
    from utils import *
    from dataset import *
    
    for n in range(10):
        w=3
        name="Save/testgit/W"+str(w)+"/net"+str(n)+"/"
        params = {'batch_size': 1,
                  'shuffle': True,
                  'num_workers': 1,
                  "satellite":"swotnadir",
                  "nb_epochs":4,
                  "window":w,
                  "path_save":name,
                  "architecture":Unet,
                  "lr":5e-4,
                  "gamma":0.97,
                  }
        h=train(params)
    
    
    # EXP WEEK END
    # for n in range(7,10):
    #     w=3   
        
    #     name="Save/Unet_swot_ens_110_120/W"+str(w)+"/net"+str(n)+"/"
    #     params = {'batch_size': 1,
    #               'shuffle': True,
    #               'num_workers': 1,
    #               "epoch":0,
    #               "satellite":"swotnadir",
    #               "nb_epochs":120,
    #               "IN":"sst",
    #               "IN2":None,
    #               "window":w,
    #               "metric_on_window":True,
    #               "path_save":name,
    #               "architecture":UnetMLIA,
    #               "loss":MaskedMSELoss,
    #               "lr":1e-3,
    #               "gamma":0.96,
    #               "saving_epoch":110}
    #     h=train(params)
        
    #     w=5
        
    #     name="Save/Unet_nadir_ens_53_63/W"+str(w)+"/net"+str(n)+"/"
    #     params = {'batch_size': 1,
    #               'shuffle': True,
    #               'num_workers': 1,
    #               "epoch":0,
    #               "satellite":"nadir",
    #               "nb_epochs":63,
    #               "IN":"sst",
    #               "IN2":None,
    #               "window":w,
    #               "metric_on_window":True,
    #               "path_save":name,
    #               "architecture":UnetMLIA,
    #               "loss":MaskedMSELoss,
    #               "lr":1e-3,
    #               "gamma":0.96,
    #               "saving_epoch":53}
    #     h=train(params)
    
    # for n in range(10):
    #     w=7  
        
    #     name="Save/STAE256_swot_ens_110_120/W"+str(w)+"/net"+str(n)+"/"
    #     params = {'batch_size': 1,
    #               'shuffle': True,
    #               'num_workers': 1,
    #               "epoch":0,
    #               "satellite":"swotnadir",
    #               "nb_epochs":110,
    #               "IN":"sst",
    #               "IN2":None,
    #               "window":w,
    #               "metric_on_window":True,
    #               "path_save":name,
    #               "architecture":ST_AutoEncoder_256,
    #               "loss":MaskedMSELoss,
    #               "lr":5e-4,
    #               "gamma":0.97,
    #               "saving_epoch":100}
    #     h=train(params)
        
    #     w=5
        
    #     name="Save/STAE256_nadir_ens_50_60/W"+str(w)+"/net"+str(n)+"/"
    #     params = {'batch_size': 1,
    #               'shuffle': True,
    #               'num_workers': 1,
    #               "epoch":0,
    #               "satellite":"nadir",
    #               "nb_epochs":60,
    #               "IN":"sst",
    #               "IN2":None,
    #               "window":w,
    #               "metric_on_window":True,
    #               "path_save":name,
    #               "architecture":ST_AutoEncoder_256,
    #               "loss":MaskedMSELoss,
    #               "lr":5e-4,
    #               "gamma":0.97,
    #               "saving_epoch":50}
    #     h=train(params)
    
    
    
        
    
    
    
    # n=2
    # w=9
    # name="Save/STAEdeep_nadir/W"+str(w)+"/net"+str(n)+"/"
    # params = {'batch_size': 1,
    #           'shuffle': True,
    #           'num_workers': 1,
    #           "epoch":0,
    #           "satellite":"nadir",
    #           "nb_epochs":100,
    #           "IN":"sst",
    #           "IN2":None,
    #           "window":w,
    #           "metric_on_window":True,
    #           "path_save":name,
    #           "architecture":ST_AutoEncoder_256,
    #           "loss":MaskedMSELoss,
    #           "lr":5e-4,
    #           "gamma":0.97,
    #           "saving_epoch":96}
    # h=train(params)
    # name="Save/STAEdeep_swot/W"+str(w)+"/net"+str(n)+"/"
    # params = {'batch_size': 1,
    #           'shuffle': True,
    #           'num_workers': 1,
    #           "epoch":0,
    #           "satellite":"swotnadir",
    #           "nb_epochs":100,
    #           "IN":"sst",
    #           "IN2":None,
    #           "window":w,
    #           "metric_on_window":True,
    #           "path_save":name,
    #           "architecture":ST_AutoEncoder_256,
    #           "loss":MaskedMSELoss,
    #           "lr":5e-4,
    #           "gamma":0.97,
    #           "saving_epoch":96}
    # h=train(params)
    # for n in range(3):/home/pequan2/archambault/src/Ocean_challenge/Ocean_Challenge_2021/loss_and_metrics.py
    #     print(n)
        
    #     for w in [5,7,9] :
    #         name="Save/STAEdeep_nadir/W"+str(w)+"/net"+str(n)+"/"
    #         params = {'batch_size': 1,
    #                   'shuffle': True,
    #                   'num_workers': 1,
    #                   "epoch":0,
    #                   "satellite":"nadir",
    #                   "nb_epochs":100,
    #                   "IN":"sst",
    #                   "IN2":None,
    #                   "window":w,
    #                   "metric_on_window":True,
    #                   "path_save":name,
    #                   "architecture":ST_AutoEncoder_256,
    #                   "loss":MaskedMSELoss,
    #                   "lr":5e-4,
    #                   "gamma":0.97,
    #                   "saving_epoch":96}
    #         h=train(params)
    #         name="Save/STAEdeep_swot/W"+str(w)+"/net"+str(n)+"/"
    #         params = {'batch_size': 1,
    #                   'shuffle': True,
    #                   'num_workers': 1,
    #                   "epoch":0,
    #                   "satellite":"swotnadir",
    #                   "nb_epochs":100,
    #                   "IN":"sst",
    #                   "IN2":None,
    #                   "window":w,
    #                   "metric_on_window":True,
    #                   "path_save":name,
    #                   "architecture":ST_AutoEncoder_256,
    #                   "loss":MaskedMSELoss,
    #                   "lr":5e-4,
    #                   "gamma":0.97,
    #                   "saving_epoch":96}
    #         h=train(params)
    #     # w=7
        # name="Save/test_stae/W"+str(w)+"/net"+str(n)+"/"
        # params = {'batch_size': 1,
        #           'shuffle': True,
        #           'num_workers': 1,
        #           "epoch":0,
        #           "satellite":"nadir",
        #           "nb_epochs":100,
        #           "IN":"sst",
        #           "IN2":None,
        #           "window":w,
        #           "metric_on_window":True,
        #           "path_save":name,
        #           "architecture":ST_AutoEncoder_256,
        #           "loss":MaskedMSELoss,
        #           "lr":5e-3,
        #           "gamma":0.98,
        #           "saving_epoch":98}
        # h=train(params)
        
    
    
    # for n in range(1):
    #     print(n)
        
        
    #     w=5
    #     name="Save/test_ensemble_55_65/W"+str(w)+"/net"+str(n)+"/"
    #     params = {'batch_size': 1,
    #               'shuffle': True,
    #               'num_workers': 1,
    #               "epoch":0,
    #               "satellite":"nadir",
    #               "nb_epochs":65,
    #               "IN":"sst",
    #               "IN2":None,
    #               "window":w,
    #               "metric_on_window":True,
    #               "path_save":name,
    #               "architecture":UnetMLIA,
    #               "loss":MaskedMSELoss,
    #               "lr":1e-3,
    #               "gamma":0.96,
    #               "saving_epoch":55}
    #     h=train(params)
        # w=5
        # name="Save/SST_nadir_57epoch/W"+str(w)+"/net"+str(n)+"/"
        # params = {'batch_size': 1,
        #           'shuffle': True,
        #           'num_workers': 1,
        #           "epoch":0,
        #           "satellite":"nadir",
        #           "nb_epochs":57,
        #           "IN":"sst",
        #           "IN2":None,
        #           "window":w,
        #           "metric_on_window":True,
        #           "path_save":name,
        #           "architecture":ST_AutoEncoder,
        #           "loss":MaskedMSELoss,
        #           "lr":1e-3,
        #           "gamma":0.96}
        # h=train(params)
        
        # Windows=[3]
        # for w in Windows:
    
        #     name="Save/Experience3/SST_swotnadir_113epoch/W"+str(w)+"/net"+str(n)+"/"
        #     params = {'batch_size': 1,
        #               'shuffle': True,
        #               'num_workers': 1,
        #               "epoch":0,
        #               "satellite":"swotnadir",
        #               "nb_epochs":113,
        #               "IN":"sst",
        #               "IN2":None,
        #               "window":w,
        #               "metric_on_window":True,
        #               "path_save":name,
        #               "architecture":UnetMLIA,
        #               "loss":MaskedMSELoss,
        #               "lr":1e-3,
        #               "gamma":0.96}
        #     h=train(params)
    
        
        # Windows=[3,5,7,9]
        # for w in Windows:
    
        #     name="Save/Experience3/STAE/nadir/W"+str(w)+"/net"+str(n)+"/"
        #     params = {'batch_size': 1,
        #               'shuffle': True,
        #               'num_workers': 1,
        #               "epoch":0,
        #               "satellite":"nadir",
        #               "nb_epochs":100,
        #               "IN":"sst",
        #               "IN2":None,
        #               "window":w,
        #               "metric_on_window":True,
        #               "path_save":name,
        #               "architecture":ST_AutoEncoder,
        #               "loss":MaskedMSELoss,
        #               "lr":1e-3,
        #               "gamma":0.96}
        #     h=train(params)
        #     name="Save/Experience3/STAE/swotnadir/W"+str(w)+"/net"+str(n)+"/"
        #     params = {'batch_size': 1,
        #               'shuffle': True,
        #               'num_workers': 1,
        #               "epoch":0,
        #               "satellite":"swotnadir",
        #               "nb_epochs":150,
        #               "IN":"sst",
        #               "IN2":None,
        #               "window":w,
        #               "metric_on_window":True,
        #               "path_save":name,
        #               "architecture":ST_AutoEncoder,
        #               "loss":MaskedMSELoss,
        #               "lr":1e-3,
        #               "gamma":0.96}
        #     h=train(params)
    
    # for n in range(15):
    #     print(n)
        
        
    #     Windows=[1,3,5,7,11,15,21]
    #     for w in Windows:
    
    #         name="Save/Experience3/SST/W"+str(w)+"/net"+str(n)+"/"
    #         params = {'batch_size': 1,
    #                   'shuffle': True,
    #                   'num_workers': 1,
    #                   "epoch":0,
    #                   "satellite":"nadir",
    #                   "nb_epochs":150,
    #                   "IN":"sst",
    #                   "IN2":None,
    #                   "window":w,
    #                   "metric_on_window":True,
    #                   "path_save":name,
    #                   "architecture":UnetMLIA,
    #                   "loss":MaskedMSELoss,
    #                   "lr":1e-3,
    #                   "gamma":0.96}
    #         h=train(params)
    # n=3       
    # Windows=[15,21]
    # for w in Windows:
    
    #     name="Save/Experience2/SST/W"+str(w)+"/net"+str(n)+"/"
    #     params = {'batch_size': 1,
    #               'shuffle': True,
    #               'num_workers': 1,
    #               "epoch":0,
    #               "satellite":"swotnadir",
    #               "nb_epochs":150,
    #               "IN":"sst",
    #               "IN2":None,
    #               "window":w,
    #               "metric_on_window":True,
    #               "path_save":name,
    #               "architecture":UnetMLIA,
    #               "loss":MaskedMSELoss,
    #               "lr":1e-3,
    #               "gamma":0.96}
    #     h=train(params)
        
        
    # for n in range(4,15):
    #     print(n)
        
        
    #     Windows=[1,3,5,7,11,15,21]
    #     for w in Windows:
    
    #         name="Save/Experience2/SST/W"+str(w)+"/net"+str(n)+"/"
    #         params = {'batch_size': 1,
    #                   'shuffle': True,
    #                   'num_workers': 1,
    #                   "epoch":0,
    #                   "satellite":"swotnadir",
    #                   "nb_epochs":150,
    #                   "IN":"sst",
    #                   "IN2":None,
    #                   "window":w,
    #                   "metric_on_window":True,
    #                   "path_save":name,
    #                   "architecture":UnetMLIA,
    #                   "loss":MaskedMSELoss,
    #                   "lr":1e-3,
    #                   "gamma":0.96}
    #         h=train(params)
    
    
            
            # name="Save/Experience2/OI/W"+str(w)+"/net"+str(n)+"/"
            # params = {'batch_size': 1,
            #           'shuffle': True,
            #           'num_workers': 1,
            #           "epoch":0,
            #           "satellite":"swotnadir",
            #           "nb_epochs":150,
            #           "IN":"duacs",
            #           "IN2":None,
            #           "window":w,
            #           "metric_on_window":True,
            #           "path_save":name,
            #           "architecture":UnetMLIA,
            #           "loss":MaskedMSELoss,
            #           "lr":1e-3,
            #           "gamma":0.96}
            # h=train(params)
          
            # name="Save/Experience2/OI_SST/W"+str(w)+"/net"+str(n)+"/"
            # params = {'batch_size': 1,
            #           'shuffle': True,
            #           'num_workers': 1,
            #           "epoch":0,
            #           "satellite":"swotnadir",
            #           "nb_epochs":150,
            #           "IN":"sst",
            #           "IN2":"duacs",
            #           "window":w,
            #           "metric_on_window":True,
            #           "path_save":name,
            #           "architecture":UnetMLIA,
            #           "loss":MaskedMSELoss,
            #           "lr":1e-3,
            #           "gamma":0.96}
            # h=train(params)
    
    
    # for k in range(10,20):
    
    #_________________________MLIA_________________________
    # for w in [11]:
    #     name="../Save/test/"
    #     params = {'batch_size': 1,
    #               'shuffle': True,
    #               'num_workers': 1,
    #               "epoch":0,
    #               "nb_epochs":150,
    #               "IN":"sst",
    #               "IN2":None,
    #               "window":w,
    #               "metric_on_window":True,
    #               "path_save":name,
    #               "architecture":UnetMLIA,
    #               "loss":MaskedMSELoss,
    #               "lr":1e-3,
    #               "gamma":0.96}
    #     h=train(params)
    
    #_________________________Conv2DP1_________________________
                
    
    #     for w in [8]:
    #         name="../Save/test/"
    #         params = {'batch_size': 1,
    #                   'shuffle': True,
    #                   'num_workers': 1,
    #                   "epoch":0,
    #                   "nb_epochs":150,
    #                   "IN":"sst",
    #                   "IN2":None,
    #                   "window":w,
    #                   "metric_on_window":True,
    #                   "path_save":name,
    #                   "architecture":BourbierNet,
    #                   "loss":MaskedMSELoss,
    #                   "lr":1e-3,
    #                   "gamma":0.96}
    #         h=train(params)
    
    
    
        # for w in [11]:
        #     name="../Save/test/"
        #     params = {'batch_size': 4,
        #               'shuffle': True,
        #               'num_workers': 1,
        #               "epoch":0,
        #               "nb_epochs":150,
        #               "IN":"tracks",
        #               "IN2":None,
        #               "window":w,
        #               "metric_on_window":True,
        #               "path_save":name,
        #               "architecture":UnetMLIA,
        #               "loss":MaskedMSELoss,
        #               "lr":1e-3,
        #               "gamma":0.96}
        #     h=train(params)
    # for k in range(10,20):
    #     for w in [3,5,7,9,11,21]:
    #         name="../Save/window/SSTW"+str(w)+"_"+str(k)+"/"
    #         params = {'batch_size': 1,
    #                   'shuffle': True,
    #                   'num_workers': 1,
    #                   "epoch":0,
    #                   "nb_epochs":150,
    #                   "IN":"sst",
    #                   "IN2":None,
    #                   "window":w,
    #                   "metric_on_window":True,
    #                   "path_save":name,
    #                   "architecture":Unet,
    #                   "loss":MaskedMSELoss,
    #                   "lr":1e-3,
    #                   "gamma":0.96}
    #         h=train(params)
    
    #     params = {'batch_size': 1,
    #               'shuffle': True,
    #               'num_workers': 1,
    #               "epoch":0,
    #               "nb_epochs":150,
    #               "IN":"sst",
    #               "IN2":None,
    #               "window":7,
    #               "metric_on_window":True,
    #               "path_save":"../Save/window/BasePriorOIronanW7_"+str(k)+"/",
    #               "architecture":Unet,
    #               "loss":MaskedMSELoss,
    #               "lr":1e-3,
    #               "gamma":0.96}
    #     train(params)
        
        
    #     params = {'batch_size': 1,
    #               'shuffle': True,
    #               'num_workers': 1,
    #               "epoch":0,
    #               "nb_epochs":150,
    #               "IN":"sst",
    #               "IN2":None,
    #               "window":15,
    #               "metric_on_window":True,
    #               "path_save":"../Save/window/BasePriorOIronanW15_"+str(k)+"/",
    #               "architecture":Unet,
    #               "loss":MaskedMSELoss,
    #               "lr":1e-3,
    #               "gamma":0.96}
    #     train(params)
        
        
        
    #     params = {'batch_size': 1,
    #               'shuffle': True,
    #               'num_workers': 1,
    #               "epoch":0,
    #               "nb_epochs":150,
    #               "IN":"sst",
    #               "IN2":None,
    #               "window":21,
    #               "metric_on_window":True,
    #               "path_save":"../Save/window/BasePriorOIronanW21_"+str(k)+"/",
    #               "architecture":Unet,
    #               "loss":MaskedMSELoss,
    #               "lr":1e-3,
    #               "gamma":0.96}
    #     train(params)
        
        
        
         
    #     params = {'batch_size': 1,
    #               'shuffle': True,
    #               'num_workers': 1,
    #               "epoch":0,
    #               "nb_epochs":150,
    #               "IN":"sst",
    #               "IN2":None,
    #               "window":31,
    #               "metric_on_window":True,
    #               "path_save":"../Save/BasePriorOIronanW31_"+str(k)+"/",
    #               "architecture":Unet,
    #               "loss":MaskedMSELoss,
    #               "lr":1e-3,
    #               "gamma":0.96}
    #     train(params)
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    # x=torch.Tensor([[[[1,2,3],[4,5,6],[7,8,9]],[[11,12,13],[14,15,16],[17,18,19]]]])
    # x
    # x.shape
    
    # mask=torch.Tensor([[[[1,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]]).bool()
    # mask
    
    # x_masked=masked_select(x,mask)
    # x_masked
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # ALL=np.load(    "/home/pequan2/archambault/src/osse/Save/BasePriorSSTronanW3_0/RESULTS.npy")
    # print(ALL.shape)
    
    
    # dataloader = DataLoader(
    #     SshDatasetPriorRonan("train",3),
    #     batch_size=1,
    #     shuffle= False,
    #     num_workers=1,
    # )
    # REF=[]
    # for i, imgs in enumerate(dataloader):
    
    
    #     REF.append(imgs["ref"][0,1,:,:].numpy())
    
    
    # REF=dataloader.dataset.scaler_ref.inverse_transform(torch.Tensor(np.array(REF)))
    # print(REF.shape)
    # meanRSST=dataloader.dataset.scaler_tracks.inverse_transform(torch.Tensor(ALL[:,1,:,:]))
    # print(meanRSST.shape)
    
    # MIN=dataloader.dataset.scaler_ref.data_min
    # print(MIN)
    # MIN=dataloader.dataset.scaler_ref.data_max
    
    # print(MAX)
    
    
    # meanRSST=meanRSST.numpy()
    # REF=REF.numpy()
    
    # rms=dataloader.dataset.rms
    # a,b,c=evaluation(meanRSST, REF,rms)
    
    # plt.figure()
    # diff=REF[30,:,:]-meanRSST[30,:,:]
    # plt.imshow(diff,cmap="seismic")
    # # plt.clim(())
    # plt.colorbar()
            
    # plt.figure()
    
    # plt.imshow(REF[30,:,:],cmap="seismic")
    # plt.figure()
    
    # plt.imshow(meanRSST[30,:,:],cmap="seismic")
