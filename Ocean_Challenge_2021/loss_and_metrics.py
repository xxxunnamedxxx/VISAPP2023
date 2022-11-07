

import numpy as np
import torch
import torch.nn as nn
from torch import masked_select
from utils import *


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))
    
class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, estimate, target, mask):
        masked_estimate=masked_select(estimate,mask)
        masked_target=masked_select(target,mask)
        return self.mse(masked_estimate,masked_target)

def normalized_rmse(estimate, reference,rms=None,gridded=True,mask_ref=None):
    if not(gridded):
        estimate=masked_select(estimate,mask_ref)
        reference=masked_select(reference,mask_ref)
    if type(estimate)==torch.Tensor:
        estimate = estimate.cpu().detach().numpy()
    if type(reference)==torch.Tensor:
        reference = reference.cpu().detach().numpy()
    
    rmse = np.sqrt(np.mean(((estimate-reference)**2)))

    if rms==None:
        rms = np.sqrt(np.mean(((reference)**2)))
    

    return float(1-rmse/rms),float(rmse)




def validation_step2021(dataloader,model,plot=False,plot_name="test"):

    rms=dataloader.dataset.rms
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")


    for i, imgs in enumerate(dataloader):
        
        x=imgs["sst"].to(device)
            
        out=model(x)
        out=dataloader.dataset.scaler_tracks.inverse_transform(out)

        if i==0:
            SCORE={}
            for k in range(out.shape[1]):
                SCORE["train nrmse t"+str(k)]=[]
                SCORE["train rmse t"+str(k)]=[]
                SCORE["validation nrmse t"+str(k)]=[]
                SCORE["validation rmse t"+str(k)]=[]

        #TRAIN
        sat=imgs["nadir"].to(device)
        sat_mask=imgs["mask_nadir"].to(device).bool()
        sat = dataloader.dataset.scaler_tracks.inverse_transform(sat)
       
        for k in range(out.shape[1]):
            n,r=normalized_rmse(out[0,k,:,:], sat[0,k,:,:],rms,False,sat_mask[0,k,:,:])
            
            SCORE["train rmse t"+str(k)].append(r)
            SCORE["train nrmse t"+str(k)].append(n)
        
        #VALIDATION
        sat=imgs["j2g"].to(device)
        sat_mask=imgs["mask_j2g"].to(device).bool()
        sat = dataloader.dataset.scaler_tracks.inverse_transform(sat)
       
        for k in range(out.shape[1]):
            n,r=normalized_rmse(out[0,k,:,:], sat[0,k,:,:],rms,False,sat_mask[0,k,:,:])
            
            SCORE["validation rmse t"+str(k)].append(r)
            SCORE["validation nrmse t"+str(k)].append(n)
        
        
    if plot:
        inter=out[0,dataloader.dataset.window//2,:,:].detach().cpu().numpy()
        refer=sat[0,dataloader.dataset.window//2,:,:].detach().cpu().numpy()
        plot_line([inter,refer],["Inter", "ref"],"terrain",plot_name,label="SSH",shrink=0.3,center_colormap=False)
   
    for k in SCORE.keys():
        SCORE[k]=np.nanmean(SCORE[k])
    return SCORE
   
    