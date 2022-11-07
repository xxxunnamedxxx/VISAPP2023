import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from dataset import *
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
def isNaN(num):
    return num!= num
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


def validation_step2020(dataloader,model,plot=False,plot_name="test",):
    l=0
    num=0
    loss=RMSELoss()
    rms=dataloader.dataset.rms
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    SCORE={}
    for k in range(dataloader.dataset.window):
        SCORE["nrmse t"+str(k)]=[]
        SCORE["rmse t"+str(k)]=[]

    for i, imgs in enumerate(dataloader):

        tracks=imgs["tracks"].to(device)
        ref=imgs["ref"].to(device)
        x=imgs["sst"].to(device)
        
        out=model(x)
    

  
        L=[]
        for k in range(out.shape[1]):
            L.append(dataloader.dataset.scaler_tracks.inverse_transform(out[0,k,:,:]).unsqueeze(0).unsqueeze(0))
        out=torch.cat(L,1)

        ref = dataloader.dataset.scaler_ref.inverse_transform(ref)
    
        for k in range(out.shape[1]):
            n,r=normalized_rmse(out[0,k,:,:], ref[0,k,:,:],rms)
            SCORE["rmse t"+str(k)].append(r)
            SCORE["nrmse t"+str(k)].append(n)

        
    if plot:
        inter=out[0,dataloader.dataset.window//2,:,:].detach().cpu().numpy()
        refer=ref[0,dataloader.dataset.window//2,:,:].detach().cpu().numpy()
        plot_line([inter,refer],["Inter", "ref"],"terrain",plot_name,label="SSH",shrink=0.3,center_colormap=False)
    
    for k in SCORE.keys():
        SCORE[k]=np.mean(SCORE[k])
    return SCORE


