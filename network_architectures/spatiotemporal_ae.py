import math

import torch.nn as nn
from torch.nn.modules.utils import _triple
import torch
import torchvision.transforms as T

class STAE(nn.Module):
    def __init__(self, in_channels,out_channels,resize=256):
        super(STAE, self).__init__()
        
        
        self.out_channels=out_channels
        
        self.resize_in= T.Resize((resize,resize))
        self.resize_out= T.Resize((200,200))

          
        self.encoder = nn.Sequential(
            unsqz(),
           

            SpatioTemporalConv(1,16,3),
            nn.MaxPool3d((1,2,2)),
            
            SpatioTemporalConv(16,32,3),
            nn.MaxPool3d((1,2,2)),

            SpatioTemporalConv(32,64,3),
            nn.MaxPool3d((1,2,2)),
            SpatioTemporalConv(64,128,3),
            nn.MaxPool3d((1,2,2)),

             

        )

        self.decoder = nn.Sequential(
       
            nn.Upsample(scale_factor = (1,2,2), mode='trilinear', align_corners=False),
            SpatioTemporalConv(128,64,3),

            nn.Upsample(scale_factor = (1,2,2), mode='trilinear', align_corners=False),
            SpatioTemporalConv(64,32,3),

            nn.Upsample(scale_factor = (1,2,2), mode='trilinear', align_corners=False),
            SpatioTemporalConv(32,16,3),
            
            nn.Upsample(scale_factor = (1,2,2), mode='trilinear', align_corners=False),
            SpatioTemporalConv_nobnr(16,1,3),

            sqz(),
           
        )

    def forward(self, x):
         x=self.resize_in(x)
         z=self.encoder(x)
         x=self.decoder(z)
         x=self.resize_out(x)
         return x


#### utils
class unsqz(nn.Module):
    def forward(self, input,dim=0):
        return input.unsqueeze(dim)
    
class pshape(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input
    
class sqz(nn.Module):
    def forward(self, input):
        return input.squeeze(0)

##### Spatio Temporal Conv R(2+1)D
## https://github.com/irhum/R2Plus1D-PyTorch/blob/master/module.py
## https://arxiv.org/abs/1711.11248

class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input 
    planes with distinct spatial and time axes, by performing a 2D convolution over the 
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time 
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x
    


class SpatioTemporalConv_nobnr(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        super(SpatioTemporalConv_nobnr, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        return x
    
   
    
    