
import matplotlib.pyplot as plt
import numpy as np

def plot_line(Images,Titres,cmap,save_name,label="SSH(m)",shrink=0.3,center_colormap=True):
    #this function plots a line of n images
    # Images : a list of images
    # Titres : The titles of the n images in the same order
    # cmap : the colormap to use (advice : "terrain" for SSH, "seismic" for difference)
    # label : the label of the colorbar
    # shrink :a float that shrinks the cbar
    # center_colorbar : a boolean to center the colobar (for differences for exemple.)
    fig,axes=plt.subplots(nrows=1,ncols=len(Images),figsize=(35,15))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    clim=(1000,-1000)
    for n in range (len(Images)):
        im=axes[0].imshow(Images[n],cmap=cmap)
        clim_new=im.properties()["clim"]
        clim=(min(clim[0],clim_new[0]),max(clim[1],clim_new[1]))
    if center_colormap:
        clim=(-max(abs(clim[0]),abs(clim[1])),max(abs(clim[0]),abs(clim[1])))

    for n in range (len(Images)):
        im=axes[0].imshow(Images[n],cmap=cmap,clim=clim)
    
    for n in range (len(Images)):
  
        axes[n].imshow(Images[n],clim=clim,cmap=cmap)
        axes[n].set_title(Titres[n],fontsize=20)

    col=fig.colorbar(im,ax= axes[:], location ="right",shrink=shrink)
    col.ax.tick_params(labelsize=20)
    col.set_label(label=label,size=20)
    
    plt.savefig(save_name+".pdf",bbox_inches="tight")

def plot_line(Images,Titres,cmap,save_name,label="SSH(m)",shrink=0.3,center_colormap=True,
              fig_width=15):
    #this function plots a line of n images
    # Images : a list of images
    # Titres : The titles of the n images in the same order
    # cmap : the colormap to use (advice : "terrain" for SSH, "seismic" for difference)
    # label : the label of the colorbar
    # shrink :a float that shrinks the cbar
    # center_colorbar : a boolean to center the colobar (for differences for exemple.)
    figsize=(len(Images)*fig_width,fig_width)
    fontsize=figsize[0]*0.6

    fig,axes=plt.subplots(nrows=1,ncols=len(Images),figsize=figsize)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    clim=(1000,-1000)
    for n in range (len(Images)):
        im=axes[n].imshow(Images[n],cmap=cmap)
        clim_new=im.properties()["clim"]
        clim=(min(clim[0],clim_new[0]),max(clim[1],clim_new[1]))
    if center_colormap:
        clim=(-max(abs(clim[0]),abs(clim[1])),max(abs(clim[0]),abs(clim[1])))

    for n in range (len(Images)):
        im=axes[n].imshow(Images[n],cmap=cmap,clim=clim)
    
    for n in range (len(Images)):
  
        axes[n].imshow(Images[n],clim=clim,cmap=cmap)
        axes[n].set_title(Titres[n],fontsize=fontsize)

    col=plt.colorbar(im,ax= axes[:], location ="right",shrink=shrink)
    col.ax.tick_params(labelsize=fontsize)
    col.set_label(label=label,size=fontsize)
    
    plt.savefig(save_name+".pdf",bbox_inches="tight")

def plot_n_lines(Images,Titres,cmap,save_name,nrows=1,label="SSH(m)",shrink=0.3,center_colormap=True,
              fig_width=15,north_flip=True):
    #this function plots a line of n images
    # Images : a list of images
    # Titres : The titles of the n images in the same order
    # cmap : the colormap to use (advice : "terrain" for SSH, "seismic" for difference)
    # label : the label of the colorbar
    # shrink :a float that shrinks the cbar
    # center_colorbar : a boolean to center the colobar (for differences for exemple.)
    
    
    figsize=(len(Images)//nrows*fig_width,fig_width*nrows)
    fontsize=figsize[0]*1.5
    
    if len(Images)%nrows!=0:
        raise ValueError
    if north_flip:
        for i in range(len(Images)) :
            Images[i]=np.flip(Images[i],0)
    fig,axes=plt.subplots(nrows=nrows,ncols=len(Images)//nrows,figsize=figsize)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    clim=(1000,-1000)
    for n in range (len(Images)):
        im=axes.flat[n].imshow(Images[n],cmap=cmap)
        clim_new=im.properties()["clim"]
        clim=(min(clim[0],clim_new[0]),max(clim[1],clim_new[1]))
    if center_colormap:
        clim=(-max(abs(clim[0]),abs(clim[1])),max(abs(clim[0]),abs(clim[1])))

    for n in range (len(Images)):
        im=axes.flat[n].imshow(Images[n],cmap=cmap,clim=clim)
    
    for n in range (len(Images)):
  
        axes.flat[n].imshow(Images[n],clim=clim,cmap=cmap)
        axes.flat[n].set_title(Titres[n],fontsize=fontsize)
    plt.tight_layout(h_pad=1,w_pad=5)

    col=plt.colorbar(im,ax= axes.flat[:], location ="right",shrink=shrink)
    col.ax.tick_params(labelsize=fontsize)
    col.set_label(label=label,size=fontsize)
    # plt.tight_layout(h_pad=50,w_pad=20)

    plt.savefig(save_name+".pdf",bbox_inches="tight")





def plot_im_plusdiff(Images,Titres,cmap,save_name,label="SSH(m)",shrink=0.3,center_colormap=True):
    #this function plots a line of n images
    # Images : a list of images
    # Titres : The titles of the n images in the same order
    # cmap : the colormap to use (advice : "terrain" for SSH, "seismic" for difference)
    # label : the label of the colorbar
    # shrink :a float that shrinks the cbar
    # center_colorbar : a boolean to center the colobar (for differences for exemple.)
    fig,axes=plt.subplots(nrows=1,ncols=len(Images)+1,figsize=(35,15))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    clim=(100,-100)
    for n in range (len(Images)):
        im=axes[0].imshow(Images[n],cmap=cmap)
        clim_new=im.properties()["clim"]
        clim=(min(clim[0],clim_new[0]),max(clim[1],clim_new[1]))
    if center_colormap:
        clim=(-max(abs(clim[0]),abs(clim[1])),max(abs(clim[0]),abs(clim[1])))

    for n in range (len(Images)):
        im=axes[0].imshow(Images[n],cmap=cmap,clim=clim)
        
    for n in range (len(Images)):
  
        axes[n].imshow(Images[n],clim=clim,cmap=cmap)
        axes[n].set_title(Titres[n],fontsize=20)

    col=fig.colorbar(im,ax= axes[0:-1], location ="right",shrink=shrink)
    col.ax.tick_params(labelsize=20)
    col.set_label(label=label,size=20)
    
    clim=(100,-100)

    
    clim=(np.min(Images[0]-Images[1]),np.max(Images[0]-Images[1]))
    clim=(-max(abs(clim[0]),abs(clim[1])),max(abs(clim[0]),abs(clim[1])))


    im=axes[-1].imshow(Images[0]-Images[1],cmap="seismic")

    
    col=fig.colorbar(im,ax= axes[0:-1], location ="right",shrink=shrink)
    col.ax.tick_params(labelsize=20)
    col.set_label(label=label,size=20)
    
    plt.savefig(save_name+".pdf",bbox_inches="tight")
    
   
def plot_history(H,metric,save=None,legend=None, title=None,fontsize_legend=None,fontsize_title=None):
    plt.figure(figsize=(15,10))
    for h in H:
        plt.plot(h[metric])
    plt.legend(legend,fontsize=fontsize_legend)
    plt.title(title,fontsize=fontsize_title)
    
    if save!=None:
        plt.savefig(save+".pdf",bbox_inches="tight")
    plt.yscale("log")


def plot_curve(H,save=None,legend=None, title=None,fontsize_legend=None,fontsize_title=None):
    plt.figure(figsize=(15,10))
    for h in H:
        plt.plot(h)
    if legend!=None:
        plt.legend(legend,fontsize=fontsize_legend)
    plt.title(title,fontsize=fontsize_title)
    
    if save!=None:
        plt.savefig(save+".pdf",bbox_inches="tight")
    plt.yscale("log")
    
    
def plot_error_on_timewindow(y1,y2,x,title,legend,xlabel,ylabel):
    
    # index=index+index
    plt.figure(figsize=(13,10))
    plt.scatter(x, y1,s=100,color="crimson")
    plt.scatter(x, y2,s=100,color="dodgerblue",marker="s")
    plt.legend(legend,fontsize=20)
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
    plt.title(title,fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)





    
    
    