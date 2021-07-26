#The scirpt implements the same nn architecture as stated by microsoft in their volume estimation paper.
#The features is same as in their paper , but we calculated the features earlier on our simulated dataset and then saved it on an .hdf5 file.
#A special data_loader is created for this script which serves the data to the architecture as required.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from asteroid.filterbanks.enc_dec import Filterbank, Encoder, Decoder
from asteroid.filterbanks import STFTFB
from asteroid.losses import multisrc_mse
from asteroid.losses import PITLossWrapper
import numpy as np
import data_loader_microsoft_feat as dl
import matplotlib.pyplot as plt
import seaborn as sns
import gammatone.gtgram as gt
from scipy.signal import butter, lfilter
import torch.fft 


#Calculate the mse between the estimated volume and the real volume with a batch_size 128.

def pred(mean_values, target_value):
    param_mse_loss = 0
    batch_size=128
    for i in range(batch_size):
        param_mse_loss += (mean_values[i] - target_value[i]) ** 2

    return param_mse_loss / batch_size

'''
def NLLloss(y, mean, var):
    """ Negative log-likelihood loss function. """
    return (torch.log(var) + ((y - mean).pow(2)) / var).sum()
'''
dvs="cuda"
'''
def lp_filter(ch1_,fs,cutoff,order):
    nyq=0.5*fs
    normal_cutoff=cutoff/nyq
    b,a=butter(order,normal_cutoff,btype="low",analog=False)

    y=lfilter(b,a,ch1_)
    #print("low-pass filter",y.shape)
    return y[:1499]
'''

def cal_features(ch1, ch2):
    
    ch1_=ch1.view(128,-1)
    #print(ch1_)
    feat_gt=torch.zeros((1,23,1499)).to(device=dvs).float()
        
    for i in range(128):
        #print(i)
        #print(ch1_.shape)
        feat_gt_=gt.gtgram(ch1_[i,:].detach().cpu().clone().numpy(),16000,0.004,0.002,58,20)[:20,:]
        #print(feat_gt_.shape) 
        feat_gt_=torch.tensor(feat_gt_).to(device=dvs).float()
        feat_gt_=torch.log(torch.abs(feat_gt_))
        
        
        #feat_gt=torch.concatenate((feat_gt,feat_gt_.reshape(1,20,-1)),axis=0)
        feat_dft=torch.abs(torch.fft.fft(ch1_[i,:],n=48000)[:1499])
        sort_feat_dft,_=torch.sort(feat_dft)
        sort_feat_dft=sort_feat_dft.to(device=dvs).float()
        feat_dft=feat_dft.to(device=dvs).float()

        lp=torch.tensor(lp_filter(ch1_[i,:].detach().cpu().clone().numpy(),16000,500,100)).to(device=dvs).float()

        tot_=torch.cat((feat_gt_,feat_dft.reshape(1,1499),sort_feat_dft.reshape(1,1499),lp.reshape(1,1499)),axis=0)
        
        feat_gt=torch.cat((feat_gt,tot_.reshape(1,23,1499)),axis=0)
        
    return feat_gt[1:,:,:]
        



#Standard deviation of volume and surface in the training set,used for scaling purposes.

std_volume = 106.02265764052878
std_surface = 84.22407627419787


#Train the model.

def train(model, train_loader, optimizer, epoch, ar_loss, batch_loss_ar, mse_loss):
    model.train()
    print("training....Epoch", epoch)

    loss_batch = 0

    tr_loss = 0

    volume_mse = 0


    for batch_idx, sample_batched in enumerate(train_loader):

        data, surface, volume, absorption, rt60 = sample_batched['bnsample'].float(), sample_batched['surface'].float().to(device=dvs), sample_batched['volume'].float().to(device=dvs), sample_batched['absorption'].float().to(device=dvs), sample_batched['rt60'].float().to(device=dvs)

        

        volume = volume.reshape(128, 1)
        
        print(batch_idx)
        

        optimizer.zero_grad()

        x_1 = data.to(device=dvs)
        
        print(x_1.shape)

        output = model(x_1)

        target_=volume.reshape(128,1)

        loss=mse_loss(output,target_)


        loss.backward()

        optimizer.step()


        # Keep adding mse at every iteration of a batch 

        volume_mse += float(pred(output[:, 0] , volume.view(128)).item())

        

        #keep adding loss at every iteration of a batch while doing a sgd.

        loss_batch = float(loss.item()) + loss_batch

        
        tr_loss = float(loss.item()) + tr_loss

    
        del loss,data

        #After every 100 batches,track batch mse loss  .

        if batch_idx % 100 == 99:
            # print("Running loss after 100 batches",(loss_batch/100),loss_batch)
            
            batch_loss_ar.append((volume_mse)/100)
            
            volume_mse = 0


    print("Epoch Loss", (tr_loss / batch_idx), epoch)
    ar_loss.append(tr_loss / batch_idx)
    return ar_loss, batch_loss_ar


def val(model, train_loader, optimizer, epoch, val_data_ar, acc_data_ar,mse_loss):
    model.eval()
    
    volume_loss_mse = 0
    
    val_loss = 0
    
    local_data_sp = np.zeros([1, 4])  # 42 #44

    for batch_idx, sample_batched in enumerate(train_loader):

        data, surface, volume, absorption, rt60, rm, vp = sample_batched['bnsample'].float(), sample_batched['surface'].float().to(device=dvs), sample_batched['volume'].float().to(device=dvs), sample_batched['absorption'].float().to(device=dvs), sample_batched['rt60'].float().to(device=dvs), sample_batched['room'].to(device=dvs), sample_batched['vp'].to(device=dvs)

        

        volume_loss_mse_t = 0
       
        x_1=data.to(device=dvs)
        
        output = model(x_1)

    

        
        target_1=volume.reshape(128,1) 

        val_loss_t=mse_loss(output,target_1)

        val_loss = float(val_loss_t.item()) + val_loss

        

        volume_loss_mse_t += pred(output[:, 0] , volume.view(128))

        
        target_ = torch.cat((volume.reshape(128,1), rm.reshape(128, 1), vp.reshape(128, 1)),axis=1)  # absorption , rt60


        #Concatenate estimated and real volume +  save data in the numpy array local_data_sp  for analysis purpose.

        save_data = np.concatenate((output.detach().cpu().clone().numpy().reshape(128, 1),
                              target_.detach().cpu().clone().numpy().reshape(128, 3)), axis=1)

        local_data_sp = np.concatenate((local_data_sp, save_data), axis=0)

        #Add MSE loss

        volume_loss_mse = volume_loss_mse + float(volume_loss_mse_t.item())

        #surface_loss_mse = surface_loss_mse + float(surface_loss_mse_t.item())


        del val_loss_t,volume_loss_mse_t
        
        #Track after every 50 batches.

        if batch_idx % 50 == 49:
            
            #surface_acc = (surface_loss_mse / 50)

            volume_acc = (volume_loss_mse / 50)

            
            acc_data_ar.append((volume_acc)) #volume_acc


            volume_loss_mse = 0

            del volume_acc #surface_acc


    val_data_ar.append((val_loss / batch_idx))

    return val_data_ar, acc_data_ar, local_data_sp
    print(val_data_ar, acc_data_ar)


class Model_1(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.batch_size = 128

        self.conv1=nn.Conv2d(1,30,(1,10),stride=(1,1))
        self.avg1=nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
        self.conv2=nn.Conv2d(30,20,(1,10),stride=(1,1))
        self.avg2=nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
        self.conv3=nn.Conv2d(20,10,(1,10),stride=(1,1))
        self.avg3=nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
        self.conv4=nn.Conv2d(10,10,(1,10),stride=(1,1))
        self.avg4=nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
        self.conv5=nn.Conv2d(10,5,(3,9),stride=(1,1))
        self.avg5=nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
        self.conv6=nn.Conv2d(5,5,(3,9),stride=(1,1))
        self.avg6=nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))

    
        

        self.drp = nn.Dropout(p=0.5)



        self.fc = nn.Linear(750, 1)

    def forward(self, x):
        
        x=torch.unsqueeze(x,1)
        #print(x.shape)

        out=self.avg5(self.conv5(self.avg4(self.conv4(self.avg3(self.conv3(self.avg2(self.conv2(self.avg1(self.conv1(x))))))))))
        
        #print(out.shape)
        
        out=self.avg6(self.conv6(out))

        out=self.drp(out)

        out=out.reshape(128,-1)

        out=self.fc(out)
        
        #print(out.shape)

        return out





train_data = dl.binuaral_dataset('train_random_ar_2.npy',
                                 '/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_lwh_2.hdf5',
                                 '/home/psrivastava/baseline/sofamyroom_2/conf_room_setup.yml')
val_data = dl.binuaral_dataset('val_random_ar_2.npy',
                               '/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_lwh_2.hdf5',
                               '/home/psrivastava/baseline/sofamyroom_2/conf_room_setup.yml')

train_dl = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
val_dl = DataLoader(val_data, batch_size=128, shuffle=True, num_workers=0, drop_last=True)


net = Model_1().to(torch.device(dvs))

mse_loss=nn.MSELoss()


optimizer = optim.Adam(net.parameters(), lr=0.0001)



ar_loss = []
batch_loss_ar = []
total_batch_idx = 0
val_data_ar = []
acc_data_ar = []
save_best_val = 0
adcc = np.zeros((1, 1))

local_dt_sp = np.zeros((1, 4))

#Directory path to save , you can modify it according to the needs.

path="/home/psrivastava/baseline/scripts/pre_processing/results_microsoft_vol_exp/"


for epoch in range(100):
    ar_loss, batch_loss_ar= train(net, train_dl, optimizer, epoch, ar_loss, batch_loss_ar,mse_loss)
    val_data_ar, acc_data_ar, local_dt_sp = val(net, val_dl, optimizer, epoch, val_data_ar, acc_data_ar,mse_loss)

    np.save(path+"microsoft_volume_ar_loss.npy", ar_loss)
    np.save(path+"microsoft_volume_batch_loss_ar.npy", batch_loss_ar)
    np.save(path+"microsoft_volume_val_data_ar.npy", val_data_ar)
    np.save(path+"microsoft_volume_acc_data_ar.npy", acc_data_ar)

    # save best model
    if epoch == 0:
        save_best_val = val_data_ar[-1]
        #np.save(path+"microsoft_volume_dummy_input_mean_sh.npy", adcc)
    elif save_best_val > val_data_ar[-1]:
        torch.save({'model_dict': net.state_dict(),'optimizer_dic': optimizer.state_dict(), 'epoch': epoch, 'loss': val_data_ar[-1]},
            path+"microsoft_volume_tas_save_best_sh.pt")
        save_best_val = val_data_ar[-1]
        np.save(path+"microsoft_volume_bnf_mag_96ms_" + str(epoch) + ".npy", local_dt_sp)

