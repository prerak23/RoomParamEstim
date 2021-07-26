import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset,DataLoader
from asteroid.filterbanks.enc_dec import Filterbank, Encoder, Decoder
from asteroid.filterbanks import STFTFB
from asteroid.losses import multisrc_mse
from asteroid.losses import PITLossWrapper
import numpy as np
import data_loader_volume as dl 
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.metrics import mean_squared_error
#import torch.onnx
#from torch.utils.tensorboard import SummaryWriter



sns.set_theme()
class Paramfilter(Filterbank):
    def __init__(self,n_filt,ks,stri):
        super(Paramfilter,self).__init__(n_filt,ks,stri)
        self._filters = torch.nn.Parameter(STFTFB(n_filters=n_filt,kernel_size=ks,stride=stri).filters())
    def filters(self):
        return self._filters



def pred(target_value,out_value):
    v_pos_t=0
    v_pos_mse=0
    for i in range(128):
        #l_pos+=np.abs(target_value[i,0]-out_value[i,0])/target_value[i,0]
        #w_pos+=np.abs(target_value[i,1]-out_value[i,1])/target_value[i,1]
        #h_pos+=np.abs(target_value[i,2]-out_value[i,2])/target_value[i,2]
        v_pos_t+=torch.abs(target_value[i,0]-out_value[i,0])/target_value[i,0]
        v_pos_mse+=(target_value[i,0]-out_value[i,0])**2
        '''
        if target_value[i,0] == out_value [i,0]:
            l_pos+=1
        if target_value[i,1] == out_value[i,1]:
            w_pos+=1
        if target_value[i,2] == out_value[i,2]:
            h_pos+=1
        '''
    v_pos_mse_t=v_pos_mse/128
    v_pos_k=(v_pos_t/128)*100
    return v_pos_k,v_pos_mse_t



def train(model, train_loader, optimizer, epoch,ar_loss,batch_loss_ar,loss_func):
    
    model.train()
    
    print("training....Epoch",epoch) 
    
    loss_batch=0
    
    tr_loss=0
    
    for batch_idx,  sample_batched in enumerate(train_loader):
        
        data ,volume = sample_batched['bnsample'].float() ,sample_batched['volume'].float().to(device='cuda')
        
        optimizer.zero_grad()

        output = model(data[:,0,:].to(device='cuda'),data[:,1,:].to(device='cuda'))
        
        #print(target)
        
        #print(target.shape)
        
        target=volume
        
        #print(target)
        
        #print(target.shape)
        
        loss = loss_func(output, torch.log10(target.view(128,1)))
        
        loss.backward()

        optimizer.step()
        
        #print(loss,loss.item())
        
        loss_batch=float(loss.item())+loss_batch
        
        tr_loss=float(loss.item())+tr_loss
        
        #loss.backward()
        
        #optimizer.step()
        
        del loss, target,data

        if batch_idx % 100 == 99:
            
            print("Running loss after 100 batches",(loss_batch/100),loss_batch)
            
            batch_loss_ar.append(loss_batch/100)
            
            loss_batch=0
        #if batch_idx == 102:
        #    break
    print("Epoch Loss", (tr_loss/batch_idx),epoch)
    ar_loss.append(tr_loss/batch_idx)
    return ar_loss,batch_loss_ar


def val(model, train_loader, optimizer, epoch, val_data_ar, acc_data_ar, loss_func):
    
    model.eval()
    
    #val_acc=0
    
    val_loss=0 #Calculate Validation Loss Per Epoch
    
    v_pos=0 #Calculate Validation Accuracy Per Batch
    
    v_pos_mse=0 #Calculate Validation Accuracy Per Batch MSE 
    
    local_data_sp=np.zeros((1,2))
    

    for batch_idx,  sample_batched in enumerate(train_loader):
        
        data, volume = sample_batched['bnsample'].float(), sample_batched['volume'].float().to(device='cuda')
        
        target=volume
        
        output = model(data[:,0,:].to(device='cuda'),data[:,1,:].to(device='cuda'))


        
        val_loss_t = loss_func(output, torch.log10(target.view(128,1)))
        
        val_loss = float(val_loss_t.item())+val_loss
        
        v_pos_ten,v_pos_mse_ten=pred(target.view(128,1),output)
        
        v_pos+=float(v_pos_ten.item())
        v_pos_mse+=float(v_pos_mse_ten.item())
        

        ass=np.concatenate((output.detach().cpu().clone().numpy().reshape(128,1),np.log10(target.detach().cpu().clone().numpy().reshape(128,1))),axis=1)
    

        local_data_sp=np.concatenate((local_data_sp,ass),axis=0)

    
        #l_pos,w_pos,h_pos=pred(target,output,l_pos,w_pos,h_pos)
        
        del val_loss_t,v_pos_ten,v_pos_mse_ten,ass
        
        if batch_idx % 100 == 99:
            
            #print("validation loss after 100 batches",val_data_ar[-1])
            
            print("Volume",v_pos/100)
           
            #acc_data_ar.append((l_acc,w_acc,h_acc))
            acc_data_ar.append([(v_pos/100),(v_pos_mse/100)])
            
            #val_loss=0

            v_pos=0
            v_pos_mse=0

        #if batch_idx == 102:
            #break
    
    val_data_ar.append((val_loss/batch_idx))
    
    '''
    plt.clf()
    plt.scatter(local_data_sp[:,1],local_data_sp[:,0],color='green',alpha=0.5)
    #plt.scatter(np.arange(local_data_sp.shape[0]),local_data_sp[:,1],color='g',alpha=0.5,label="Target")
    plt.xlabel("Target")
    plt.ylabel("Estimated log of volume")
    #plt.legend()
    plt.title("Scatter Plot")

    plt.savefig("sp_val_"+str(epoch)+".png")
    
    plt.clf()
    plt.scatter(10**(local_data_sp[:,1]),10**(local_data_sp[:,0]),color='orange',alpha=0.5)
    #plt.scatter(np.arange(local_data_sp.shape[0]),10**(local_data_sp[:,1]),color='g',alpha=0.5,label="Target")
    plt.xlabel("Target")
    plt.ylabel("Estimated Volume m3")
    #plt.legend()
    plt.title("Scatter Plot In Percentage Volume")
    plt.savefig("sp_val_vol_"+str(epoch)+".png")
    '''


    #del local_data_sp

    return val_data_ar,acc_data_ar,local_data_sp
    
    print(val_data_ar,acc_data_ar)







class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dft_filters_ch1 = Paramfilter(n_filt=1024, ks=1024, stri=512)
        self.dft_filters_ch2 = Paramfilter(n_filt=1024, ks=1024, stri=512)
        self.batch_size=128
        self.stft_ch1 = Encoder(self.dft_filters_ch1)
        self.stft_ch2 = Encoder(self.dft_filters_ch2)
        self.conv1d_down_1_depth = nn.Conv1d(769,769,10,stride=1,groups=769)
        self.conv1d_down_1_point = nn.Conv1d(769,384,1,stride=1,padding=0)
        self.bn_1= nn.LayerNorm([384,54])
        self.relu=nn.ReLU()
        self.avgpool=nn.AvgPool1d(31,stride=2)
        self.conv1d_down_2_depth = nn.Conv1d(384, 384, 10, stride=1,groups=384,dilation=2)
        self.conv1d_down_2_point = nn.Conv1d(384,192,1,stride=1)
        self.bn_2=nn.LayerNorm([192,36])
        self.conv1d_down_3_depth = nn.Conv1d(192, 192,2, stride=1,groups=192,dilation=4)
        self.conv1d_down_3_point = nn.Conv1d(192,96,1,stride=1)
        self.bn_3=nn.LayerNorm([96,32])
        #self.conv1d_down_4_depth = nn.Conv1d(64, 64, 2, stride=1)
        #self.conv1d_down_4 = nn.Conv1d(10, 5, 10, stride=1)
        self.drp_1=nn.Dropout(p=0.2)
        self.drp = nn.Dropout(p=0.5)

        self.fc = nn.Linear(96,1)


    def forward(self,ch1,ch2):
        #print(ch1.shape)
        #enc_ch1=self.stft_ch1(ch1)
        #enc_ch2=self.stft_ch2(ch2)
        #print(enc_ch1.shape)
        #enc_ch1=enc_ch1[:,:257,:]**2+enc_ch1[:,257:,:]**2
        #enc_ch2=enc_ch2[:,:257,:]**2+enc_ch2[:,257:,:]**2
        enc_ch1=torch.stft(ch1.view(128,-1),n_fft=1536,hop_length=768,normalized=True)
        #enc_ch2=self.stft_ch2(ch2)
        #print(enc_ch1.shape)

        enc_ch1=enc_ch1[:,:,:,0]**2+enc_ch1[:,:,:,1]**2
        
        
        #print(enc_ch1.shape,enc_ch2.shape)
        #enc_ch1=enc_ch1.view(self.batch_size,1,-1)
        #enc_ch2=enc_ch2.view(self.batch_size,1,-1)
        #enc_ch1=enc_ch1**2
        #enc_ch2=enc_ch2**2
        #x=torch.cat((enc_ch1,enc_ch2),1)
        #print(x.shape)
        #print("After cat",x.shape)
        x=enc_ch1
        x=self.relu(self.conv1d_down_1_depth(x))
        
        #print("Depth",x.shape)
        x=self.bn_1(self.relu(self.conv1d_down_1_point(x)))
        
        #print("Point",x.shape)
        #x=self.avgpool(x)
        #print("Avg_pool",x.shape)
        #print("ch1_conv1d",x1.shape)
        #x2=F.relu(self.conv1d_ch1_ch2(enc_ch2))
        #x=self.avgpool(torch.cat((x1,x2),1))
    
        #print(x.shape)
        #print("concat in dimen 1",x.shape)
        x=self.relu(self.conv1d_down_2_depth(x))
        
        #print(x.shape)
        x=self.bn_2(self.relu(self.conv1d_down_2_point(x)))
        
        x=self.drp_1(x)

        #x=self.drp(x)
        
        print(x.shape)
        #x=self.avgpool(x)
        #print("Avg_pool",x.shape)

        x = self.relu(self.conv1d_down_3_depth(x))
        
        #print(x.shape)
        x=self.bn_3(self.relu(self.conv1d_down_3_point(x)))
        
        #x=self.drp_1(x)

        #print(x.shape)
        x=self.avgpool(x)
        
        #print("Avg_pool",x.shape)
        #print(x.shape)
        
        #x = self.relu(self.conv1d_down_3_depth(x))
        #print(x.shape)
        #x = self.relu(self.conv1d_down_3_point(x))
        #print(x.shape)

        #print(x.shape)
        #x = self.avgpool(F.relu(self.conv1d_down_4(x)))
        #print(x.shape)
        x=self.drp(x)

        x=self.fc(x.view(self.batch_size,-1))
        
        #print(x.shape)
        
        return x


train_data=dl.binuaral_dataset('train_random_ar.npy','/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_.hdf5','/home/psrivastava/baseline/sofamyroom_2/conf_room_setup.yml')
val_data=dl.binuaral_dataset('val_random_ar.npy','/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_.hdf5','/home/psrivastava/baseline/sofamyroom_2/conf_room_setup.yml')

#test_data=dl.binuaral_dataset('test_random_ar.npy','/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_.hdf5','/home/psrivastava/baseline/sofamyroom_2/conf_room_setup.yml')

train_dl=DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
val_dl=DataLoader(val_data, batch_size=128, shuffle=True, num_workers=0, drop_last=True)


#test_dl=Dataloader(test_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
#writer = SummaryWriter()

#use_cuda=False
#device = torch.device("cuda" if use_cuda else "cpu")
#net=Model().to(device)
net=Model().to(torch.device('cuda'))
#net=Model()

optimizer = optim.Adam(net.parameters(), lr=0.0001)

#loss_func=PITLossWrapper(multisrc_mse,pit_from='perm_avg')
loss_func=nn.MSELoss()
#tr_loss=0
ar_loss=[]
batch_loss_ar=[]
total_batch_idx=0
val_data_ar=[]
acc_data_ar=[]
save_best_val=0
local_dt_sp=np.zeros((1,2))
#scatter_plot_data_acc=[]

#dummy_input=(torch.randn(4,1,48000).float(),torch.randn(4,1,48000).float())
#net(torch.randn(128,1,48000),torch.randn(128,1,48000))
#torch.onnx.export(net,dummy_input,"model.onnx")
#writer.add_graph(net,dummy_input)
#writer.flush()

path="/home/psrivastava/baseline/scripts/pre_processing/exp-1/"

for epoch in range(40):
        ar_loss,batch_loss_ar=train(net, train_dl, optimizer, epoch,ar_loss,batch_loss_ar,loss_func)
        val_data_ar , acc_data_ar , local_dt_sp= val(net, val_dl, optimizer ,epoch , val_data_ar, acc_data_ar, loss_func)

        np.save("ar_loss_vol_961.npy",ar_loss)
        np.save("batch_loss_ar_vol_961.npy",batch_loss_ar)
        np.save("val_data_ar_vol_961.npy",val_data_ar)
        np.save("acc_data_ar_vol_961.npy",acc_data_ar)
        
        #save best model
        if epoch == 0:
            save_best_val=val_data_ar[-1]
        elif save_best_val > val_data_ar[-1]:
            torch.save({'model_dict':net.state_dict(),'optimizer_dic':optimizer.state_dict(),'epoch':epoch}, "tas_save_best_vol.pt")
            save_best_val=val_data_ar[-1]
            np.save("layer_norm_961ms_mag_"+str(epoch)+".npy",local_dt_sp)



'''
abc=torch.randn(48000)
cdf=torch.randn(48000)
output=net(abc,cdf)
print(output.shape)
target=torch.randn(1,3)
crite=nn.MSELoss()
loss=crite(output,target)
print(loss)
'''




