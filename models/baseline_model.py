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
import data_loader as dl
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



def pred(target_value,out_value,l_pos,w_pos,h_pos):
    l_pos_t=0
    w_pos_t=0
    h_pos_t=0

    #s_pos_t=0
    for i in range(128):
        #l_pos_t+=torch.abs(target_value[i,0]-out_value[i,0])/target_value[i,0]
        #l_pos_t+=(target_value[i,0]-out_value[i,0])**2
        #w_pos_t+=torch.abs(target_value[i,1]-out_value[i,1])/target_value[i,1]
        #w_pos_t+=(target_value[i,1]-out_value[i,1])**2
        #h_pos_t+=torch.abs(target_value[i,2]-out_value[i,2])/target_value[i,2]
        h_pos_t+=(target_value[i,2]-out_value[i,2])**2

        w_pos_t+=(target_value[i,1]-out_value[i,1])**2
        l_pos_t+=(target_value[i,0]-out_value[i,0])**2

        '''
        if target_value[i,0] == out_value [i,0]:
            l_pos+=1
        if target_value[i,1] == out_value[i,1]:
            w_pos+=1
        if target_value[i,2] == out_value[i,2]:
            h_pos+=1
        '''

    return (l_pos_t/128),(w_pos_t/128),(h_pos_t/128)



def train(model, train_loader, optimizer, epoch,ar_loss,batch_loss_ar,loss_func):
    model.train()
    print("training....Epoch",epoch)
    loss_batch=0
    tr_loss=0
    adcc=np.zeros((1,3))
    for batch_idx,  sample_batched in enumerate(train_loader):
        data ,target = sample_batched['bnsample'].float() ,sample_batched['label_dimen'].float().to(device="cuda")

        optimizer.zero_grad()

        output = model(data[:,0,:].to(device="cuda"),data[:,1,:].to(device="cuda"))

        #target=target[:,0]*target[:,1]*target[:,2]
        #surface_area=(target[:,0]*target[:,1]).view(128,-1)

        #target_=torch.cat([surface_area,target[:,2].view(128,-1)],axis=1)
        
        target_,idx=torch.sort(target,1)
        
        #output[:,2]=output[:,1]+output[:,2]
        #output[:,1]=output[:,0]+output[:,1]

        loss = loss_func(output, torch.log10(target_))

        loss.backward()
        optimizer.step()
        #print(loss,loss.item())
        loss_batch=float(loss.item())+loss_batch

        tr_loss=float(loss.item())+tr_loss

        if epoch==0:
            adcc=np.concatenate((adcc,np.log10(target_.detach().cpu().clone().numpy().reshape(128,3))),axis=0)



        del loss, data,target
        #loss.backward()

        #optimizer.step()

        if batch_idx % 100 == 99:
            print("Running loss after 100 batches",(loss_batch/100),loss_batch)
            batch_loss_ar.append(loss_batch/100)
            loss_batch=0
        #if batch_idx == 102:
        #    break
    print("Epoch Loss", (tr_loss/batch_idx),epoch)
    ar_loss.append(tr_loss/batch_idx)
    return ar_loss,batch_loss_ar,adcc


def val(model, train_loader, optimizer, epoch, val_data_ar, acc_data_ar, loss_func):
    model.eval()
    #val_acc=0

    val_loss=0

    l_pos=0

    h_pos=0

    w_pos=0

    #s_pos=0

    local_data_sp=np.zeros([1,6])

    for batch_idx,  sample_batched in enumerate(train_loader):

        data, target = sample_batched['bnsample'].float(), sample_batched['label_dimen'].float().to(device="cuda")

        l_pos_t=0

        h_pos_t=0

        w_pos_t=0

        #s_pos_t=0

        output = model(data[:,0,:].to(device="cuda"),data[:,1,:].to(device="cuda"))

        #surface_area=(target[:,0]*target[:,1]).view(128,-1)

        #target_=torch.cat([surface_area,target[:,2].view(128,-1)],axis=1)

        target_,idx=torch.sort(target,1)

        val_loss_t = loss_func(output, torch.log10(target_))

        val_loss = float(val_loss_t.item())+val_loss

        l_pos_t,w_pos_t,h_pos_t=pred(torch.log10(target_),output,l_pos,w_pos,h_pos)

        ass=np.concatenate((output.detach().cpu().clone().numpy().reshape(128,3),np.log10(target_.detach().cpu().clone().numpy().reshape(128,3))),axis=1)


        local_data_sp=np.concatenate((local_data_sp,ass),axis=0)


        l_pos=l_pos+float(l_pos_t.item())

        h_pos=h_pos+float(h_pos_t.item())

        w_pos=w_pos+float(w_pos_t.item())

        #s_pos=s_pos+float(s_pos_t.item())


        del val_loss_t,h_pos_t,w_pos_t,l_pos_t,ass

        if batch_idx % 100 == 99:
            #val_data_ar.append((val_loss/100))
            #print("validation loss after 100 batches",val_data_ar[-1])

            #print("Accuracy after 100 batches",(s_pos/100),(h_pos/100),s_pos,h_pos)

            l_acc=(l_pos/100)

            w_acc=(w_pos/100)

            h_acc=(h_pos/100)

            #s_acc=(s_pos/100)

            acc_data_ar.append((l_acc,w_acc,h_acc))

            #val_loss=0

            l_pos=0

            h_pos=0

            w_pos=0

            #s_pos=0

            del l_acc,h_acc,w_acc

        #if batch_idx == 102:
            #break

    val_data_ar.append((val_loss/batch_idx))

    '''
    plt.clf()
    plt.scatter(10**(local_data_sp[:,2]),10**(local_data_sp[:,0]),color='green',alpha=0.5)
    #plt.scatter(np.arange(local_data_sp.shape[0]),local_data_sp[:,1],color='g',alpha=0.5,label="Target")
    plt.xlabel("Target")
    plt.ylabel("Estimated Surface Area m2")
    #plt.legend()
    plt.title("Scatter Plot")

    plt.savefig("height_surface_"+str(epoch)+".png")

    plt.clf()
    plt.scatter(10**(local_data_sp[:,3]),10**(local_data_sp[:,1]),color='orange',alpha=0.5)
    #plt.scatter(np.arange(local_data_sp.shape[0]),10**(local_data_sp[:,1]),color='g',alpha=0.5,label="Target")
    plt.xlabel("Target")
    plt.ylabel("Estimated Height")
    #plt.legend()
    plt.title("Scatter Plot")
    plt.savefig("h_surf_"+str(epoch)+".png")

    del local_data_sp
    '''




    return val_data_ar,acc_data_ar,local_data_sp
    print(val_data_ar,acc_data_ar)




class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.dft_filters_ch1 = STFTFB(n_filters=1024, kernel_size=1024, stride=512)
        # self.dft_filters_ch2 = STFTFB(n_filters=1024, kernel_size=1024, stride=512)

        self.batch_size = 128
        # self.stft_ch1 = Encoder(self.dft_filters_ch1)
        # self.stft_ch2 = Encoder(self.dft_filters_ch2)
        self.conv1d_down_1_depth = nn.Conv1d(769, 769, kernel_size=10, stride=1, groups=769, padding=0)
        self.conv1d_down_1_point = nn.Conv1d(769, 384, kernel_size=1, stride=1, padding=0)
        self.bn_1 = nn.LayerNorm([384, 54])

        #self.pip_conv_1d = nn.Conv1d(2307, 2307, kernel_size=(10),stride=1, groups=2307, padding=0)
        #self.pip_conv_1p = nn.Conv1d(2307, 1152, kernel_size=(1), stride=1, padding=0)
        #self.bn_pip_1 = nn.LayerNorm([1152, 54])

        # self.bn_1= nn.LayerNorm([64,365])
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool1d(32, stride=1)

        self.conv1d_down_2_depth = nn.Conv1d(384, 384, kernel_size=10, stride=1, groups=384, dilation=2, padding=0)
        self.conv1d_down_2_point = nn.Conv1d(384, 192, kernel_size=1, stride=1)
        self.bn_2 = nn.LayerNorm([192, 36])

        #self.pip_conv_2d = nn.Conv1d(1152, 1152, kernel_size=10, stride=1, groups=1152 ,dilation=2, padding=0)
        #self.pip_conv_2p = nn.Conv1d(1152, 576, kernel_size=(1), stride=1, padding=0)
        #self.bn_pip_2 = nn.LayerNorm([576, 36])

        # self.bn_2=nn.LayerNorm([32,347])
        self.conv1d_down_3_depth = nn.Conv1d(192, 192, kernel_size=2, stride=1, groups=192, dilation=4, padding=0)
        self.conv1d_down_3_point = nn.Conv1d(192, 96, kernel_size=1, stride=1)
        self.bn_3 = nn.LayerNorm([96, 32])

        #self.pip_conv_3d = nn.Conv1d(576, 576, kernel_size=2, stride=1, groups=576, dilation=4, padding=0)
        #self.pip_conv_3p = nn.Conv1d(576, 288, kernel_size=1, stride=1, padding=0)
        #self.bn_pip_3 = nn.LayerNorm([288, 32])
        #self.attn_self=MultiHeadAttention(in_features=320,head_num=2)
        # self.rnn=nn.GRU(31,20,2,batch_first=True)
        # self.bn_3=nn.LayerNorm([16,311])
        # self.conv1d_down_4_depth = nn.Conv1d(64, 64, 10, stride=1,groups=64,dilation=8)
        # self.conv1d_down_4 = nn.Conv1d(64, 32, 1, stride=1)
        self.drp_1 = nn.Dropout(p=0.2)
        self.drp = nn.Dropout(p=0.5)

        self.fc = nn.Linear(96, 3)

    def forward(self, ch1, ch2):
        print(ch1.shape)
        enc_ch1 = torch.stft(ch1.view(128, -1), n_fft=1536, hop_length=768, return_complex=True)

        #enc_ch2 = torch.stft(ch2.view(128, -1), n_fft=1536, hop_length=768, return_complex=True)

        f=torch.view_as_real(enc_ch1)

        f=torch.sqrt(f[:,:,:,0]**2+f[:,:,:,1]**2)

        '''
        cc = enc_ch1 * torch.conj(enc_ch2)
        ipd = cc / torch.abs(cc)
        ipd_ri=torch.view_as_real(ipd)
        ild=torch.log(torch.abs(enc_ch1)+10e-8)-torch.log(torch.abs(enc_ch2)+10e-8)

        x2=torch.cat((ipd_ri[:,:,:,0],ipd_ri[:,:,:,1],ild),axis=1)
        '''
        
        x=f



        x = self.relu(self.conv1d_down_1_depth(x))

        #x2=self.bn_pip_1(self.relu(self.pip_conv_1p(self.relu(self.pip_conv_1d(x2)))))

        x = self.bn_1(self.relu(self.conv1d_down_1_point(x)))

        x = self.relu(self.conv1d_down_2_depth(x))


        x = self.bn_2(self.relu(self.conv1d_down_2_point(x)))

        #x2 = self.bn_pip_2(self.relu(self.pip_conv_2p(self.relu(self.pip_conv_2d(x2)))))


        x = self.drp_1(x)

        #x2 = self.drp_1(x2)

        x = self.relu(self.conv1d_down_3_depth(x))


        x = self.bn_3(self.relu(self.conv1d_down_3_point(x)))

        #x2 = self.bn_pip_3(self.relu(self.pip_conv_3p(self.relu(self.pip_conv_3d(x2)))))

        #x=torch.cat((x,x2),axis=1)
        
        #print(x.shape)
        x= self.avgpool(x)

        print(x.shape)
        x = self.drp(x)

        x = self.fc(x.reshape(self.batch_size, -1))

        x[:,2]=x[:,1]+x[:,2]
        
        x[:,1]=x[:,0]+x[:,1]

        print(x.shape)

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

net=Model().to(torch.device("cuda"))
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
adcc=np.zeros((1,3))
local_dt_sp=np.zeros((1,6))
#dummy_input=(torch.randn(4,1,48000).float(),torch.randn(4,1,48000).float())
#net(torch.randn(128,1,48000),torch.randn(128,1,48000))
#torch.onnx.export(net,dummy_input,"model.onnx")
#writer.add_graph(net,dummy_input)
#writer.flush()


for epoch in range(30):
        ar_loss,batch_loss_ar,adcc=train(net, train_dl, optimizer, epoch,ar_loss,batch_loss_ar,loss_func)
        val_data_ar , acc_data_ar,local_dt_sp = val(net, val_dl, optimizer ,epoch , val_data_ar, acc_data_ar, loss_func)

        np.save("ar_loss.npy",ar_loss)
        np.save("batch_loss_ar.npy",batch_loss_ar)
        np.save("val_data_ar.npy",val_data_ar)
        np.save("acc_data_ar.npy",acc_data_ar)

         #save best model
        if epoch == 0:
            save_best_val=val_data_ar[-1]
            np.save("dummy_input_mean_sh.npy",adcc)
        elif save_best_val > val_data_ar[-1]:
            torch.save({'model_dict':net.state_dict(),'optimizer_dic':optimizer.state_dict(),'epoch':epoch}, "tas_save_best_sh.pt")
            save_best_val=val_data_ar[-1]
            np.save("sh_mag_96ms_"+str(epoch)+".npy",local_dt_sp)


#torch.save({'model_state_dict':net.state_dict(),'optimizer_state_dict':optimizer.state_dict()}, "tas_bas.pt")


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


'''

class Model(torch.nn.Module):

    def __init__(self, p_dropout):
        super().__init__()

        self.dft_filters_ch1 = Paramfilter(n_filt=1024, ks=1024, stri=512)

        self.batch_size=128
        self.stft_ch1 = Encoder(self.dft_filters_ch1)

        #self.tdnn1_d = nn.Conv1d(1026,1026,5,stride=1,groups=1026,dilation=1)
        #self.tdnn1_p = nn.Conv1d(1026,512,1,stride=1,padding=0)

        self.bn_tdnn1= nn.LayerNorm([512,88])

        self.tdnn1 = nn.Conv1d(in_channels=1026, out_channels=512, kernel_size=5, dilation=1)
        #self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)


        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        #self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        #self.tdnn2_d = nn.Conv1d(512,512,5,stride=1,groups=512,dilation=2)
        #self.tdnn2_p = nn.Conv1d(512,512,1,stride=1,padding=0)
        self.bn_tdnn2= nn.LayerNorm([512,80])


        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        #self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        #self.tdnn3_d = nn.Conv1d(512,512,7,stride=1,groups=512,dilation=3)
        #self.tdnn3_p = nn.Conv1d(512,512,1,stride=1,padding=0)
        self.bn_tdnn3= nn.LayerNorm([512,62])




        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn_tdnn4= nn.LayerNorm([512,62])

        #self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn_tdnn5= nn.LayerNorm([1500,62])

        #self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(3000,512)
        self.bn_fc1= nn.LayerNorm([512])

        #self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(512,512)
        #self.bn_fc2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.bn_fc2= nn.LayerNorm([512])

        self.dropout_fc2 = nn.Dropout(p=p_dropout)

        self.fc3 = nn.Linear(512,2)

    def forward(self, ch1,ch2):
        # Note: x must be (batch_size, feat_dim, chunk_len)

        x = self.stft_ch1(ch1)
        #print(x.shape)
        x= self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))

        #x = self.dropout_tdnn1(self.bn_tdnn1(self.tdnn1_p((F.relu(self.tdnn1_d(x)))))
        print(x.shape)
        #x= self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2_p(F.relu(self.tdnn2_d(x))))))

        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        print(x.shape)
        #x= self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3_p(F.relu(self.tdnn3_d(x))))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        print(x.shape)
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        #print(x.shape)
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))
        print(x.shape)


        if self.training:
            shape = x.size()
            noise = torch.cuda.FloatTensor(shape)
            torch.randn(shape, out=noise)
            x += noise*eps


        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        print("stats",stats.shape)
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))
        print(x.shape)
        x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        print(x.shape)
        x = self.fc3(x)
        print(x.shape)
        return x
'''
