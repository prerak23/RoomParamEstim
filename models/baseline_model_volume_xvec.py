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
import data_loader_volume as dl
import matplotlib.pyplot as plt
import seaborn as sns
from torch_multi_head_attention import MultiHeadAttention

# from sklearn.metrics import mean_squared_error
# import torch.onnx
# from torch.utils.tensorboard import SummaryWriter


sns.set_theme()


class Paramfilter(Filterbank):
    def __init__(self, n_filt, ks, stri):
        super(Paramfilter, self).__init__(n_filt, ks, stri)
        self._filters = torch.nn.Parameter(STFTFB(n_filters=n_filt, kernel_size=ks, stride=stri).filters())

    def filters(self):
        return self._filters


def pred(target_value, out_value):
    v_pos_t = 0
    v_pos_mse = 0
    for i in range(128):
        # l_pos+=np.abs(target_value[i,0]-out_value[i,0])/target_value[i,0]
        # w_pos+=np.abs(target_value[i,1]-out_value[i,1])/target_value[i,1]
        # h_pos+=np.abs(target_value[i,2]-out_value[i,2])/target_value[i,2]
        v_pos_t += torch.abs(target_value[i, 0] - out_value[i, 0]) / target_value[i, 0]
        v_pos_mse += (target_value[i, 0] - out_value[i, 0]) ** 2
        '''
        if target_value[i,0] == out_value [i,0]:
            l_pos+=1
        if target_value[i,1] == out_value[i,1]:
            w_pos+=1
        if target_value[i,2] == out_value[i,2]:
            h_pos+=1
        '''
    v_pos_mse_t = v_pos_mse / 128
    v_pos_k = (v_pos_t / 128) * 100
    return v_pos_k, v_pos_mse_t


def train(model, train_loader, optimizer, epoch, ar_loss, batch_loss_ar, loss_func):
    model.train()

    print("training....Epoch", epoch)

    loss_batch = 0

    tr_loss = 0

    adcc = np.zeros((1, 1))

    for batch_idx, sample_batched in enumerate(train_loader):

        data, target = sample_batched['bnsample'].float(), sample_batched['label_dimen'].float().to(device='cuda')

        optimizer.zero_grad()
        print(data.shape)
        output = model(data[:, 0, :].to(device='cuda'), data[:, 1, :].to(device='cuda'))

        # print(target)

        # print(target.shape)

        target = target[:, 0] * target[:, 1] * target[:, 2]

        # print(target)

        # print(target.shape)

        loss = loss_func(output, torch.log10(target.view(128, 1)))

        loss.backward()

        optimizer.step()

        # print(loss,loss.item())

        loss_batch = float(loss.item()) + loss_batch

        tr_loss = float(loss.item()) + tr_loss

        # loss.backward()

        # optimizer.step()

        if epoch == 0:
            adcc = np.concatenate((adcc, np.log10(target.detach().cpu().clone().numpy().reshape(128, 1))), axis=0)

        del loss, target, data

        if batch_idx % 100 == 99:
            print("Running loss after 100 batches", (loss_batch / 100), loss_batch)

            batch_loss_ar.append(loss_batch / 100)

            loss_batch = 0
        # if batch_idx == 102:
        #    break
    print("Epoch Loss", (tr_loss / batch_idx), epoch)
    ar_loss.append(tr_loss / batch_idx)
    return ar_loss, batch_loss_ar, adcc


def val(model, train_loader, optimizer, epoch, val_data_ar, acc_data_ar, loss_func):
    model.eval()

    # val_acc=0

    val_loss = 0  # Calculate Validation Loss Per Epoch

    v_pos = 0  # Calculate Validation Accuracy Per Batch

    v_pos_mse = 0  # Calculate Validation Accuracy Per Batch MSE

    local_data_sp = np.zeros((1, 2))

    for batch_idx, sample_batched in enumerate(train_loader):

        data, target = sample_batched['bnsample'].float(), sample_batched['label_dimen'].float().to(device='cuda')

        target = target[:, 0] * target[:, 1] * target[:, 2]

        output = model(data[:, 0, :].to(device='cuda'), data[:, 1, :].to(device='cuda'))

        val_loss_t = loss_func(output, torch.log10(target.view(128, 1)))

        val_loss = float(val_loss_t.item()) + val_loss

        v_pos_ten, v_pos_mse_ten = pred(torch.log10(target.view(128, 1)), output)

        v_pos += float(v_pos_ten.item())
        v_pos_mse += float(v_pos_mse_ten.item())

        ass = np.concatenate((output.detach().cpu().clone().numpy().reshape(128, 1),
                              np.log10(target.detach().cpu().clone().numpy().reshape(128, 1))), axis=1)

        local_data_sp = np.concatenate((local_data_sp, ass), axis=0)

        # l_pos,w_pos,h_pos=pred(target,output,l_pos,w_pos,h_pos)

        del val_loss_t, v_pos_ten, v_pos_mse_ten, ass

        if batch_idx % 100 == 99:
            # print("validation loss after 100 batches",val_data_ar[-1])

            print("Volume", v_pos / 100)

            # acc_data_ar.append((l_acc,w_acc,h_acc))
            acc_data_ar.append([(v_pos / 100), (v_pos_mse / 100)])

            # val_loss=0

            v_pos = 0
            v_pos_mse = 0

        # if batch_idx == 102:
        # break

    val_data_ar.append((val_loss / batch_idx))

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

    # del local_data_sp

    return val_data_ar, acc_data_ar, local_data_sp

    print(val_data_ar, acc_data_ar)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.dft_filters_ch1 = STFTFB(n_filters=1024, kernel_size=1024, stride=512)
        # self.dft_filters_ch2 = STFTFB(n_filters=1024, kernel_size=1024, stride=512)

        self.batch_size = 128
        # self.stft_ch1 = Encoder(self.dft_filters_ch1)
        # self.stft_ch2 = Encoder(self.dft_filters_ch2)
        self.conv1d_down_1_depth = nn.Conv1d(1025, 1025, kernel_size=10, stride=1, groups=1025, padding=0)
        self.conv1d_down_1_point = nn.Conv1d(1025, 512, kernel_size=1, stride=1, padding=0)
        self.bn_1 = nn.LayerNorm([512, 38])

        #self.pip_conv_1d = nn.Conv1d(3075, 3075, kernel_size=(10),stride=1, groups=3075, padding=0)
        #self.pip_conv_1p = nn.Conv1d(3075, 1537, kernel_size=(1), stride=1, padding=0)
        #self.bn_pip_1 = nn.LayerNorm([1537, 38])

        # self.bn_1= nn.LayerNorm([64,365])
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool1d(16, stride=1)

        self.conv1d_down_2_depth = nn.Conv1d(512, 512, kernel_size=10, stride=1, groups=512, dilation=2, padding=0)
        self.conv1d_down_2_point = nn.Conv1d(512, 256, kernel_size=1, stride=1)
        self.bn_2 = nn.LayerNorm([256, 20])

        #self.pip_conv_2d = nn.Conv1d(1537, 1537, kernel_size=10, stride=1, groups=1537 ,dilation=2, padding=0)
        #self.pip_conv_2p = nn.Conv1d(1537, 764, kernel_size=(1), stride=1, padding=0)
        #self.bn_pip_2 = nn.LayerNorm([764, 20])

        # self.bn_2=nn.LayerNorm([32,347])
        self.conv1d_down_3_depth = nn.Conv1d(256, 256, kernel_size=2, stride=1, groups=256, dilation=4, padding=0)
        self.conv1d_down_3_point = nn.Conv1d(256, 128, kernel_size=1, stride=1)
        self.bn_3 = nn.LayerNorm([128, 16])

        #self.pip_conv_3d = nn.Conv1d(764, 764, kernel_size=2, stride=1, groups=764, dilation=4, padding=0)
        #self.pip_conv_3p = nn.Conv1d(764, 382, kernel_size=1, stride=1, padding=0)
        #self.bn_pip_3 = nn.LayerNorm([382, 16])
        #self.attn_self=MultiHeadAttention(in_features=320,head_num=2)
        # self.rnn=nn.GRU(31,20,2,batch_first=True)
        # self.bn_3=nn.LayerNorm([16,311])
        # self.conv1d_down_4_depth = nn.Conv1d(64, 64, 10, stride=1,groups=64,dilation=8)
        # self.conv1d_down_4 = nn.Conv1d(64, 32, 1, stride=1)
        self.drp_1 = nn.Dropout(p=0.2)
        self.drp = nn.Dropout(p=0.5)

        self.fc = nn.Linear(128, 1)

    def forward(self, ch1, ch2):
        print(ch1.shape)
        enc_ch1 = torch.stft(ch1.view(128, -1), n_fft=2048, hop_length=1024, return_complex=True)
        #print("Stft output",enc_ch1.shape)
        #enc_ch2 = torch.stft(ch2.view(128, -1), n_fft=2048, hop_length=1024, return_complex=True)
        #print("Stft output 2",enc_ch2.shape)
        # enc_ch2=self.stft_ch2(ch2)
        # print(enc_ch1.shape)
        f=torch.view_as_real(enc_ch1)
        # print("imag to real",f.shape)
        # f=torch.unsqueeze(,3)
        f=torch.sqrt(f[:,:,:,0]**2+f[:,:,:,1]**2)
        # print(f.shape)
        # conj_mul=torch.view_as_real(enc_ch1*torch.conj(enc_ch2))
        '''
        cc = enc_ch1 * torch.conj(enc_ch2)
        ipd = cc / torch.abs(cc)
        ipd_ri=torch.view_as_real(ipd)
        ild=torch.log(torch.abs(enc_ch1)+10e-8)-torch.log(torch.abs(enc_ch2)+10e-8)
        x2=torch.cat((ipd_ri[:,:,:,0],ipd_ri[:,:,:,1],ild),axis=1)
        '''
        x=f




        # print("conj_mul",conj_mul.shape)
        # gcc_phat=torch.view_as_real(gcc_phat)
        # r=gcc_phat[:,:,:,0]
        # print(r.shape)
        # i=gcc_phat[:,:,:,1]
        # print(i.shape)
        # x=torch.cat((f,r,i),axis=1)
        # print(x.shape)

        # enc_ch2=enc_ch2[:,:257,:]**2+enc_ch2[:,257:,:]**2

        # print(enc_ch1.shape,enc_ch2.shape)
        # enc_ch1=enc_ch1.view(self.batch_size,1,-1)
        # enc_ch2=enc_ch2.view(self.batch_size,1,-1)
        # enc_ch1=enc_ch1**2
        # enc_ch2=enc_ch2**2
        # x=torch.cat((enc_ch1,enc_ch2),1)
        # print(x.shape)
        # print("After cat",x.shape)
        # print(x.shape)
        #x_r = torch.view_as_real(gcc_phat)
        #x = x_r[:, :, :, 0] ** 2 + x_r[:, :, :, 1] ** 2
        # x=torch.cat((x_r[:,:,:,0],x_r[:,:,:,1]),axis=1)

        # x=torch.sqrt(x_r[:,:,:,0]**2+x_r[:,:,:,1]**2)
        # print(x.shape)

        x = self.relu(self.conv1d_down_1_depth(x))
        #x2=self.bn_pip_1(self.relu(self.pip_conv_1p(self.relu(self.pip_conv_1d(x2)))))

        # print("Depth",x.shape)
        x = self.bn_1(self.relu(self.conv1d_down_1_point(x)))

        # print("Point",x.shape)
        # x=self.avgpool(x)
        # print("Avg_pool",x.shape)
        # print("ch1_conv1d",x1.shape)
        # x2=F.relu(self.conv1d_ch1_ch2(enc_ch2))
        # x=self.avgpool(torch.cat((x1,x2),1))

        # print(x.shape)
        # print("concat in dimen 1",x.shape)
        x = self.relu(self.conv1d_down_2_depth(x))

        # print(x.shape)
        x = self.bn_2(self.relu(self.conv1d_down_2_point(x)))

        #x2 = self.bn_pip_2(self.relu(self.pip_conv_2p(self.relu(self.pip_conv_2d(x2)))))

        x = self.drp_1(x)
        #x2 = self.drp_1(x2)

        # x=self.drp(x)

        # print(x.shape)
        # x=self.avgpool(x)
        # print("Avg_pool",x.shape)

        x = self.relu(self.conv1d_down_3_depth(x))

        # print(x.shape)
        x = self.bn_3(self.relu(self.conv1d_down_3_point(x)))

        #x2 = self.bn_pip_3(self.relu(self.pip_conv_3p(self.relu(self.pip_conv_3d(x2)))))
        #print(x2.shape)
        #x=torch.cat((x,x2),axis=1)
        #print(x.shape)
        x= self.avgpool(x)
    
        # x,h=self.rnn(x)
        # print("Avg_pool",x.shape)
        # print(x.shape)

        # x = self.relu(self.conv1d_down_3_depth(x))
        # print(x.shape)
        # x = self.relu(self.conv1d_down_3_point(x))
        # print(x.shape)

        # print(x.shape)
        # x = self.avgpool(F.relu(self.conv1d_down_4(x)))
        # print(x.shape)
        x = self.drp(x)
        
        x = self.fc(x.reshape(self.batch_size, -1))

        print(x.shape)

        return x




train_data = dl.binuaral_dataset('train_random_ar_volume.npy','/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_2.hdf5','/home/psrivastava/baseline/sofamyroom_2/conf_room_setup_volume.yml')
val_data = dl.binuaral_dataset('val_random_ar_volume.npy','/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_2.hdf5','/home/psrivastava/baseline/sofamyroom_2/conf_room_setup_volume.yml')

# test_data=dl.binuaral_dataset('test_random_ar.npy','/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_.hdf5','/home/psrivastava/baseline/sofamyroom_2/conf_room_setup.yml')

train_dl = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
val_dl = DataLoader(val_data, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
# test_dl=Dataloader(test_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
# writer = SummaryWriter()

# use_cuda=False
# device = torch.device("cuda" if use_cuda else "cpu")
# net=Model().to(device)
net = Model().to(torch.device('cuda'))
#net=Model()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
#loss_func=PITLossWrapper(multisrc_mse,pit_from='perm_avg')
loss_func = nn.MSELoss()
# tr_loss=0
ar_loss = []
batch_loss_ar = []
total_batch_idx = 0
val_data_ar = []
acc_data_ar = []
save_best_val = 0
local_dt_sp = np.zeros((1, 2))
adcc = np.zeros((1, 1))
# scatter_plot_data_acc=[]

# dummy_input=(torch.randn(4,1,48000).float(),torch.randn(4,1,48000).float())
#net(torch.randn(128,1,48000),torch.randn(128,1,48000))
# torch.onnx.export(net,dummy_input,"model.onnx")
# writer.add_graph(net,dummy_input)
# writer.flush()


for epoch in range(30):
    ar_loss, batch_loss_ar, adcc = train(net, train_dl, optimizer, epoch, ar_loss, batch_loss_ar, loss_func)
    val_data_ar, acc_data_ar, local_dt_sp = val(net, val_dl, optimizer, epoch, val_data_ar, acc_data_ar, loss_func)

    np.save("ar_loss_vol.npy", ar_loss)
    np.save("batch_loss_ar_vol.npy", batch_loss_ar)
    np.save("val_data_ar_vol.npy", val_data_ar)
    np.save("acc_data_ar_vol.npy", acc_data_ar)

    # save best model
    if epoch == 0:
        save_best_val = val_data_ar[-1]
        np.save("dummy_data_train_net", adcc)
    elif save_best_val > val_data_ar[-1]:
        torch.save({'model_dict': net.state_dict(), 'optimizer_dic': optimizer.state_dict(), 'epoch': epoch},
                   "tas_save_best_vol.pt")
        save_best_val = val_data_ar[-1]
        np.save("mag_128ms_" + str(epoch) + ".npy", local_dt_sp)

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

        #self.dft_filters_ch1 = STFTFB(n_filters=1024, kernel_size=1024, stride=512)

        self.batch_size=128
        #self.stft_ch1 = Encoder(self.dft_filters_ch1)

        self.tdnn1_d = nn.Conv2d(513,513,5,stride=1,groups=513,dilation=1)
        self.tdnn1_p = nn.Conv1d(513,256,1,stride=1,padding=0)

        self.bn_tdnn1= nn.LayerNorm([256,90])

        #self.tdnn1 = nn.Conv1d(in_channels=1026, out_channels=512, kernel_size=5, dilation=1)
        #self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        #self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        #self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.tdnn2_d = nn.Conv1d(256,256,5,stride=1,groups=256,dilation=2)
        self.tdnn2_p = nn.Conv1d(256,256,1,stride=1,padding=0)
        self.bn_tdnn2= nn.LayerNorm([256,82])


        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)
        #self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        #self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.tdnn3_d = nn.Conv1d(256,256,7,stride=1,groups=256,dilation=3)
        self.tdnn3_p = nn.Conv1d(256,256,1,stride=1,padding=0)
        self.bn_tdnn3= nn.LayerNorm([256,64])




        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)
        self.tdnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, dilation=1)
        self.bn_tdnn4= nn.LayerNorm([256,64])

        #self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)
        self.tdnn5 = nn.Conv1d(in_channels=256, out_channels=1500, kernel_size=1, dilation=1)
        self.bn_tdnn5= nn.LayerNorm([1500,64])

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
        self.fc3 = nn.Linear(512,1)
    def forward(self, ch1,ch2):
        # Note: x must be (batch_size, feat_dim, chunk_len)
        #x = self.stft_ch1(ch1)
        enc_ch1=torch.stft(ch1.view(128,-1),n_fft=1024,hop_length=512)
        enc_ch1=enc_ch1[:,:,:,0]**2+enc_ch1[:,:,:,1]**2
        x=enc_ch1
        #print(x.shape)
        #x= self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))

        x = self.dropout_tdnn1(self.bn_tdnn1(self.tdnn1_p((F.relu(self.tdnn1_d(x))))))
        print(x.shape)
        x= self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2_p(F.relu(self.tdnn2_d(x))))))

        #x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        print(x.shape)
        x= self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3_p(F.relu(self.tdnn3_d(x))))))
        #x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
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
