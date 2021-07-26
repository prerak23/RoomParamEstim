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
import data_loader as dl
import matplotlib.pyplot as plt
import seaborn as sns


def pred(mean_values, target_value):
    param_mse_loss = 0

    for i in range(128):
        param_mse_loss += (mean_values[i] - target_value[i]) ** 2

    return param_mse_loss / 128


def NLLloss(y, mean, var):
    """ Negative log-likelihood loss function. """
    return (torch.log(var) + ((y - mean).pow(2)) / var).sum()


def cal_features(ch1, ch2):
    enc_ch1 = torch.stft(ch1.view(128, -1), n_fft=1536, hop_length=768, return_complex=True)

    enc_ch2 = torch.stft(ch2.view(128, -1), n_fft=1536, hop_length=768, return_complex=True)

    f = torch.view_as_real(enc_ch1)

    f = torch.sqrt(f[:, :, :, 0] ** 2 + f[:, :, :, 1] ** 2)  # Magnitude

    # Ipd ild calculation

    cc = enc_ch1 * torch.conj(enc_ch2)
    ipd = cc / torch.abs(cc)
    ipd_ri = torch.view_as_real(ipd)
    ild = torch.log(torch.abs(enc_ch1) + 10e-8) - torch.log(torch.abs(enc_ch2) + 10e-8)

    x2 = torch.cat((ipd_ri[:, :, :, 0], ipd_ri[:, :, :, 1], ild), axis=1)

    print(f.shape, x2.shape)

    return f, x2


std_volume = 106.02265764052878
std_surface = 84.22407627419787

std_rt60=[ 0.7793691,0.7605436,0.6995225, 0.7076664, 0.6420753,0.51794204]
std_abs=[0.100825,0.1172430,0.1002776,0.09108845,0.09378748,0.091663032]

vari_volume=np.array([7731.65694]*128)
vari_surface=np.array([4684.43793]*128)
vari_rt60=[0.23330093,0.1962361,0.0954811,0.08517269,0.06436219,0.04586754]
vari_abs=[0.00571186,0.00716683,0.00592367,0.00545315,0.00556748,0.00549911]

'''
std_rt60_125 = 0.7793691
std_rt60_250 = 0.7605436
std_rt60_500 = 0.6995225
std_rt60_1000 = 0.7076664
std_rt60_2000 = 0.6420753
std_rt60_4000 = 0.51794204
'''
#0.100825,0.1172430,0.1002776,0.09108845,0.09378748,0.091663032












def train(model, train_loader, optimizer, epoch, ar_loss, batch_loss_ar):
    model.train()
    print("training....Epoch", epoch)
    loss_batch = 0
    tr_loss = 0

    #adcc = np.zeros((1, 14))  # Total 14 Values

    #track_var = np.zeros((1, 14))  # Total 14 values

    surface_mse = 0
    volume_mse = 0
    rt60_125_ = 0
    rt60_250_ = 0
    rt60_500_ = 0
    rt60_1000_ = 0
    rt60_2000_ = 0
    rt60_4000_ = 0

    ab_125_ = 0
    ab_250_ = 0
    ab_500_ = 0
    ab_1000_ = 0
    ab_2000_ = 0
    ab_4000_ = 0

    for batch_idx, sample_batched in enumerate(train_loader):
        data, surface, volume, absorption, rt60 = sample_batched['bnsample'].float(), sample_batched[
            'surface'].float().to(device="cuda"), sample_batched['volume'].float().to(device="cuda"), sample_batched[
                                                      'absorption'].float().to(device="cuda"), sample_batched[
                                                      'rt60'].float().to(device="cuda")

        surface = surface.reshape(128, 1)

        volume = volume.reshape(128, 1)

        # rt60=torch.log(rt60)
        # absorption=torch.log(absorption)

        optimizer.zero_grad()

        x_1, ipd_ild = cal_features(data[:, 0, :].to(device="cuda"), data[:, 1, :].to(device="cuda"))

        mean = model(x_1, ipd_ild)

        # target_,idx=torch.sort(target,1)

        loss_surface = NLLloss(surface.view(128) , mean[:, 0], torch.from_numpy(vari_surface).float().to(device="cuda"))

        loss_volume = NLLloss(volume.view(128) , mean[:, 1], torch.from_numpy(vari_volume).float().to(device="cuda"))

        rt60_loss = 0

        absorp_loss = 0

        for i in range(6):
            var_t_rt60=np.array([vari_rt60[i]]*128)
            var_t_abs=np.array([vari_abs[i]]*128)
            var_f_rt60=torch.from_numpy(var_t_rt60).float().to(device="cuda")
            var_f_abs=torch.from_numpy(var_t_abs).float().to(device="cuda")

            rt60_loss += NLLloss(rt60[:, i], mean[:, 2 + i], var_f_rt60)
            absorp_loss += NLLloss(absorption[:, i], mean[:, 8 + i], var_f_abs)

        loss = (rt60_loss + absorp_loss + loss_surface + loss_volume) / 14
        # rt60_loss=rt60_loss/6
        # absorp_loss=absorp_loss/6
        # loss=loss_surface+loss_volume+rt60+absorp_loss/4

        loss.backward()
        optimizer.step()

        surface_mse += float(pred(mean[:, 0], surface.view(128)).item())

        volume_mse += float(pred(mean[:, 1] , volume.view(128)).item())

        rt60_125_ += float(pred(mean[:, 2], rt60[:, 0]).item())
        rt60_250_ += float(pred(mean[:, 3], rt60[:, 1]).item())
        rt60_500_ += float(pred(mean[:, 4], rt60[:, 2]).item())
        rt60_1000_ += float(pred(mean[:, 5], rt60[:, 3]).item())
        rt60_2000_ += float(pred(mean[:, 6], rt60[:, 4]).item())
        rt60_4000_ += float(pred(mean[:, 7], rt60[:, 5]).item())

        ab_125_ += float(pred(mean[:, 8], absorption[:, 0]).item())
        ab_250_ += float(pred(mean[:, 9], absorption[:, 1]).item())
        ab_500_ += float(pred(mean[:, 10], absorption[:, 2]).item())
        ab_1000_ += float(pred(mean[:, 11], absorption[:, 3]).item())
        ab_2000_ += float(pred(mean[:, 12], absorption[:, 4]).item())
        ab_4000_ += float(pred(mean[:, 13], absorption[:, 5]).item())

        loss_batch = float(loss.item()) + loss_batch

        tr_loss = float(loss.item()) + tr_loss

        # print(surface.shape,volume.shape,absorption.shape,rt60.shape)

        #total = torch.cat((surface, volume, rt60, absorption), axis=1)

        #track_var = np.concatenate((track_var, variance.detach().cpu().clone().numpy().reshape(128, 14)), axis=0)

        '''
        if epoch == 0:
            adcc = np.concatenate((adcc, total.detach().cpu().clone().numpy().reshape(128, 14)), axis=0)
        '''

        del loss, data, loss_surface, loss_volume, rt60_loss, absorp_loss

        if batch_idx % 100 == 99:
            # print("Running loss after 100 batches",(loss_batch/100),loss_batch)
            batch_loss_ar.append((surface_mse / 100, volume_mse / 100, (rt60_125_) / 100, (rt60_250_) / 100,
                                  (rt60_500_) / 100, (rt60_1000_) / 100, (rt60_2000_) / 100, (rt60_4000_) / 100,
                                  (ab_125_) / 100, (ab_250_) / 100, (ab_500_) / 100, (ab_1000_) / 100, (ab_2000_) / 100,
                                  (ab_4000_) / 100))

            surface_mse = 0
            volume_mse = 0
            rt60_125_ = 0
            rt60_250_ = 0
            rt60_500_ = 0
            rt60_1000_ = 0
            rt60_2000_ = 0
            rt60_4000_ = 0
            ab_125_ = 0
            ab_250_ = 0
            ab_500_ = 0
            ab_1000_ = 0
            ab_2000_ = 0
            ab_4000_ = 0

            # del surface_mse,volume_mse,rt60_125_,rt60_250_,rt60_500_,rt60_1000_,rt60_2000_,rt60_4000_,ab_125_,ab_250_,ab_500_,ab_1000_,ab_2000_,ab_4000_

            # loss_batch=0

    print("Epoch Loss", (tr_loss / batch_idx), epoch)
    ar_loss.append(tr_loss / batch_idx)
    return ar_loss, batch_loss_ar


def val(model, train_loader, optimizer, epoch, val_data_ar, acc_data_ar):
    model.eval()
    surface_loss_mse = 0
    volume_loss_mse = 0
    val_loss = 0

    rt60_loss_mse_125 = 0
    rt60_loss_mse_250 = 0
    rt60_loss_mse_500 = 0
    rt60_loss_mse_1000 = 0
    rt60_loss_mse_2000 = 0
    rt60_loss_mse_4000 = 0

    absorp_loss_mse_125 = 0
    absorp_loss_mse_250 = 0
    absorp_loss_mse_500 = 0
    absorp_loss_mse_1000 = 0
    absorp_loss_mse_2000 = 0
    absorp_loss_mse_4000 = 0

    local_data_sp = np.zeros([1, 30])  # 42

    for batch_idx, sample_batched in enumerate(train_loader):

        data, surface, volume, absorption, rt60, rm, vp = sample_batched['bnsample'].float(), sample_batched[
            'surface'].float().to(device="cuda"), sample_batched['volume'].float().to(device="cuda"), sample_batched[
                                                              'absorption'].float().to(device="cuda"), sample_batched[
                                                              'rt60'].float().to(device="cuda"), sample_batched[
                                                              'room'].to(device="cuda"), sample_batched['vp'].to(
            device="cuda")

        surface_loss_mse_t = 0

        volume_loss_mse_t = 0

        rt60_loss_mse_125_t = 0
        rt60_loss_mse_250_t = 0
        rt60_loss_mse_500_t = 0
        rt60_loss_mse_1000_t = 0
        rt60_loss_mse_2000_t = 0
        rt60_loss_mse_4000_t = 0

        absorp_loss_mse_125_t = 0
        absorp_loss_mse_250_t = 0
        absorp_loss_mse_500_t = 0
        absorp_loss_mse_1000_t = 0
        absorp_loss_mse_2000_t = 0
        absorp_loss_mse_4000_t = 0

        x_1, ipd_ild = cal_features(data[:, 0, :].to(device="cuda"), data[:, 1, :].to(device="cuda"))

        mean = model(x_1, ipd_ild)

        surface = surface.reshape(128, 1)
        volume = volume.reshape(128, 1)

        # rt60=torch.log(rt60)
        # absorption=torch.log(absorption)
        loss_surface = NLLloss(surface.view(128) , mean[:, 0], torch.from_numpy(vari_surface).float().to(device="cuda"))

        loss_volume = NLLloss(volume.view(128) , mean[:, 1], torch.from_numpy(vari_volume).float().to(device="cuda"))


        #loss_surface = NLLloss(surface.view(128) / std_surface, mean[:, 0], vari_surface)
        #loss_volume = NLLloss(volume.view(128) / std_volume, mean[:, 1], vari_volume)

        rt60_loss = 0
        absorp_loss = 0

        for i in range(6):
            
            var_t_rt60=np.array([vari_rt60[i]]*128)
            var_t_abs=np.array([vari_abs[i]]*128)
            var_f_rt60=torch.from_numpy(var_t_rt60).float().to(device="cuda")
            var_f_abs=torch.from_numpy(var_t_abs).float().to(device="cuda")


            rt60_loss += NLLloss(rt60[:, i], mean[:, 2 + i], var_f_rt60)
            absorp_loss += NLLloss(absorption[:, i], mean[:, 8 + i], var_f_abs)

        # rt60_loss=rt60_loss/6
        # absorp_loss=absorp_loss/6

        # val_loss_t=(rt60_loss+absorp_loss+loss_surface+loss_volume)/4

        val_loss_t = (loss_surface + loss_volume + rt60_loss + absorp_loss) / 14

        val_loss = float(val_loss_t.item()) + val_loss

        surface_loss_mse_t += pred(mean[:, 0] , surface.view(128))
        volume_loss_mse_t += pred(mean[:, 1] , volume.view(128))

        rt60_loss_mse_125_t += pred(mean[:, 2], rt60[:, 0])
        rt60_loss_mse_250_t += pred(mean[:, 3], rt60[:, 1])
        rt60_loss_mse_500_t += pred(mean[:, 4], rt60[:, 2])
        rt60_loss_mse_1000_t += pred(mean[:, 5], rt60[:, 3])
        rt60_loss_mse_2000_t += pred(mean[:, 6], rt60[:, 4])
        rt60_loss_mse_4000_t += pred(mean[:, 7], rt60[:, 5])
        absorp_loss_mse_125_t += pred(mean[:, 8], absorption[:, 0])
        absorp_loss_mse_250_t += pred(mean[:, 9], absorption[:, 1])
        absorp_loss_mse_500_t += pred(mean[:, 10], absorption[:, 2])
        absorp_loss_mse_1000_t += pred(mean[:, 11], absorption[:, 3])
        absorp_loss_mse_2000_t += pred(mean[:, 12], absorption[:, 4])
        absorp_loss_mse_4000_t += pred(mean[:, 13], absorption[:, 5])

        target_ = torch.cat((surface, volume, rt60, absorption, rm.reshape(128, 1), vp.reshape(128, 1)),
                            axis=1)  # absorption , rt60
        #output = torch.cat((mean, variance), axis=1)
        output=mean
        ass = np.concatenate((output.detach().cpu().clone().numpy().reshape(128, 14),
                              target_.detach().cpu().clone().numpy().reshape(128, 16)), axis=1)

        local_data_sp = np.concatenate((local_data_sp, ass), axis=0)

        volume_loss_mse = volume_loss_mse + float(volume_loss_mse_t.item())
        surface_loss_mse = surface_loss_mse + float(surface_loss_mse_t.item())

        rt60_loss_mse_125 = rt60_loss_mse_125 + float(rt60_loss_mse_125_t.item())
        rt60_loss_mse_250 = rt60_loss_mse_250 + float(rt60_loss_mse_250_t.item())
        rt60_loss_mse_500 = rt60_loss_mse_500 + float(rt60_loss_mse_500_t.item())
        rt60_loss_mse_1000 = rt60_loss_mse_1000 + float(rt60_loss_mse_1000_t.item())
        rt60_loss_mse_2000 = rt60_loss_mse_2000 + float(rt60_loss_mse_2000_t.item())
        rt60_loss_mse_4000 = rt60_loss_mse_4000 + float(rt60_loss_mse_4000_t.item())

        absorp_loss_mse_125 = absorp_loss_mse_125 + float(absorp_loss_mse_125_t.item())
        absorp_loss_mse_250 = absorp_loss_mse_250 + float(absorp_loss_mse_250_t.item())
        absorp_loss_mse_500 = absorp_loss_mse_500 + float(absorp_loss_mse_500_t.item())
        absorp_loss_mse_1000 = absorp_loss_mse_1000 + float(absorp_loss_mse_1000_t.item())
        absorp_loss_mse_2000 = absorp_loss_mse_2000 + float(absorp_loss_mse_2000_t.item())
        absorp_loss_mse_4000 = absorp_loss_mse_4000 + float(absorp_loss_mse_4000_t.item())

        del val_loss_t, volume_loss_mse_t, surface_loss_mse_t, rt60_loss_mse_125_t, rt60_loss_mse_250_t, rt60_loss_mse_500_t, rt60_loss_mse_1000_t, rt60_loss_mse_2000_t, rt60_loss_mse_4000_t, absorp_loss_mse_125_t, absorp_loss_mse_250_t, absorp_loss_mse_500_t, absorp_loss_mse_1000_t, absorp_loss_mse_2000_t, absorp_loss_mse_4000_t
        # del val_loss_t,volume_loss_mse_t,surface_loss_mse_t

        if batch_idx % 50 == 49:
            surface_acc = (surface_loss_mse / 50)

            volume_acc = (volume_loss_mse / 50)

            rt60_125_acc = (rt60_loss_mse_125 / 50)

            rt60_250_acc = (rt60_loss_mse_250 / 50)

            rt60_500_acc = (rt60_loss_mse_500 / 50)
            rt60_1000_acc = (rt60_loss_mse_1000 / 50)
            rt60_2000_acc = (rt60_loss_mse_2000 / 50)
            rt60_4000_acc = (rt60_loss_mse_4000 / 50)

            absorp_125_acc = (absorp_loss_mse_125 / 50)

            absorp_250_acc = (absorp_loss_mse_250 / 50)

            absorp_500_acc = (absorp_loss_mse_500 / 50)
            absorp_1000_acc = (absorp_loss_mse_1000 / 50)
            absorp_2000_acc = (absorp_loss_mse_2000 / 50)
            absorp_4000_acc = (absorp_loss_mse_4000 / 50)

            acc_data_ar.append((surface_acc, volume_acc, rt60_125_acc, rt60_250_acc, rt60_500_acc, rt60_1000_acc,
                                rt60_2000_acc, rt60_4000_acc, absorp_125_acc, absorp_250_acc, absorp_500_acc,
                                absorp_1000_acc, absorp_2000_acc, absorp_4000_acc))

            # acc_data_ar.append((surface_acc,volume_acc))
            surface_loss_mse = 0
            volume_loss_mse = 0

            rt60_loss_mse_125 = 0
            rt60_loss_mse_250 = 0
            rt60_loss_mse_500 = 0
            rt60_loss_mse_1000 = 0
            rt60_loss_mse_2000 = 0
            rt60_loss_mse_4000 = 0

            absorp_loss_mse_125 = 0
            absorp_loss_mse_250 = 0
            absorp_loss_mse_500 = 0
            absorp_loss_mse_1000 = 0
            absorp_loss_mse_2000 = 0
            absorp_loss_mse_4000 = 0

            del surface_acc, volume_acc, rt60_125_acc, rt60_250_acc, rt60_500_acc, rt60_1000_acc, rt60_2000_acc, rt60_4000_acc, absorp_125_acc, absorp_250_acc, absorp_500_acc, absorp_1000_acc, absorp_2000_acc, absorp_4000_acc
            # del surface_acc,volume_acc

    val_data_ar.append((val_loss / batch_idx))

    return val_data_ar, acc_data_ar, local_data_sp
    print(val_data_ar, acc_data_ar)


class Model_1(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.batch_size = 128

        self.conv1d_down_1_depth = nn.Conv1d(769, 769, kernel_size=10, stride=1, groups=769, padding=0)
        self.conv1d_down_1_point = nn.Conv1d(769, 384, kernel_size=1, stride=1, padding=0)
        self.bn_1 = nn.LayerNorm([384, 54])

        self.relu=nn.ReLU()


        self.conv1d_down_2_depth = nn.Conv1d(384, 384, kernel_size=10, stride=1, groups=384, dilation=2, padding=0)
        self.conv1d_down_2_point = nn.Conv1d(384, 192, kernel_size=1, stride=1)
        self.bn_2 = nn.LayerNorm([192, 36])



        self.conv1d_down_3_depth = nn.Conv1d(192, 192, kernel_size=2, stride=1, groups=192, dilation=4, padding=0)
        self.conv1d_down_3_point = nn.Conv1d(192, 96, kernel_size=1, stride=1)
        self.bn_3 = nn.LayerNorm([96, 32])



        self.drp_1=nn.Dropout(p=0.2)

        self.drp = nn.Dropout(p=0.5)



        self.fc = nn.Linear(384, 28)

    def forward(self, x):
        # print(ch1.shape)

        x = self.relu(self.conv1d_down_1_depth(x))

        x = self.bn_1(self.relu(self.conv1d_down_1_point(x)))



        x = self.drp_1(x)



        x = self.relu(self.conv1d_down_2_depth(x))

        x = self.bn_2(self.relu(self.conv1d_down_2_point(x)))



        x = self.drp_1(x)



        x = self.relu(self.conv1d_down_3_depth(x))

        x = self.bn_3(self.relu(self.conv1d_down_3_point(x)))
        #print(x.shape)

        '''
        x = torch.cat((x, x2), axis=1)
        #print(x.shape)
        x= self.avgpool(x)
        #print(x.shape)
        x = self.drp(x)
        x = self.fc(x.reshape(self.batch_size, -1))
        mean,variance = x[:,:14],x[:,14:] #:14,14:
        variance=self.softplus(variance)
        return (mean+10e-7),(variance+10e-7)
        '''

        return x

class Model_2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.pip_conv_1d = nn.Conv1d(2307, 2307, kernel_size=(10), stride=1, groups=2307, padding=0)
        self.pip_conv_1p = nn.Conv1d(2307, 1152, kernel_size=(1), stride=1, padding=0)
        self.bn_pip_1 = nn.LayerNorm([1152, 54])

        self.relu = nn.ReLU()

        self.pip_conv_2d = nn.Conv1d(1152, 1152, kernel_size=10, stride=1, groups=1152, dilation=2, padding=0)
        self.pip_conv_2p = nn.Conv1d(1152, 576, kernel_size=(1), stride=1, padding=0)
        self.bn_pip_2 = nn.LayerNorm([576, 36])

        self.pip_conv_3d = nn.Conv1d(576, 576, kernel_size=2, stride=1, groups=576, dilation=4, padding=0)
        self.pip_conv_3p = nn.Conv1d(576, 288, kernel_size=1, stride=1, padding=0)
        self.bn_pip_3 = nn.LayerNorm([288, 32])

        self.drp_1 = nn.Dropout(p=0.2)
    def forward(self,x2):

        x2 = self.bn_pip_1(self.relu(self.pip_conv_1p(self.relu(self.pip_conv_1d(x2)))))
        
    
        x2 = self.drp_1(x2)

        x2 = self.bn_pip_2(self.relu(self.pip_conv_2p(self.relu(self.pip_conv_2d(x2)))))

        
        x2 = self.drp_1(x2)

        
        x2 = self.bn_pip_3(self.relu(self.pip_conv_3p(self.relu(self.pip_conv_3d(x2)))))

    
        return x2

class Model_3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.drp = nn.Dropout(p=0.5)
        self.fc_1 = nn.Linear(384, 96)
        self.fc_2 = nn.Linear(96, 48)
        self.fc_3 = nn.Linear(48, 14)
        self.softplus = nn.Softplus()

    def forward(self, x):
        
        x = self.fc_1(x)
        
        x = self.drp(x)
        x = self.fc_3(self.fc_2(x))
        
        #mean, variance = x[:, :14], x[:, 14:]
        
        #variance = self.softplus(variance)
        return x
        #return mean, (variance + 10e-7)


class Ensemble(torch.nn.Module):
    def __init__(self, model1, model2,model3):
        super().__init__()
        self.batch_size = 128
        self.model_a = model1
        self.model_b = model2
        self.model_c = model3
        self.avgpool = nn.AvgPool1d(32, stride=1)

    def forward(self, ch1, ch2):
        x = self.model_a(ch1)
        x2 = self.model_b(ch2)
        # print(x.shape,x2.shape)
        x = torch.cat((x, x2), axis=1)
        #print(x.shape)
        x = self.avgpool(x)
        #print(x.shape)
        x = x.reshape(self.batch_size, -1)
        
        mean = self.model_c(x)

        return mean


train_data = dl.binuaral_dataset('train_random_ar.npy','/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_lwh_.hdf5','/home/psrivastava/baseline/sofamyroom_2/conf_room_setup.yml')
val_data = dl.binuaral_dataset('val_random_ar.npy','/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_lwh_.hdf5',
                               '/home/psrivastava/baseline/sofamyroom_2/conf_room_setup.yml')

train_dl = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
val_dl = DataLoader(val_data, batch_size=128, shuffle=True, num_workers=0, drop_last=True)

net_1 = Model_1().to(torch.device("cuda"))
net_2 = Model_2().to(torch.device("cuda"))
net_3 = Model_3().to(torch.device("cuda"))

net = Ensemble(net_1, net_2, net_3).to(torch.device("cuda"))

# net=Model()

optimizer = optim.Adam(net.parameters(), lr=0.0001)
#a1,b1=cal_features(torch.randn(128,1,48000),torch.randn(128,1,48000))
#net(a1,b1)


ar_loss = []
batch_loss_ar = []
total_batch_idx = 0
val_data_ar = []
acc_data_ar = []
save_best_val = 0
#adcc = np.zeros((1, 14))
#track_var = np.zeros((1, 14))
local_dt_sp = np.zeros((1, 30))

path="/home/psrivastava/baseline/scripts/pre_processing/results_fivar_exp/"
for epoch in range(100):
    ar_loss, batch_loss_ar = train(net, train_dl, optimizer, epoch, ar_loss, batch_loss_ar)
    val_data_ar, acc_data_ar, local_dt_sp = val(net, val_dl, optimizer, epoch, val_data_ar, acc_data_ar)

    np.save(path+"fixed_var_ar_loss.npy", ar_loss)
    np.save(path+"fixed_var_batch_loss_ar.npy", batch_loss_ar)
    np.save(path+"fixed_var_val_data_ar.npy", val_data_ar)
    np.save(path+"fixed_var_data_ar.npy", acc_data_ar)
    #np.save("mlh_bnf_track_" + str(epoch) + "_var_.npy", track_var)

    # save best model
    if epoch == 0:
        save_best_val = val_data_ar[-1]
        #np.save("fixed_var_dummy_input_mean_sh.npy", adcc)
    elif save_best_val > val_data_ar[-1]:
        torch.save(
            {'model_dict_1': net_1.state_dict(), 'model_dict_2': net_2.state_dict(), 'model_dict_3':net_3.state_dict(),'model_dict_ens': net.state_dict(),
             'optimizer_dic': optimizer.state_dict(), 'epoch': epoch, 'loss': val_data_ar[-1]},
            path+"fixed_var_tas_save_best_sh.pt")
        save_best_val = val_data_ar[-1]
        np.save(path+"fixed_var_bnf_mag_96ms_" + str(epoch) + ".npy", local_dt_sp)

