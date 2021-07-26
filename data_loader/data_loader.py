#Data loader for pytorch architecture 

import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import h5py
import yaml
from asteroid.filterbanks import STFTFB
from asteroid.filterbanks.enc_dec import Filterbank, Encoder, Decoder


abcd=h5py.File('/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_lwh_.hdf5','r') #Open the noisy mixture file 
rt60_file=h5py.File('rt60_anno_room_20_median.hdf5','r') #Open the annotation file 
anno_file=h5py.File('absorption_surface_calcul.hdf5','r') #Open the annotation file 

class binuaral_dataset(Dataset):
    def __init__(self,randomized_arr,root_dir_noisy_mix,root_dir_labels):

        self.random_arr=np.load(randomized_arr) #Randomize the room_number and view points , therefore [(Room_0,vp_1),(Room_2,vp_4),(Room_3,vp_5),(Room_2,vp_3)]
        #self.nsmix_file=h5py.File(root_dir_noisy_mix,"r")
        #self.nsmix_file=abcd
        #f=open(root_dir_labels)
        #self.root_labels=yaml.load(f,Loader=yaml.FullLoader)
    def __len__(self):
        return len(self.random_arr)

    def __getitem__(self, item):
        item=self.random_arr[item] #Get room number 
        #print(item)
        vp=int(item[1]) #Get view-point 
        bn_sample_vp_ch1=abcd['room_nos'][item[0]]['nsmix_f'][(vp-1)*2,:] 
        bn_sample_vp_ch2=abcd['room_nos'][item[0]]['nsmix_f'][((vp*2)-1),:]
        absorp=anno_file['room_nos'][item[0]]['absorption'][()]
        surface=anno_file['room_nos'][item[0]]['surface_area'][0]
        volume=anno_file['room_nos'][item[0]]['volume'][0]
        rt60=rt60_file['room_nos'][item[0]]['rt60'][()].reshape(6)

        #dimen=np.array(self.root_labels[item[0]]['dimension'])
        #diffusion=anno_file['room_nos'][item[0]]['diff']
        sample={'bnsample':np.vstack([bn_sample_vp_ch1,bn_sample_vp_ch2]),'surface':surface,'volume':volume,'absorption':absorp,'rt60':rt60,'room':int(item[0].split("_")[1]),'vp':vp}
        return sample

#bn_dataset=binuaral_dataset('train_random_ar.npy','/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_lwh.hdf5','/home/psrivastava/baseline/sofamyroom_2/conf_room_setup.yml')

#dataloader = DataLoader(bn_dataset, batch_size=4,
#                        shuffle=True, num_workers=0)


'''
dft_filters= STFTFB(n_filters=512, kernel_size=256 , stride=128, sample_rate=16000)
stft=Encoder(dft_filters)
'''
'''
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['bnsample'].size(),
          sample_batched['absorption'].size(),sample_batched['rt60'].size())
    if i_batch == 3:
        print(sample_batched['bnsample'].size())
        print(sample_batched['bnsample'][0,1,:].size())
        #print(stft(sample_batched['bnsample'][0,1,:].float()).shape)
        print(sample_batched['volume'].size())
        print(sample_batched['surface'].size())
        print("rt60",sample_batched['rt60'].size())
        print("absorption",sample_batched['absorption'].size())
        break
'''
