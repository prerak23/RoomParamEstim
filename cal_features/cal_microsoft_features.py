#Calculate state of the art , microsoft feature set . Used for comparasion with our script.
import numpy as np
import h5py 
import gammatone.gtgram as gt
from scipy.signal import butter, lfilter

#train_file=np.load("train_random_ar_2.npy")
#val_file=np.load("val_random_ar_2.npy")

file_h5py=h5py.File("/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_lwh_2.hdf5","r") #Open noisy mixture file 
new_file_feat=h5py.File("/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/microsoft_feat.hdf5","w")


#Low pass filter 
def lp_filter(ch1_,fs,cutoff,order):
    nyq=0.5*fs
    normal_cutoff=cutoff/nyq
    b,a=butter(order,normal_cutoff,btype="low",analog=False)
    y=lfilter(b,a,ch1_)
    #print("low-pass filter",y.shape)
    return y[:1499]
#Keeps the same number of features as the rest

def cal_features(ch1):
    #channel_1,sampling_freq,window_size in ms,hop_size in ms ,channels,f_min

    feat_gt_=gt.gtgram(ch1,16000,0.004,0.002,58,20)[:20,:]
    
    #Mag of STFT
    feat_dft=np.abs(np.fft.fft(ch1,n=48000)[:1499])
    #Sorted Mag of STFT 
    sort_feat_dft=np.sort(feat_dft)
    lp=lp_filter(ch1,16000,500,10)
    #Cepstrum: inverse fft of log(Mag of STFT)
    cepstrum=np.fft.ifft(np.log(np.abs(np.fft.fft(ch1,n=48000))))[:1499]
    
    #Concatenate Everything
    cat_all=np.concatenate((feat_gt_,feat_dft.reshape(1,1499),sort_feat_dft.reshape(1,1499),lp.reshape(1,1499),cepstrum.reshape(1,1499)),axis=0)

    return cat_all 
        


room_save=new_file_feat.create_group("room_nos")
room_ll=np.arange(17999)+1 #Create a list of rooms , just so that we can iterate it starts from [Room_1,....Room_18000] 
vps=np.arange(5)+1 #Create a list of view points , just so that we can iterate it [Vp_1,....Vp_5]
for x in room_ll:
    k=np.zeros((1,24,1499))
    print("room_no",x)
    for vp in vps:
        #print("vp--------")
        nsmix_=file_h5py['room_nos']['room_'+str(x)]['nsmix_f'][(vp-1)*2,:] #Get the mixture from the file (vp-1)*2 : get's one of the channel from that view-point.
        tot_feat=cal_features(nsmix_)
        
        k=np.concatenate((k,tot_feat.reshape(1,24,1499)),axis=0) #Concatenate in the axis=0, so that we can get (5,24,1499)
    room_id=room_save.create_group('room_'+str(x))
    #5 vp's/room 
    room_id.create_dataset('nsmix_f',(5,24,1499),data=k[1:,:,:])
