import numpy as np
import torch
import h5py 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
#The SNR of the final noisy mixture dataset used for training the deep learning model's are calculated in a wrong way , hence to assure that the Snr level is not less the -20dB in the final mixtures. We made this script to deduce actual SNR from the wrong calculated ones saved in the hdf file below.
#We found out that SNR lie between -10dB to 60dB which is good in a way so at the end we did'nt have to generate the noisy mixtures again.

abc=h5py.File("/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_lwh_.hdf5","r")
new_snr=h5py.File("/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/snr_histogram_check.hdf5","w")
room_save=new_snr.create_group("room_nos")
vp_nos=[1,2,3,4,5]
room_nos=[i+1 for i in range(20000)]
adc=[]
acd=[]
akc=[]

for i in range(19999):
    room=room_nos[i]
    room_id=room_save.create_group("room_"+str(room))
    print(room)
    for j in vp_nos:
        snr_w_f=abc["room_nos"]["room_"+str(room)]["nsmix_snr_f"][()][j-1]
        snr_w_d=abc["room_nos"]["room_"+str(room)]["nsmix_snr_diff"][()][j-1]
        snr_w_s=abc["room_nos"]["room_"+str(room)]["nsmix_snr_static"][()][j-1]
        a_=10**(snr_w_s/10)
        b_=10**(snr_w_d/10)
        c_=10**(snr_w_f/10)
        snr_static=10*np.log10(a_*(1-(1/c_)))
        snr_diff=10*np.log10(b_*(1-(1/c_)))
        snr_all=10*np.log10(c_-1)
        adc.append(snr_static)
        acd.append(snr_diff)
        akc.append(snr_all)
    room_id.create_dataset("nsmix_snr_f",5,data=akc[-5:])
    room_id.create_dataset("nsmix_snr_diff",5,data=acd[-5:])
    room_id.create_dataset("nsmix_snr_static",5,data=adc[-5:])


#Plotting histograms 

fig,ax2=plt.subplots()
ax2.hist(akc)
ax2.set_ylabel("No of Samples")
ax2.set_xlabel("Negative log of variance")
fig.tight_layout()
plt.savefig("histf_correct_cal_from_old.jpeg")

fig,ax2=plt.subplots()
ax2.hist(acd)
ax2.set_ylabel("No of Samples")
ax2.set_xlabel("SNR Diff")
fig.tight_layout()
plt.savefig("histd_correct_cal_from_old.jpeg")


fig,ax2=plt.subplots()
ax2.hist(adc)
ax2.set_ylabel("No of Samples")
ax2.set_xlabel("SNR Static")
fig.tight_layout()
plt.savefig("hists_correct_cal_from_old.jpeg")


