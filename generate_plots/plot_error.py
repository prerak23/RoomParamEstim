import numpy as np
import seaborn as sns
import h5py 
import matplotlib.pyplot as plt


abc=np.load("mlh_bnf_mag_96ms_95.npy")
snr_file=h5py.File("/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_lwh_.hdf5","r")

m1=abc[1:,0]
t1=abc[1:,28]
m2=abc[1:,1]
t2=abc[1:,29]
'''
m3=abc[1:,10]
t3=abc[1:,38]
m4=abc[1:,11]
t4=abc[1:,39]
m5=abc[1:,12]
t5=abc[1:,40]
m6=abc[1:,13]
t6=abc[1:,41]
'''





room_no=abc[1:,42]
print(room_no)
vp_no=abc[1:,43]
snr_=[]
err_=[]
snr_1=[]
err_1=[]


snr_2=[]
err_2=[]
'''
snr_3=[]
err_3=[]
snr_4=[]
err_4=[]
snr_5=[]
err_5=[]
snr_6=[]
err_6=[]
'''







for a in range(len(m1)):
    err_1.append((m1[a]-t1[a])**2)
    err_2.append((m2[a]-t2[a])**2)
    #err_3.append((m3[a]-t3[a])**2)
    #err_4.append((m4[a]-t4[a])**2)
    #err_5.append((m5[a]-t5[a])**2)
    #err_6.append((m6[a]-t6[a])**2)

    
    room="room_"+str(int(room_no[a]))
    
    snr=snr_file["room_nos"][room]["nsmix_snr_f"][int(vp_no[a])-1]
    snr_.append(snr)


sns.set_theme()
fig,axs=plt.subplots(2,1,figsize=(15,10))

axs[0].scatter(err_1,snr_)
axs[0].set_ylabel("SNR dB")
axs[0].set_xlabel("Error M2")
axs[0].set_title("Surface")


axs[1].scatter(err_2,snr_)
axs[1].set_ylabel("SNR dB")
axs[1].set_xlabel("Error M3")
axs[1].set_title("Volume")

'''
axs[0,1].scatter(err_3,snr_)
axs[0,1].set_ylabel("SNR dB")
axs[0,1].set_xlabel("MSE Mean And Target (Absorption)")
axs[0,1].set_title("500hz")

axs[0,2].scatter(err_4,snr_)
axs[0,2].set_ylabel("SNR dB")
axs[0,2].set_xlabel("MSE Mean And Target (Absorption)")
axs[0,2].set_title("1000hz")

axs[1,1].scatter(err_5,snr_)
axs[1,1].set_ylabel("SNR dB")
axs[1,1].set_xlabel("MSE Mean And Target (Absorption)")
axs[1,1].set_title("2000hz")

axs[1,2].scatter(err_6,snr_)
axs[1,2].set_ylabel("SNR dB")
axs[1,2].set_xlabel("MSE Mean And Target (Absorption)")
axs[1,2].set_title("4000hz")
'''


fig.tight_layout(pad=4.0)
plt.savefig("snr_mse_surface_vol.png")





