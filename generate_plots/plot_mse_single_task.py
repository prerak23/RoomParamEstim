import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme()
abc=np.load("/home/psrivastava/baseline/scripts/pre_processing/rt60_acc_data_ar.npy")
bcd=np.load("/home/psrivastava/baseline/scripts/pre_processing/rt60_val_data_ar.npy")
kcd=np.load("/home/psrivastava/baseline/scripts/pre_processing/rt60_ar_loss.npy")

bc=np.load("/home/psrivastava/baseline/scripts/pre_processing/abs_acc_data_ar.npy")
cd=np.load("/home/psrivastava/baseline/scripts/pre_processing/abs_val_data_ar.npy")
kd=np.load("/home/psrivastava/baseline/scripts/pre_processing/abs_ar_loss.npy")

surf_val=np.load("/home/psrivastava/baseline/scripts/pre_processing/surface_val_data_ar.npy")
surf_ar=np.load("/home/psrivastava/baseline/scripts/pre_processing/surface_ar_loss.npy")
surf_acc=np.load("/home/psrivastava/baseline/scripts/pre_processing/surface_acc_data_ar.npy")

vol_val=np.load("/home/psrivastava/baseline/scripts/pre_processing/volume_val_data_ar.npy")
vol_ar=np.load("/home/psrivastava/baseline/scripts/pre_processing/volume_ar_loss.npy")
vol_acc=np.load("/home/psrivastava/baseline/scripts/pre_processing/volume_acc_data_ar.npy")



fig,axs=plt.subplots(6,1,figsize=(8,20))
#len_=abc.shape[0]
len_=50
axs[0].plot(np.arange(100),bcd,color='blue',label='Validation')
axs[0].plot(np.arange(100),kcd,color='green',label='Training')
axs[0].set_ylabel("Avg MSE Error")
axs[0].set_xlabel("Epochs")
axs[0].set_title("Training Plots RT60 Single Task")
axs[0].legend()

axs[1].plot(np.arange(50),cd,color='blue',label='Validation')
axs[1].plot(np.arange(50),kd,color='green',label='Training')
axs[1].set_ylabel("Avg MSE Error")
axs[1].set_xlabel("Epochs")
axs[1].set_title("Training Plots Ab Coeff Single Task")
axs[1].legend()

axs[2].plot(np.arange(50),surf_val,color='blue',label='Validation')
axs[2].plot(np.arange(50),surf_ar,color='green',label='Training')
axs[2].set_ylabel("Avg MSE Error")
axs[2].set_xlabel("Epochs")
axs[2].set_title("Training Plots Surface Single Task")
axs[2].legend()

axs[3].plot(np.arange(50),vol_val,color='blue',label='Validation')
axs[3].plot(np.arange(50),vol_ar,color='green',label='Training')
axs[3].set_ylabel("Avg MSE Error")
axs[3].set_xlabel("Epochs")
axs[3].set_title("Training Plots Volume Single Task")
axs[3].legend()



'''
axs[1].plot(np.arange(len_),(abc[:,0]*100)/50,color='cyan',label='surface')
axs[1].plot(np.arange(len_),(abc[:,1]*100)/50,color='green',label='volume')
axs[1].set_ylabel("MSE Error Mean Vs Target")
axs[1].set_xlabel("1 Sample Per 50 Batches")
axs[1].legend()
'''
len_1=100
axs[4].plot(np.arange(len_1),abc[:,0],color='cyan',label='rt 125')
axs[4].plot(np.arange(len_1),abc[:,1],color='green',label='rt 250')
axs[4].plot(np.arange(len_1),abc[:,2],color='red',label='rt 500')
axs[4].plot(np.arange(len_1),abc[:,3],color='blue',label='rt 1000')
axs[4].plot(np.arange(len_1),abc[:,4],color='orange',label='rt 2000')
axs[4].plot(np.arange(len_1),abc[:,5],color='yellow',label='rt 4000')
axs[4].set_title("Rt 60 mse Dev set")


axs[4].set_ylabel("MSE Error Mean Vs Target")
axs[4].set_xlabel("1 Sample Per 50 Batches")
axs[4].legend()

axs[5].plot(np.arange(len_),bc[:,0],color='cyan',label='ab 125')
axs[5].plot(np.arange(len_),bc[:,1],color='green',label='ab 250')
axs[5].plot(np.arange(len_),bc[:,2],color='red',label='ab 500')
axs[5].plot(np.arange(len_),bc[:,3],color='blue',label='ab 1000')
axs[5].plot(np.arange(len_),bc[:,4],color='orange',label='ab 2000')
axs[5].plot(np.arange(len_),bc[:,5],color='yellow',label='ab 4000')
axs[5].set_title("Ab Coeff mse Dev set")
axs[5].legend()

axs[5].set_ylabel("MSE Error Mean Vs Target")
axs[5].set_xlabel("1 Sample Per 50 Batches")

fig.tight_layout(pad=3.0)

plt.savefig("mse_singletask.png")



