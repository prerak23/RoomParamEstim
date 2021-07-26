import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme()
abc=np.load("/home/psrivastava/baseline/scripts/pre_processing/mlh_acc_data_ar.npy")
bc=np.load("/home/psrivastava/baseline/scripts/pre_processing/fixed_var_data_ar.npy")


fig,axs=plt.subplots(6,1,figsize=(8,22))
len_=abc.shape[0]-1
len_1=bc.shape[0]-1

axs[0].plot(np.arange(len_),abc[1:,0],color='cyan',label='surface')
axs[0].plot(np.arange(len_),abc[1:,1],color='green',label='volume')
axs[0].set_ylabel("MSE Error Mean Vs Target")
axs[0].set_xlabel("1 Point Per 50 Batches")
axs[0].legend()

axs[1].plot(np.arange(len_1),bc[1:,0],color='cyan',label='surface')
axs[1].plot(np.arange(len_1),bc[1:,1],color='green',label='volume')
axs[1].set_ylabel("MSE Error Mean Vs Target")
axs[1].set_xlabel("1 Point Per 50 Batches")
axs[1].set_title("Fixed Variance")
axs[1].legend()


axs[2].plot(np.arange(len_),abc[1:,2],color='cyan',label='rt 125')
axs[2].plot(np.arange(len_),abc[1:,3],color='green',label='rt 250')
axs[2].plot(np.arange(len_),abc[1:,4],color='red',label='rt 500')
axs[2].plot(np.arange(len_),abc[1:,5],color='blue',label='rt 1000')
axs[2].plot(np.arange(len_),abc[1:,6],color='orange',label='rt 2000')
axs[2].plot(np.arange(len_),abc[1:,7],color='yellow',label='rt 4000')
axs[2].set_title("Rt 60 mse Dev set")


axs[2].set_ylabel("MSE Error Mean Vs Target")
axs[2].set_xlabel("1 Point Per 50 Batches")
axs[2].legend()

axs[3].plot(np.arange(len_1),bc[1:,2],color='cyan',label='rt 125')
axs[3].plot(np.arange(len_1),bc[1:,3],color='green',label='rt 250')
axs[3].plot(np.arange(len_1),bc[1:,4],color='red',label='rt 500')
axs[3].plot(np.arange(len_1),bc[1:,5],color='blue',label='rt 1000')
axs[3].plot(np.arange(len_1),bc[1:,6],color='orange',label='rt 2000')
axs[3].plot(np.arange(len_1),bc[1:,7],color='yellow',label='rt 4000')
axs[3].set_title("Rt 60 mse Dev set Fixed Variance")


axs[3].set_ylabel("MSE Error Mean Vs Target")
axs[3].set_xlabel("1 Point Per 50 Batches")
axs[3].legend()


axs[4].plot(np.arange(len_),abc[1:,8],color='cyan',label='ab 125')
axs[4].plot(np.arange(len_),abc[1:,9],color='green',label='ab 250')
axs[4].plot(np.arange(len_),abc[1:,10],color='red',label='ab 500')
axs[4].plot(np.arange(len_),abc[1:,11],color='blue',label='ab 1000')
axs[4].plot(np.arange(len_),abc[1:,12],color='orange',label='ab 2000')
axs[4].plot(np.arange(len_),abc[1:,13],color='yellow',label='ab 4000')
axs[4].set_title("Ab Coeff mse Dev set")
axs[4].legend()

axs[4].set_ylabel("MSE Error Mean Vs Target")
axs[4].set_xlabel("1 Point Per 50 Batches")

axs[5].plot(np.arange(len_1),bc[1:,8],color='cyan',label='ab 125')
axs[5].plot(np.arange(len_1),bc[1:,9],color='green',label='ab 250')
axs[5].plot(np.arange(len_1),bc[1:,10],color='red',label='ab 500')
axs[5].plot(np.arange(len_1),bc[1:,11],color='blue',label='ab 1000')
axs[5].plot(np.arange(len_1),bc[1:,12],color='orange',label='ab 2000')
axs[5].plot(np.arange(len_1),bc[1:,13],color='yellow',label='ab 4000')
axs[5].set_title("Ab Coeff mse Dev set Fixed Variance")
axs[5].legend()

axs[5].set_ylabel("MSE Error Mean Vs Target")
axs[5].set_xlabel("1 Point Per 50 Batches")


fig.tight_layout(pad=3.0)

plt.savefig("ff_mse.png")


