import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
file_name="/home/psrivastava/baseline/scripts/pre_processing/mlh_bnf_track_"
file_name_last="_var_.npy"
#kr=[]
#volume=[]

rt_125_=[]
rt_250_=[]
rt_500_=[]
rt_1000_=[]
rt_2000_=[]
rt_4000_=[]
ab_125_=[]
ab_250_=[]
ab_500_=[]
ab_1000_=[]
ab_2000_=[]
ab_4000_=[]
s_=[]
v_=[]
std_surf=84.224076
std_vol=106.022657

for i in range(103):
    abc=np.mean(np.load(file_name+str(i)+file_name_last)[:,2])
    bc=np.mean(np.load(file_name+str(i)+file_name_last)[:,3])
    c=np.mean(np.load(file_name+str(i)+file_name_last)[:,4])
    d=np.mean(np.load(file_name+str(i)+file_name_last)[:,5])
    e=np.mean(np.load(file_name+str(i)+file_name_last)[:,6])
    f=np.mean(np.load(file_name+str(i)+file_name_last)[:,7])
    abc_2=np.mean(np.load(file_name+str(i)+file_name_last)[:,8])
    bc_2=np.mean(np.load(file_name+str(i)+file_name_last)[:,9])
    c_2=np.mean(np.load(file_name+str(i)+file_name_last)[:,10])
    d_2=np.mean(np.load(file_name+str(i)+file_name_last)[:,11])
    e_2=np.mean(np.load(file_name+str(i)+file_name_last)[:,12])
    f_2=np.mean(np.load(file_name+str(i)+file_name_last)[:,13])
    s=np.mean(np.load(file_name+str(i)+file_name_last)[:,0]*std_surf)
    v=np.mean(np.load(file_name+str(i)+file_name_last)[:,1]*std_vol)
    #kr.append(abc)
    #volume.append(bc)
    rt_125_.append(abc)
    rt_250_.append(bc)
    rt_500_.append(c)
    rt_1000_.append(d)
    rt_2000_.append(e)
    rt_4000_.append(f)
    ab_125_.append(abc_2)
    ab_250_.append(bc_2)
    ab_500_.append(c_2)
    ab_1000_.append(d_2)
    ab_2000_.append(e_2)
    ab_4000_.append(f_2)
    s_.append(s)
    v_.append(v)


fig,axs=plt.subplots(3,1,figsize=(8,15))
ep=103
axs[0].plot(np.arange(ep),rt_125_,color='orange',label='rt 125')
axs[0].plot(np.arange(ep),rt_250_,color='green',label='rt 250')
axs[0].plot(np.arange(ep),rt_500_,color='red',label='rt 500')
axs[0].plot(np.arange(ep),rt_1000_,color='yellow',label='rt 1000')
axs[0].plot(np.arange(ep),rt_2000_,color='cyan',label='rt 2000')
axs[0].plot(np.arange(ep),rt_4000_,color='black',label='rt 4000')
axs[0].set_title("Rt60 Variance")
axs[0].set_ylabel("Seconds")
axs[0].legend()

axs[1].plot(np.arange(ep),ab_125_,color='orange',label='ab 125')
axs[1].plot(np.arange(ep),ab_250_,color='green',label='ab 250')
axs[1].plot(np.arange(ep),ab_500_,color='red',label='ab 500')
axs[1].plot(np.arange(ep),ab_1000_,color='yellow',label='ab 1000')
axs[1].plot(np.arange(ep),ab_2000_,color='cyan',label='ab 2000')
axs[1].plot(np.arange(ep),ab_4000_,color='black',label='ab 4000')
axs[1].set_title("Absorption Variance")
#axs[1].set_ylim(0.0,0.05)
axs[1].legend()

axs[2].plot(np.arange(ep),s_,color='orange',label='Surface Area')
axs[2].plot(np.arange(ep),v_,color='green',label='Volume')
axs[2].set_title("Surface Area And Volume Variance")
axs[2].legend()

#plt.plot(np.arange(50),kr,color='orange',label='surface')
#plt.plot(np.arange(50),volume,color='green',label='volume')
#plt.xlabel("epochs")
#plt.ylabel("Mean of Varaiance on train")
fig.tight_layout(pad=3.0)
#fig.legend()
plt.savefig("variance_comparasion_bnf_3ff.png")



