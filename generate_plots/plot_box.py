import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

'''
abc=np.load("batch_norm_64ms_mag_24.npy")
bcd=np.load("layer_norm_64ms_logmag_24.npy")
kcd=np.load("layer_norm_64ms_sqrtmag_20.npy")
xcd=np.load("layer_norm_64ms_mag_27.npy")
ncd=np.load("layer_norm_32ms_mag_26.npy")
lcd=np.load("boxplot_data_vol_dil_23.npy")
ecd=np.load("boxplot_data_vol_dil_16ms_mag_23.npy")
hcd=np.load("Ixvec_64ms_mag_25.npy")
bnf=np.load("BNF_64ms_sqrtmag_rimagcat_18.npy")
bns=np.load("BNF_64ms_2pipeconv1d_concatfreqaxis_28.npy")
bnr=np.load("BNF_96ms_2pipeconv1d_concatfreqaxis_15.npy")
sks=np.load("BNF_128ms_2pipeconv1d_concatfreqaxis_9.npy")
lsl=np.load("mag_128ms_7.npy")
dcd=np.load("dummy_data_train_net.npy")
'''
'''
bnr=np.load("BNF_96ms_2pipeconv1d_concatfreqaxis_15.npy")
sks=np.load("BNF_128ms_2pipeconv1d_concatfreqaxis_9.npy")
lsl=np.load("mag_128ms_7.npy")
kcd=np.load("layer_norm_64ms_sqrtmag_20.npy")
hcd=np.load("Ixvec_64ms_mag_25.npy")
dcd=np.load("dummy_data_train_net.npy")
'''
sns.set_theme()
path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/"
path_tmp="/home/psrivastava/baseline/scripts/pre_processing/"
dummy_bnf=np.load("/home/psrivastava/baseline/scripts/pre_processing/mlh_dummy_input_mean_sh.npy")
dummy_m=np.load("mlh_nobnf_dummy_input_mean_sh.npy")
#bnf_96=np.load("sh_BNF_96ms_2pipe_12.npy")
#bnf_mag_96=np.load(path+"bn_mlh_/bn_mlh_k2mlh_bnf_mag_96ms_38.npy") #79 real
bnf_mag_96=np.load(path+"bn_mlh_layer_norm/mlh_bnf_mag_96ms_23.npy") 
#m_mag_96=np.load(path+"results_mono_exp/mlhm_bnf_mag_96ms_84.npy") real
m_mag_96=np.load(path+"results_mono_exp_layer_norm/mlhm2_bnf_mag_96ms_83.npy")


mse_mlh=np.load(path+"results_msemt_exp/mse_bnf_mag_96ms_50_11.npy")
rt60_mse=np.load(path+"results_mse_rt60_exp/volume_bnf_mag_96ms_114.npy")
ab_mse=np.load(path+"results_mse_abs_exp/abs_bnf_mag_96ms_33.npy")
surf_data=np.load(path+"results_mse_surface_exp/surface_bnf_mag_96ms_9.npy")
vol_data=np.load(path+"results_mse_vol_exp/volume_bnf_mag_96ms_10.npy")
fi_var=np.load(path+"results_fivar_exp/fixed_var_bnf_mag_96ms_93.npy")
micro_vol=np.load("/home/psrivastava/baseline/scripts/pre_processing/results_microsoft_vol_exp/microsoft_volume_bnf_mag_96ms_98.npy")




dum_bnf_1=np.abs((np.array([np.mean(dummy_bnf[1:,8])]*(bnf_mag_96.shape[0]-1)))-(bnf_mag_96[1:,36]))
#dum_m_1=np.abs((np.array([np.mean(dummy_m[1:,2])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,30]))

dum_bnf_2=np.abs((np.array([np.mean(dummy_bnf[1:,9])]*(bnf_mag_96.shape[0]-1)))-(bnf_mag_96[1:,37]))
#dum_m_2=np.abs((np.array([np.mean(dummy_m[1:,3])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,31]))


dum_bnf_3=np.abs((np.array([np.mean(dummy_bnf[1:,10])]*(bnf_mag_96.shape[0]-1)))-(bnf_mag_96[1:,38]))
#dum_m_3=np.abs((np.array([np.mean(dummy_m[1:,10])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,38]))


dum_bnf_4=np.abs((np.array([np.mean(dummy_bnf[1:,11])]*(bnf_mag_96.shape[0]-1)))-(bnf_mag_96[1:,39]))
#dum_m_4=np.abs((np.array([np.mean(dummy_m[1:,11])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,39]))

dum_bnf_5=np.abs((np.array([np.mean(dummy_bnf[1:,12])]*(bnf_mag_96.shape[0]-1)))-(bnf_mag_96[1:,40]))
#dum_m_5=np.abs((np.array([np.mean(dummy_m[1:,12])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,40]))

dum_bnf_6=np.abs((np.array([np.mean(dummy_bnf[1:,13])]*(bnf_mag_96.shape[0]-1)))-(bnf_mag_96[1:,41]))
k=(np.array(dum_bnf_1).mean()+np.array(dum_bnf_2).mean()+np.array(dum_bnf_3).mean()+np.array(dum_bnf_4).mean()+np.array(dum_bnf_5).mean()+np.array(dum_bnf_6).mean())/6
print(k)
k=(np.array(dum_bnf_1).std()+np.array(dum_bnf_2).std()+np.array(dum_bnf_3).std()+np.array(dum_bnf_4).std()+np.array(dum_bnf_5).std()+np.array(dum_bnf_6).std())/6
print(k)






#dum_m_6=np.abs((np.array([np.mean(dummy_m[1:,13])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,41]))
dum_bnf_7=np.abs((np.array([np.mean(dummy_bnf[1:,2])]*(bnf_mag_96.shape[0]-1)))-(bnf_mag_96[1:,30]))
dum_bnf_8=np.abs((np.array([np.mean(dummy_bnf[1:,3])]*(bnf_mag_96.shape[0]-1)))-(bnf_mag_96[1:,31]))
dum_bnf_9=np.abs((np.array([np.mean(dummy_bnf[1:,4])]*(bnf_mag_96.shape[0]-1)))-(bnf_mag_96[1:,32]))
dum_bnf_10=np.abs((np.array([np.mean(dummy_bnf[1:,5])]*(bnf_mag_96.shape[0]-1)))-(bnf_mag_96[1:,33]))
dum_bnf_11=np.abs((np.array([np.mean(dummy_bnf[1:,6])]*(bnf_mag_96.shape[0]-1)))-(bnf_mag_96[1:,34]))
dum_bnf_12=np.abs((np.array([np.mean(dummy_bnf[1:,7])]*(bnf_mag_96.shape[0]-1)))-(bnf_mag_96[1:,35]))
k=(np.array(dum_bnf_7).mean()+np.array(dum_bnf_8).mean()+np.array(dum_bnf_9).mean()+np.array(dum_bnf_10).mean()+np.array(dum_bnf_11).mean()+np.array(dum_bnf_12).mean())/6
print(k)
k=(np.array(dum_bnf_7).std()+np.array(dum_bnf_8).std()+np.array(dum_bnf_9).std()+np.array(dum_bnf_10).std()+np.array(dum_bnf_11).std()+np.array(dum_bnf_12).std())/6
print(k)




dum_bnf_13=np.abs((np.array([np.mean(dummy_bnf[1:,0])]*(bnf_mag_96.shape[0]-1)))-(bnf_mag_96[1:,28]))
dum_bnf_14=np.abs((np.array([np.mean(dummy_bnf[1:,1])]*(bnf_mag_96.shape[0]-1)))-(bnf_mag_96[1:,29]))
print(np.array(dum_bnf_13).mean(),np.array(dum_bnf_13).std())

print(np.array(dum_bnf_14).mean(),np.array(dum_bnf_14).std())













std_vol=106.02265
std_surf=84.2240762
std_rt60=[ 0.7793691,0.7605436,0.6995225, 0.7076664, 0.6420753,0.51794204]
std_abs=[0.100825,0.1172430,0.1002776,0.09108845,0.09378748,0.091663032]


#bnf_96_s=np.abs((bnf_96[:,1])-(bnf_96[:,3]))

bnf_mag_96_1=np.abs((bnf_mag_96[:,8]*std_abs[0])-(bnf_mag_96[:,36]))

m_mag_96_1=np.abs((m_mag_96[:,8]*std_abs[0])-(m_mag_96[:,36]))

bnf_mag_96_2=np.abs((bnf_mag_96[:,9]*std_abs[1])-(bnf_mag_96[:,37]))
m_mag_96_2=np.abs((m_mag_96[:,9]*std_abs[1])-(m_mag_96[:,37]))



bnf_mag_96_3=np.abs((bnf_mag_96[:,10]*std_abs[2])-(bnf_mag_96[:,38]))
m_mag_96_3=np.abs((m_mag_96[:,10]*std_abs[2])-(m_mag_96[:,38]))

bnf_mag_96_4=np.abs((bnf_mag_96[:,11]*std_abs[3])-(bnf_mag_96[:,39]))
m_mag_96_4=np.abs((m_mag_96[:,11]*std_abs[3])-(m_mag_96[:,39]))

bnf_mag_96_5=np.abs((bnf_mag_96[:,12]*std_abs[4])-(bnf_mag_96[:,40]))
m_mag_96_5=np.abs((m_mag_96[:,12]*std_abs[4])-(m_mag_96[:,40]))

bnf_mag_96_6=np.abs((bnf_mag_96[:,13]*std_abs[5])-(bnf_mag_96[:,41]))
m_mag_96_6=np.abs((m_mag_96[:,13]*std_abs[5])-(m_mag_96[:,41]))

k=(np.array(bnf_mag_96_1).mean()+np.array(bnf_mag_96_2).mean()+np.array(bnf_mag_96_3).mean()+np.array(bnf_mag_96_4).mean()+np.array(bnf_mag_96_5).mean()+np.array(bnf_mag_96_6).mean())/6
print(k)
k=(np.array(bnf_mag_96_1).std()+np.array(bnf_mag_96_2).std()+np.array(bnf_mag_96_3).std()+np.array(bnf_mag_96_4).std()+np.array(bnf_mag_96_5).std()+np.array(bnf_mag_96_6).std())/6
print(k)
print("------")

k=(np.array(m_mag_96_1).mean()+np.array(m_mag_96_2).mean()+np.array(m_mag_96_3).mean()+np.array(m_mag_96_4).mean()+np.array(m_mag_96_5).mean()+np.array(m_mag_96_6).mean())/6
print(k)
k=(np.array(m_mag_96_1).std()+np.array(m_mag_96_2).std()+np.array(m_mag_96_3).std()+np.array(m_mag_96_4).std()+np.array(m_mag_96_5).std()+np.array(m_mag_96_6).std())/6
print(k)


bnf_mag_96_7=np.abs((bnf_mag_96[:,2]*std_rt60[0])-(bnf_mag_96[:,30]))
m_mag_96_7=np.abs((m_mag_96[:,2]*std_rt60[0])-(m_mag_96[:,30]))

bnf_mag_96_8=np.abs((bnf_mag_96[:,3]*std_rt60[1])-(bnf_mag_96[:,31]))
m_mag_96_8=np.abs((m_mag_96[:,3]*std_rt60[1])-(m_mag_96[:,31]))



bnf_mag_96_9=np.abs((bnf_mag_96[:,4]*std_rt60[2])-(bnf_mag_96[:,32]))

m_mag_96_9=np.abs((m_mag_96[:,4]*std_rt60[2])-(m_mag_96[:,32]))

bnf_mag_96_10=np.abs((bnf_mag_96[:,5]*std_rt60[3])-(bnf_mag_96[:,33]))

m_mag_96_10=np.abs((m_mag_96[:,5]*std_rt60[3])-(m_mag_96[:,33]))

bnf_mag_96_11=np.abs((bnf_mag_96[:,6]*std_rt60[4])-(bnf_mag_96[:,34]))

m_mag_96_11=np.abs((m_mag_96[:,6]*std_rt60[4])-(m_mag_96[:,34]))

bnf_mag_96_12=np.abs((bnf_mag_96[:,7]*std_rt60[5])-(bnf_mag_96[:,35]))

m_mag_96_12=np.abs((m_mag_96[:,7]*std_rt60[5])-(m_mag_96[:,35]))

bnf_mag_96_13=np.abs((bnf_mag_96[:,0]*std_surf)-(bnf_mag_96[:,28]))

m_mag_96_13=np.abs((m_mag_96[:,0]*std_surf)-(m_mag_96[:,28]))

k=(np.array(bnf_mag_96_7).mean()+np.array(bnf_mag_96_8).mean()+np.array(bnf_mag_96_9).mean()+np.array(bnf_mag_96_10).mean()+np.array(bnf_mag_96_11).mean()+np.array(bnf_mag_96_12).mean())/6
print(k)
k=(np.array(bnf_mag_96_7).std()+np.array(bnf_mag_96_8).std()+np.array(bnf_mag_96_9).std()+np.array(bnf_mag_96_10).std()+np.array(bnf_mag_96_11).std()+np.array(bnf_mag_96_12).std())/6
print(k)
print("rt60 sc")
k=(np.array(m_mag_96_7).mean()+np.array(m_mag_96_8).mean()+np.array(m_mag_96_9).mean()+np.array(m_mag_96_10).mean()+np.array(m_mag_96_11).mean()+np.array(m_mag_96_12).mean())/6
print(k)
k=(np.array(m_mag_96_7).std()+np.array(m_mag_96_8).std()+np.array(m_mag_96_9).std()+np.array(m_mag_96_10).std()+np.array(m_mag_96_11).std()+np.array(m_mag_96_12).std())/6
print(k)


bnf_mag_96_14=np.abs((bnf_mag_96[:,1]*std_vol)-(bnf_mag_96[:,29]))

m_mag_96_14=np.abs((m_mag_96[:,1]*std_vol)-(m_mag_96[:,29]))

k=(np.array(bnf_mag_96_13).mean())
print(k)
k=(np.array(bnf_mag_96_14).mean())
print(k)
k=(np.array(bnf_mag_96_13).std())
print(k)
k=(np.array(bnf_mag_96_14).std())
print(k)


print("surf vol sc")
k=(np.array(m_mag_96_13).mean())
print(k)
k=(np.array(m_mag_96_13).std())
print(k)
k=(np.array(m_mag_96_14).mean())
print(k)
k=(np.array(m_mag_96_14).std())
print(k)





print("mse_mt")
mse_1=np.abs((mse_mlh[:,0])-(mse_mlh[:,14]))
mse_2=np.abs((mse_mlh[:,1])-(mse_mlh[:,15]))

mse_3=np.abs((mse_mlh[:,2])-(mse_mlh[:,16]))
mse_4=np.abs((mse_mlh[:,3])-(mse_mlh[:,17]))
mse_5=np.abs((mse_mlh[:,4])-(mse_mlh[:,18]))
mse_6=np.abs((mse_mlh[:,5])-(mse_mlh[:,19]))

print(np.array(mse_1).mean(),np.array(mse_1).std())
print(np.array(mse_2).mean(),np.array(mse_2).std())










mse_7=np.abs((mse_mlh[:,6])-(mse_mlh[:,20]))
mse_8=np.abs((mse_mlh[:,7])-(mse_mlh[:,21]))

k=(np.array(mse_3).mean()+np.array(mse_4).mean()+np.array(mse_5).mean()+np.array(mse_6).mean()+np.array(mse_7).mean()+np.array(mse_8).mean())/6
print(k)
k=(np.array(mse_3).std()+np.array(mse_4).std()+np.array(mse_5).std()+np.array(mse_6).std()+np.array(mse_7).std()+np.array(mse_8).std())/6
print(k)



mse_9=np.abs((mse_mlh[:,8])-(mse_mlh[:,22]))
mse_10=np.abs((mse_mlh[:,9])-(mse_mlh[:,23]))
mse_11=np.abs((mse_mlh[:,10])-(mse_mlh[:,24]))
mse_12=np.abs((mse_mlh[:,11])-(mse_mlh[:,25]))


mse_13=np.abs((mse_mlh[:,12])-(mse_mlh[:,26]))

mse_14=np.abs((mse_mlh[:,13])-(mse_mlh[:,27]))
k=(np.array(mse_9).mean()+np.array(mse_10).mean()+np.array(mse_11).mean()+np.array(mse_12).mean()+np.array(mse_13).mean()+np.array(mse_14).mean())/6
print(k)
k=(np.array(mse_9).std()+np.array(mse_10).std()+np.array(mse_11).std()+np.array(mse_12).std()+np.array(mse_13).std()+np.array(mse_14).std())/6
print(k)



print("mse_mt")
rt60_1=np.abs((rt60_mse[:,0])-(rt60_mse[:,6]))
rt60_2=np.abs((rt60_mse[:,1])-(rt60_mse[:,7]))

rt60_3=np.abs((rt60_mse[:,2])-(rt60_mse[:,8]))
rt60_4=np.abs((rt60_mse[:,3])-(rt60_mse[:,9]))
rt60_5=np.abs((rt60_mse[:,4])-(rt60_mse[:,10]))
rt60_6=np.abs((rt60_mse[:,5])-(rt60_mse[:,11]))

k=(np.array(rt60_1).mean()+np.array(rt60_2).mean()+np.array(rt60_3).mean()+np.array(rt60_4).mean()+np.array(rt60_5).mean()+np.array(rt60_6).mean())/6
print(k)
k=(np.array(rt60_1).std()+np.array(rt60_2).std()+np.array(rt60_3).std()+np.array(rt60_4).std()+np.array(rt60_5).std()+np.array(rt60_6).std())/6
print(k)


print("absorption_values")
ab_1=np.abs((ab_mse[:,0])-(ab_mse[:,6]))
ab_2=np.abs((ab_mse[:,1])-(ab_mse[:,7]))

ab_3=np.abs((ab_mse[:,2])-(ab_mse[:,8]))
ab_4=np.abs((ab_mse[:,3])-(ab_mse[:,9]))
ab_5=np.abs((ab_mse[:,4])-(ab_mse[:,10]))
ab_6=np.abs((ab_mse[:,5])-(ab_mse[:,11]))
k=(np.array(ab_1).mean()+np.array(ab_2).mean()+np.array(ab_3).mean()+np.array(ab_4).mean()+np.array(ab_5).mean()+np.array(ab_6).mean())/6
print(k)
k=(np.array(ab_1).std()+np.array(ab_2).std()+np.array(ab_3).std()+np.array(ab_4).std()+np.array(ab_5).std()+np.array(ab_6).std())/6
print(k)

print("surface and volume")
surf_mse=np.abs((surf_data[:,0])-(surf_data[:,1]))

print(np.array(surf_mse).mean())
print(np.array(surf_mse).std())


vol_mse=np.abs((vol_data[:,0])-(vol_data[:,1]))
print(np.array(vol_mse).mean())
print(np.array(vol_mse).std())


microsoft_mse=np.abs((micro_vol[:,0])-(micro_vol[:,1]))
print("microsoft")
print(np.array(microsoft_mse).mean())
print(np.array(microsoft_mse).std())



var_1=np.abs((fi_var[:,0])-(fi_var[:,14]))

var_2=np.abs((fi_var[:,1])-(fi_var[:,15]))

var_3=np.abs((fi_var[:,2])-(fi_var[:,16]))
var_4=np.abs((fi_var[:,3])-(fi_var[:,17]))
var_5=np.abs((fi_var[:,4])-(fi_var[:,18]))
var_6=np.abs((fi_var[:,5])-(fi_var[:,19]))

print(np.array(var_1).mean(),np.array(var_1).std())
print(np.array(var_2).mean(),np.array(var_2).std())




var_7=np.abs((fi_var[:,6])-(fi_var[:,20]))
var_8=np.abs((fi_var[:,7])-(fi_var[:,21]))

k=(np.array(var_3).mean()+np.array(var_4).mean()+np.array(var_5).mean()+np.array(var_6).mean()+np.array(var_7).mean()+np.array(var_8).mean())/6
print(k)
k=(np.array(var_3).std()+np.array(var_4).std()+np.array(var_5).std()+np.array(var_6).std()+np.array(var_7).std()+np.array(var_8).std())/6
print(k)





var_9=np.abs((fi_var[:,8])-(fi_var[:,22]))
var_10=np.abs((fi_var[:,9])-(fi_var[:,23]))
var_11=np.abs((fi_var[:,10])-(fi_var[:,24]))
var_12=np.abs((fi_var[:,11])-(fi_var[:,25]))




var_13=np.abs((fi_var[:,12])-(fi_var[:,26]))
var_14=np.abs((fi_var[:,13])-(fi_var[:,27]))

k=(np.array(var_9).mean()+np.array(var_10).mean()+np.array(var_11).mean()+np.array(var_12).mean()+np.array(var_13).mean()+np.array(var_14).mean())/6
print(k)
k=(np.array(var_9).std()+np.array(var_10).std()+np.array(var_11).std()+np.array(var_12).std()+np.array(var_13).std()+np.array(var_14).std())/6
print(k)



fig,axs=plt.subplots(7,2,figsize=(10,25))
#ep=10e-5
bplot1=axs[0,0].boxplot([dum_bnf_1,bnf_mag_96_1,mse_9,ab_1,var_9,m_mag_96_1],showmeans=True,vert=True,showfliers=True,patch_artist=True)
axs[0,0].set_xticks([1,2,3,4,5,6])
axs[0,0].set_xticklabels(['N M','MLH BN','MSE MT','MSE','FI-VAR','MLH M'],rotation=45)
axs[0,0].set_ylabel("Abs Err Ab")
axs[0,0].set_title("125hz")

#print(bplot1["fliers"][1].get_data()[1])




bplot2=axs[0,1].boxplot([dum_bnf_2,bnf_mag_96_2,mse_10,ab_2,var_10,m_mag_96_2],showmeans=True,vert=True,showfliers=False,patch_artist=True)
axs[0,1].set_xticks([1,2,3,4,5,6])
axs[0,1].set_xticklabels(['N M','MLH BN','MSE MT','MSE','FI-VAR','MLH M'],rotation=45)
axs[0,1].set_ylabel("Abs Err Ab")
axs[0,1].set_title("AB Coeff 250hz")


bplot3=axs[1,0].boxplot([dum_bnf_3,bnf_mag_96_3,mse_11,ab_3,var_11,m_mag_96_3],showmeans=True,vert=True,showfliers=False,patch_artist=True)
axs[1,0].set_xticks([1,2,3,4,5,6])
axs[1,0].set_xticklabels(['N M','MLH BN','MSE MT','MSE','FI-VAR','MLH M'],rotation=45)
axs[1,0].set_ylabel("Abs Err Ab")
axs[1,0].set_title("AB Coeff 500hz")


bplot4=axs[1,1].boxplot([dum_bnf_4,bnf_mag_96_4,mse_12,ab_4,var_12,m_mag_96_4],showmeans=True,vert=True,showfliers=False,patch_artist=True)
axs[1,1].set_xticks([1,2,3,4,5,6])
axs[1,1].set_xticklabels(['N M','MLH BN','MSE MT','MSE','FI-VAR','MLH M'],rotation=45)
axs[1,1].set_ylabel("Abs Err Ab")
axs[1,1].set_title("Ab Coeff 1000hz")


bplot5=axs[2,0].boxplot([dum_bnf_5,bnf_mag_96_5,mse_13,ab_5,var_13,m_mag_96_5],showmeans=True,vert=True,showfliers=False,patch_artist=True)
axs[2,0].set_xticks([1,2,3,4,5,6])
axs[2,0].set_xticklabels(['N M','MLH BN','MSE MT','MSE','FI-VAR','MLH M'],rotation=45)
axs[2,0].set_ylabel("Abs Err Ab")
axs[2,0].set_title("Ab Coeff 2000hz")


bplot6=axs[2,1].boxplot([dum_bnf_6,bnf_mag_96_6,mse_14,ab_6,var_14,m_mag_96_6],showmeans=True,vert=True,showfliers=False,patch_artist=True)
axs[2,1].set_xticks([1,2,3,4,5,6])
axs[2,1].set_xticklabels(['N M','MLH BN','MSE MT','MSE','FI-VAR','MLH M'],rotation=45)
axs[2,1].set_ylabel("Abs Err Ab")
axs[2,1].set_title("Ab Coeff 4000hz")


out_rt=False
bplot7=axs[3,0].boxplot([dum_bnf_7,bnf_mag_96_7,mse_3,rt60_1,var_3,m_mag_96_7],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[3,0].set_xticks([1,2,3,4,5,6])
axs[3,0].set_xticklabels(['N M','MLH BN','MSE MT','MSE','FI-VAR','MLH M'],rotation=45)
axs[3,0].set_ylabel("Abs Err Ab")
axs[3,0].set_title("RT 60 125hz")

#print(bplot1["fliers"][1].get_data()[1])




bplot8=axs[3,1].boxplot([dum_bnf_8,bnf_mag_96_8,mse_4,rt60_2,var_4,m_mag_96_8],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[3,1].set_xticks([1,2,3,4,5,6])
axs[3,1].set_xticklabels(['N M','MLH BN','MSE MT','MSE','FI-VAR','MLH M'],rotation=45)
axs[3,1].set_ylabel("Abs Err Sec")
axs[3,1].set_title("RT 60 250hz")


bplot9=axs[4,0].boxplot([dum_bnf_9,bnf_mag_96_9,mse_5,rt60_3,var_5,m_mag_96_9],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[4,0].set_xticks([1,2,3,4,5,6])
axs[4,0].set_xticklabels(['N M','MLH BN','MSE MT','MSE','FI-VAR','MLH M'],rotation=45)
axs[4,0].set_ylabel("Abs Err Sec")
axs[4,0].set_title("RT 60 500hz")
        
bplot10=axs[4,1].boxplot([dum_bnf_10,bnf_mag_96_10,mse_6,rt60_4,var_6,m_mag_96_10],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[4,1].set_xticks([1,2,3,4,5,6])
axs[4,1].set_xticklabels(['N M','MLH BN','MSE MT','MSE','FI-VAR','MLH M'],rotation=45)
axs[4,1].set_ylabel("Abs Err Sec")
axs[4,1].set_title("RT60 1000hz")


bplot11=axs[5,0].boxplot([dum_bnf_11,bnf_mag_96_11,mse_7,rt60_5,var_7,m_mag_96_11],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[5,0].set_xticks([1,2,3,4,5,6])
axs[5,0].set_xticklabels(['N M','MLH BN','MSE MT','MSE','FI-VAR','MLH M'],rotation=45)
axs[5,0].set_ylabel("Abs Err Sec")
axs[5,0].set_title("RT 60 2000hz")


bplot12=axs[5,1].boxplot([dum_bnf_12,bnf_mag_96_12,mse_8,rt60_6,var_8,m_mag_96_12],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[5,1].set_xticks([1,2,3,4,5,6])
axs[5,1].set_xticklabels(['N M','MLH BN','MSE MT','MSE','FI-VAR','MLH M'],rotation=45)
axs[5,1].set_ylabel("Abs Err Sec")
axs[5,1].set_title("RT 60 4000hz")

bplot13=axs[6,0].boxplot([dum_bnf_13,bnf_mag_96_13,mse_1,surf_mse,var_1,m_mag_96_13],showmeans=True,vert=True,showfliers=True,patch_artist=True)
axs[6,0].set_xticks([1,2,3,4,5,6])
axs[6,0].set_xticklabels(['NM','MLH BN','MSE MLH','MSE','FI-VAR','MLH M'],rotation=45)
axs[6,0].set_ylabel("Abs Err M2")
axs[6,0].set_title("Surface")


bplot14=axs[6,1].boxplot([dum_bnf_14,bnf_mag_96_14,mse_2,vol_mse,var_2,m_mag_96_14,microsoft_mse],showmeans=True,vert=True,showfliers=False,patch_artist=True)
axs[6,1].set_xticks([1,2,3,4,5,6,7])
axs[6,1].set_xticklabels(['NM','MLH BN','MSE MT','MSE','FI-VAR','MLH M','MS VOL'],rotation=45)
axs[6,1].set_ylabel("Abs Err M3")
axs[6,1].set_title("Volume")



colors=['pink','lightblue','lightgreen','orange','cyan']


for bplot in (bplot1,bplot2,bplot3,bplot4,bplot5,bplot6,bplot7,bplot8,bplot9,bplot10,bplot11,bplot12,bplot13,bplot14):
    for patch,color in zip(bplot['boxes'],colors):
        patch.set_facecolor(color)


fig.tight_layout(pad=3.0)
#plt.xticks([1,2,3],('Dummy Bnf','bnf','Dummy M','M'))
#plt.title("Absolute Diff Estimated Mean And Target RT60")
plt.savefig("volume_surface_comparasion_ff.png")


#surface_mean=[np.mean(dcd[:,0])]*abc.shape[0]
#surface_mean=np.array(surface_mean)
#height_mean=np.array([np.mean(dcd[:,1])]*abc.shape[0])

'''
volume_mean=np.array([np.mean(dcd[:,0])]*bnr.shape[0])

sns.set_theme()
rn=np.abs((bnr[:,0])-(bnr[:,1]))
sk=np.abs((sks[:,0])-(sks[:,1]))
ls=np.abs((lsl[:,0])-(lsl[:,1]))
vm=np.abs((volume_mean[:])-(lsl[:,1]))
kc=np.abs((kcd[:,0])-(kcd[:,1]))
hc=np.abs((hcd[:,0])-(hcd[:,1]))
'''


'''
ab=np.abs((abc[:,0])-(abc[:,1]))
bc=np.abs((bcd[:,0])-(bcd[:,1]))
kc=np.abs((kcd[:,0])-(kcd[:,1]))
xc=np.abs((xcd[:,0])-(xcd[:,1]))
nc=np.abs((ncd[:,0])-(ncd[:,1]))
lc=np.abs((lcd[:,0])-(lcd[:,1]))
ec=np.abs((ecd[:,0])-(ecd[:,1]))
hc=np.abs((hcd[:,0])-(hcd[:,1]))
bn=np.abs((bnf[:,0])-(bnf[:,1]))
bs=np.abs((bns[:,0])-(bns[:,1]))
rn=np.abs((bnr[:,0])-(bnr[:,1]))
sk=np.abs((sks[:,0])-(sks[:,1]))
ls=np.abs((lsl[:,0])-(lsl[:,1]))
vm=np.abs((volume_mean[:])-(dcd[:,1]))
'''

#ab10=np.abs((10**abc[:,0])-(10**bcd[:,2]))
#bc10=np.abs((10**bcd[:,0])-(10**bcd[:,2]))
#kc10=np.abs((10**kcd[:,0])-(10**kcd[:,2]))
#abh=np.abs((abc[:,1])-(abc[:,3]))
#bch=np.abs((bcd[:,1])-(bcd[:,3]))
#kch=np.abs((kcd[:,1])-(kcd[:,3]))
#xch=np.abs((xcd[:,1])-(xcd[:,3]))
#nch=np.abs((ncd[:,1])-(ncd[:,3]))
#lch=np.abs((lcd[:,1])-(lcd[:,3]))




#abh10=np.abs((10**abc[:,1])-(10**bcd[:,3]))
#bch10=np.abs((10**bcd[:,1])-(10**bcd[:,3]))
#kch10=np.abs((10**kcd[:,1])-(10**kcd[:,3]))


'''
plt.title("Abosolute Diff predicted mean vs target Rt60 4000")
plt.xlabel("Time In Seconds")
plt.boxplot([dum_sh,mag_96_s],showmeans=True,vert=False,showfliers=True)
#plt.xlim(-0.1,1)
plt.yticks([1,2],('Dummy Method','96ms mag'))
plt.savefig("mlh_nobnf_rt60_4000_bxp.png",bbox_inches='tight')
plt.clf()
'''
'''
plt.title("Abosolute Diff (Volume) predicted vs target")
plt.xlabel("Log (Volume) ")
plt.boxplot([kc,ls,hc,rn,sk,vm],showmeans=True,vert=False,showfliers=True)
plt.xlim(-0.1,1)
plt.yticks([1,2,3,4,5,6],('64 mag','128 mag','Ixvec 64 mag2','BNF 96 2 PIPE IPD ILD','BNF 128 2 PIPE IPD ILD','Dummy Method'))
plt.savefig("log_volume_ppt.png",bbox_inches='tight')
plt.clf()
'''

'''
plt.title("Abosolute Diff (Volume) predicted vs target")
plt.ylabel("Volume m3")
plt.boxplot([ab,bc,kc,rn,sk,ls,xc,bn,bs,nc,lc,ec,hc,vm],showmeans=True,vert=False,showfliers=True)
plt.xlim(-0.1,1)
plt.yticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14],('Batch Norm Mag','Ln 64 log mag','Ln 64 mag','BNF 96 2 PIPE IPD ILD','BNF 128 2 PIPE IPD ILD','Ln 128 mag','Ln 64 mag2','BNF catrimag','BNF 64 2 PIPE IPD ILD','Ln 32 mag2','LN 16 mag2','LF','Ixvec 64 mag2','Dummy input'))
plt.savefig("log_volume_final_correct_way_m3.png",bbox_inches='tight')
plt.clf()
#kd=0
'''
'''
for i in range(ab.shape[0]):
    kd+=np.abs(np.log((10**(xcd[i,0]))/(10**(xcd[i,1]))))
kd=np.exp(kd/ab.shape[0])
p
plt.title("Abosolute Diff (Volume) predicted vs target")
plt.ylabel("Volume m3")
plt.boxplot([ab,bc,kc,rn,sk,ls,xc,bn,bs,nc,lc,ec,hc,vm],showmeans=True,vert=False,showfliers=True)
plt.xlim(-0.1,1)
plt.yticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14],('Batch Norm Mag','Ln 64 log mag','Ln 64 mag','BNF 96 2 PIPE IPD ILD','BNF 128 2 PIPE IPD ILD','Ln 128 mag','Ln 64 mag2','BNF catrimag','BNF 64 2 PIPE IPD ILD','Ln 32 mag2','LN 16 mag2','LF','Ixvec 64 mag2','Dummy input'))
plt.savefig("log_volume_final_correct_way_m3.png",bbox_inches='tight')
plt.clf()
#rint(kd)
'''
'''
plt.title("Abosolute Diff log10 (Height) predicted vs target")
plt.ylabel("log 10 (Height)")
plt.boxplot([abh,bch,kch,xch,nch,lch,height_mean])
plt.xticks([1,2,3,4,5,6,7],('No Dilated','Dilated 1248 mag','Dilated 124','dilated fixed','dilated mag','Improv xvec mag','Dummy input'),rotation=5)
plt.savefig("log_surface_height.png")
plt.clf()
'''
#




'''
plt.title("Abosolute Diff log10 (Volume) predicted vs target Xvec")
plt.ylabel("log 10 (Volume)")
plt.xlabel("Xvec")
plt.boxplot([kc])
#plt.xticks([1],('Xvec'),rotation=5)
plt.savefig("log_volume_xvec.png")
plt.clf()
'''



'''
plt.title("Abosolute Diff surface area predicted vs target")
plt.ylabel("surface area m2")
plt.boxplot([ab10,bc10,kc10])
plt.xticks([1,2,3],('Learned FB','Fixed FB Re(STFT) Im(STFT)','Mag(STFT)'),rotation=5)
plt.savefig("surface_area.png")
plt.clf()
plt.title("Abosolute Diff log10 (height) predicted vs target")
plt.ylabel("log 10 (height)")
plt.boxplot([abh,bch,kch])
plt.xticks([1,2,3],('Learned FB','Fixed FB Re(STFT) Im(STFT)','Mag(STFT)'),rotation=5)
plt.savefig("log_height.png")
plt.clf()
plt.title("Abosolute Diff height predicted vs target")
plt.ylabel("height in m")
plt.boxplot([abh10,bch10,kch10])
plt.xticks([1,2,3],('Learned FB','Fixed FB Re(STFT) Im(STFT)','Mag(STFT)'),rotation=5)
plt.savefig("height.png")
'''
