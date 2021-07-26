import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

abc=np.load("vp_1_test.npy")
abcd=np.load("vp_2_test.npy")
abcde=np.load("vp_3_test.npy")
abcdef=np.load("vp_2_weightmean_test.npy")

std_vol=106.02265
std_surf=84.2240762
std_rt60=[ 0.7793691,0.7605436,0.6995225, 0.7076664, 0.6420753,0.51794204]
std_abs=[0.100825,0.1172430,0.1002776,0.09108845,0.09378748,0.091663032]


err_1=np.abs(abc[:,0]*std_surf-abc[:,14])
err_2=np.abs(abc[:,1]*std_vol-abc[:,15])
err_3=np.abs(abc[:,2]*std_rt60[0]-abc[:,16])
err_4=np.abs(abc[:,3]*std_rt60[1]-abc[:,17])
err_5=np.abs(abc[:,4]*std_rt60[2]-abc[:,18])
err_6=np.abs(abc[:,5]*std_rt60[3]-abc[:,19])
err_7=np.abs(abc[:,6]*std_rt60[4]-abc[:,20])
err_8=np.abs(abc[:,7]*std_rt60[5]-abc[:,21])
err_9=np.abs(abc[:,8]*std_abs[0]-abc[:,22])
err_10=np.abs(abc[:,9]*std_abs[1]-abc[:,23])
err_11=np.abs(abc[:,10]*std_abs[2]-abc[:,24])
err_12=np.abs(abc[:,11]*std_abs[3]-abc[:,25])
err_13=np.abs(abc[:,12]*std_abs[4]-abc[:,26])
err_14=np.abs(abc[:,13]*std_abs[5]-abc[:,27])

err2_1=np.abs(abcd[:,0]*std_surf-abcd[:,14])
err2_2=np.abs(abcd[:,1]*std_vol-abcd[:,15])
err2_3=np.abs(abcd[:,2]*std_rt60[0]-abcd[:,16])
err2_4=np.abs(abcd[:,3]*std_rt60[1]-abcd[:,17])
err2_5=np.abs(abcd[:,4]*std_rt60[2]-abcd[:,18])
err2_6=np.abs(abcd[:,5]*std_rt60[3]-abcd[:,19])
err2_7=np.abs(abcd[:,6]*std_rt60[4]-abcd[:,20])
err2_8=np.abs(abcd[:,7]*std_rt60[5]-abcd[:,21])
err2_9=np.abs(abcd[:,8]*std_abs[0]-abcd[:,22])
err2_10=np.abs(abcd[:,9]*std_abs[1]-abcd[:,23])
err2_11=np.abs(abcd[:,10]*std_abs[2]-abcd[:,24])
err2_12=np.abs(abcd[:,11]*std_abs[3]-abcd[:,25])
err2_13=np.abs(abcd[:,12]*std_abs[4]-abcd[:,26])
err2_14=np.abs(abcd[:,13]*std_abs[5]-abcd[:,27])

err3_1=np.abs(abcde[:,0]*std_surf-abcde[:,14])
err3_2=np.abs(abcde[:,1]*std_vol-abcde[:,15])
err3_3=np.abs(abcde[:,2]*std_rt60[0]-abcde[:,16])
err3_4=np.abs(abcde[:,3]*std_rt60[1]-abcde[:,17])
err3_5=np.abs(abcde[:,4]*std_rt60[2]-abcde[:,18])
err3_6=np.abs(abcde[:,5]*std_rt60[3]-abcde[:,19])
err3_7=np.abs(abcde[:,6]*std_rt60[4]-abcde[:,20])
err3_8=np.abs(abcde[:,7]*std_rt60[5]-abcde[:,21])
err3_9=np.abs(abcde[:,8]*std_abs[0]-abcde[:,22])
err3_10=np.abs(abcde[:,9]*std_abs[1]-abcde[:,23])
err3_11=np.abs(abcde[:,10]*std_abs[2]-abcde[:,24])
err3_12=np.abs(abcde[:,11]*std_abs[3]-abcde[:,25])
err3_13=np.abs(abcde[:,12]*std_abs[4]-abcde[:,26])
err3_14=np.abs(abcde[:,13]*std_abs[5]-abcde[:,27])


err4_1=np.abs(abcdef[:,0]*std_surf-abcdef[:,14])
err4_2=np.abs(abcdef[:,1]*std_vol-abcdef[:,15])
err4_3=np.abs(abcdef[:,2]*std_rt60[0]-abcdef[:,16])
err4_4=np.abs(abcdef[:,3]*std_rt60[1]-abcdef[:,17])
err4_5=np.abs(abcdef[:,4]*std_rt60[2]-abcdef[:,18])
err4_6=np.abs(abcdef[:,5]*std_rt60[3]-abcdef[:,19])
err4_7=np.abs(abcdef[:,6]*std_rt60[4]-abcdef[:,20])
err4_8=np.abs(abcdef[:,7]*std_rt60[5]-abcdef[:,21])
err4_9=np.abs(abcdef[:,8]*std_abs[0]-abcdef[:,22])
err4_10=np.abs(abcdef[:,9]*std_abs[1]-abcdef[:,23])
err4_11=np.abs(abcdef[:,10]*std_abs[2]-abcdef[:,24])
err4_12=np.abs(abcdef[:,11]*std_abs[3]-abcdef[:,25])
err4_13=np.abs(abcdef[:,12]*std_abs[4]-abcdef[:,26])
err4_14=np.abs(abcdef[:,13]*std_abs[5]-abcdef[:,27])


std_vol=106.02265
std_surf=84.2240762
std_rt60=[ 0.7793691,0.7605436,0.6995225, 0.7076664, 0.6420753,0.51794204]
std_abs=[0.100825,0.1172430,0.1002776,0.09108845,0.09378748,0.091663032]

fig,axs=plt.subplots(7,2,figsize=(10,25))
#ep=10e-5
bplot1=axs[0,0].boxplot([err_9,err2_9,err3_9,err4_9],showmeans=True,vert=True,showfliers=True,patch_artist=True)
axs[0,0].set_xticks([1,2,3,4])
axs[0,0].set_xticklabels(['MN 1 VP','MN 2 VP','MN 3 vp','MN WM'],rotation=45)
axs[0,0].set_ylabel("Abs Err Ab")
axs[0,0].set_title("125hz")

#print(bplot1["fliers"][1].get_data()[1])




bplot2=axs[0,1].boxplot([err_10,err2_10,err3_10,err4_10],showmeans=True,vert=True,showfliers=False,patch_artist=True)
axs[0,1].set_xticks([1,2,3,4])
axs[0,1].set_xticklabels(['MN 1 VP','MN 2 VP','MN 3 vp','MN WM'],rotation=45)
axs[0,1].set_ylabel("Abs Err Ab")
axs[0,1].set_title("AB Coeff 250hz")


bplot3=axs[1,0].boxplot([err_11,err2_11,err3_11,err4_11],showmeans=True,vert=True,showfliers=False,patch_artist=True)
axs[1,0].set_xticks([1,2,3,4])
axs[1,0].set_xticklabels(['MN 1 vp','MN 2 vp','MN 3 vp','MN WM'],rotation=45)
axs[1,0].set_ylabel("Abs Err Ab")
axs[1,0].set_title("AB Coeff 500hz")


bplot4=axs[1,1].boxplot([err_12,err2_12,err3_12,err4_12],showmeans=True,vert=True,showfliers=False,patch_artist=True)
axs[1,1].set_xticks([1,2,3,4])
axs[1,1].set_xticklabels(['MN 1 vp','MN 2 vp','MN 3 vp','MN WM'],rotation=45)
axs[1,1].set_ylabel("Abs Err Ab")
axs[1,1].set_title("Ab Coeff 1000hz")


bplot5=axs[2,0].boxplot([err_13,err2_13,err3_13,err4_13],showmeans=True,vert=True,showfliers=False,patch_artist=True)
axs[2,0].set_xticks([1,2,3,4])
axs[2,0].set_xticklabels(['MN 1 vp','MN 2 vp','MN 3 vp','MN WM'],rotation=45)
axs[2,0].set_ylabel("Abs Err Ab")
axs[2,0].set_title("Ab Coeff 2000hz")


bplot6=axs[2,1].boxplot([err_14,err2_14,err3_14,err4_14],showmeans=True,vert=True,showfliers=False,patch_artist=True)
axs[2,1].set_xticks([1,2,3,4])
axs[2,1].set_xticklabels(['MN 1 vp','MN 2 vp','MN 3 vp','MN WM'],rotation=45)
axs[2,1].set_ylabel("Abs Err Ab")
axs[2,1].set_title("Ab Coeff 4000hz")


out_rt=False
bplot7=axs[3,0].boxplot([err_3,err2_3,err3_3,err4_3],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[3,0].set_xticks([1,2,3,4])
axs[3,0].set_xticklabels(['MN 1 src','MN 2 vp','MN 3 vp','MN WM'],rotation=45)
axs[3,0].set_ylabel("Abs Err Ab")
axs[3,0].set_title("RT 60 125hz")

#print(bplot1["fliers"][1].get_data()[1])




bplot8=axs[3,1].boxplot([err_4,err2_4,err3_4,err4_4],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[3,1].set_xticks([1,2,3,4])
axs[3,1].set_xticklabels(['MN 1 vp','MN 2 vp','MN 3 vp','MN WM'],rotation=45)
axs[3,1].set_ylabel("Abs Err Sec")
axs[3,1].set_title("RT 60 250hz")


bplot9=axs[4,0].boxplot([err_5,err2_5,err3_5,err4_5],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[4,0].set_xticks([1,2,3,4])
axs[4,0].set_xticklabels(['MN 1 vp','MN 2 vp','MN 3 vp','MN WM'],rotation=45)
axs[4,0].set_ylabel("Abs Err Sec")
axs[4,0].set_title("RT 60 500hz")
        
bplot10=axs[4,1].boxplot([err_6,err2_6,err3_6,err4_6],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[4,1].set_xticks([1,2,3,4])
axs[4,1].set_xticklabels(['MN 1 vp','MN 2 vp','MN 3 vp','MN WM'],rotation=45)
axs[4,1].set_ylabel("Abs Err Sec")
axs[4,1].set_title("RT60 1000hz")


bplot11=axs[5,0].boxplot([err_7,err2_7,err3_7,err4_7],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[5,0].set_xticks([1,2,3,4])
axs[5,0].set_xticklabels(['MN 1 vp','MN 2 vp','MN 3 vp','MN WM'],rotation=45)
axs[5,0].set_ylabel("Abs Err Sec")
axs[5,0].set_title("RT 60 2000hz")


bplot12=axs[5,1].boxplot([err_8,err2_8,err3_8,err4_8],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[5,1].set_xticks([1,2,3,4])
axs[5,1].set_xticklabels(['MN 1 vp','MN 2 vp','MN 3 vp','MN WM'],rotation=45)
axs[5,1].set_ylabel("Abs Err Sec")
axs[5,1].set_title("RT 60 4000hz")

bplot13=axs[6,0].boxplot([err_1,err2_1,err3_1,err4_1],showmeans=True,vert=True,showfliers=True,patch_artist=True)
axs[6,0].set_xticks([1,2,3,4])
axs[6,0].set_xticklabels(['MN 1 vp','MN 2 vp','MN 3 vp','MN WM'],rotation=45)
axs[6,0].set_ylabel("Abs Err M2")
axs[6,0].set_title("Surface")

bplot14=axs[6,1].boxplot([err_2,err2_2,err3_2,err4_2],showmeans=True,vert=True,showfliers=True,patch_artist=True)
axs[6,1].set_xticks([1,2,3,4])
axs[6,1].set_xticklabels(['MN 1 vp','MN 2 vp','MN 3 vp','MN WM'],rotation=45)
axs[6,1].set_ylabel("Abs Err M3")
axs[6,1].set_title("Volume")



colors=['pink','lightblue','lightgreen','orange','cyan']


for bplot in (bplot1,bplot2,bplot3,bplot4,bplot5,bplot6,bplot7,bplot8,bplot9,bplot10,bplot11,bplot12,bplot13,bplot14):
    for patch,color in zip(bplot['boxes'],colors):
        patch.set_facecolor(color)


fig.tight_layout(pad=3.0)
#plt.xticks([1,2,3],('Dummy Bnf','bnf','Dummy M','M'))
#plt.title("Absolute Diff Estimated Mean And Target RT60")
plt.savefig("test_mono_comparasion.png")
