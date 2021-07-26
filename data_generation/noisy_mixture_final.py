import os 
import numpy as np 
import h5py
import soundfile as sf
from scipy import signal 
import random 
from audiogen.audiogen.noise.synthetic import SSN
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from scipy.io.wavfile import write
from scipy.spatial import distance
import math
import acoustics 



#Train speech data 100000
root_speech_data='/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/train-clean-360/'
#Speech shape noise data 28000
root_ssn_data='/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/train-clean-100/'
#Validation data speech
root_speech_val_data='/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/dev-clean/'
#Test speech data
root_speech_test_data='/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/test-clean/'
#RIR Data
rir_data_path='/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/generated_rirs_lwh_correct.hdf5'
#Reverb data
rir_noise_data_path='/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/generated_rirs_lwh_correct_noise.hdf5'


speech_files=[]
num=0

#open speech file 
train_file=np.load('speech_files_.npy')
np.random.shuffle(train_file)

val_file=np.load('speech_val_files_.npy')
np.random.shuffle(val_file)

test_file=np.load('speech_test_files_.npy')
np.random.shuffle(test_file)


#open rir file
rir_data=h5py.File(rir_data_path,'r')
rir_late_reverb_data=h5py.File(rir_noise_data_path,'r')

#Speech shape noise
ssn_no=np.load('ssn_files_.npy')
print(ssn_no)

#Random noise for ref signal 
no,sr=sf.read('rand_sig.flac')

#referense signal
ref_sig=h5py.File('gd_rif.hdf5','r')['rif']['room_0']['noise_rir']
ref_sig_ch1=ref_sig[0,:]
ref_sig_ch2=ref_sig[1,:]
filter_ref_sig_ch1=signal.convolve(no,ref_sig_ch1,mode='full')
filter_ref_sig_ch2=signal.convolve(no,ref_sig_ch2,mode='full')
mean_var=np.var(np.concatenate((filter_ref_sig_ch1,filter_ref_sig_ch2),axis=0))

print(mean_var)

#Randomize room
room_nos=[i+1 for i in range(20000)]

#room_nos=[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6]
#room_nos=[207,207,207,207,207]
#Randomize view points
#vp_nos=[1,2,3,4,5]*20000
#vp_nos=[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
vp_nos=[1,2,3,4,5]

#np.random.shuffle(room_nos)
#np.random.shuffle(vp_nos)

adc=[]
kbc=[]
alc=[]
euc_dist=[]
#rir_enr=[]
#conv_enr=[]
n=[]
sp_var=[]
count_file=0
#Generate 100000 noisy mixtures
#Save file
test_room=np.arange(18000,20000)

nf_file=h5py.File("/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_lwh_.hdf5",'w')
room_save=nf_file.create_group("room_nos")
for i in test_room:#19999

    room=room_nos[i]
    n.append(str(room))
    room_id=room_save.create_group("room_"+str(room))
    np_save_mix=np.zeros((10,48000))
    print("room_no",room)
    for j in vp_nos:
        vp=j
        #print((vp-1)*2,(vp*2)-1,room)
        ch_1=rir_data['rirs']['room_'+str(room)]['rir'][(vp-1)*2,:]
        ch_2=rir_data['rirs']['room_'+str(room)]['rir'][((vp*2)-1),:]
        split_sp=train_file[count_file].split('-')
    
        sp,srs=sf.read(root_speech_data+"/"+split_sp[0]+"/"+split_sp[1]+"/"+train_file[count_file])
        count_file+=1
        
        sp_var.append(-10*np.log(np.var(sp)))
        #print(count_file)

        #Just for the sake of expirement i am taking 3 second signal
        reverb_sig_ch1=signal.convolve(sp,ch_1,mode='full')[20:48020]
        reverb_sig_ch2=signal.convolve(sp,ch_2,mode='full')[20:48020]

        #Reverb After 50 seconds
        laterev_ch1=rir_late_reverb_data['rirs_noise']['room_'+str(room)]['noise_rir'][(vp-1)*2,:][800:]
        laterev_ch2=rir_late_reverb_data['rirs_noise']['room_'+str(room)]['noise_rir'][((vp*2)-1),:][800:]

        rand_10_no=[random.randint(0,28300) for k in range(10)]

        ssn_file=[root_ssn_data+"/"+ssn_no[k].split("-")[0]+"/"+ssn_no[k].split("-")[1]+"/"+ssn_no[k] for k in rand_10_no]

        #Generate speech shape noise
        ssn_obj=SSN(ssn_file)
        ssn_noise=ssn_obj.generate(3*ssn_obj.target_sr,rnd_gen=np.random.default_rng(123))
        #Convolve with late reverb
        laterev_ch1_ssn=signal.convolve(ssn_noise,laterev_ch1,mode='full')[20:48020]
        #print("laterev",laterev_ch1.shape)
        laterev_ch2_ssn=signal.convolve(ssn_noise,laterev_ch2,mode='full')[20:48020]

        #Calculate alpha and beta
        snr_static=random.randint(60,70)

        snr_diff=random.randint(30,60)

        sigma_diff=mean_var/(np.power(10,(snr_diff/10)))
        sigma_static=mean_var/(np.power(10,(snr_static/10)))


        #Calculate alpha
        sigma_ssn=np.var(np.concatenate((laterev_ch1_ssn,laterev_ch2_ssn),axis=0))
        alpha=np.sqrt((sigma_diff/sigma_ssn))


        #2 Channel White noise
        white_noise_ch1=np.random.normal(0,1,size=48000)
        white_noise_ch2=np.random.normal(0,1,size=48000)

        #Calculate beta
        beta=np.sqrt(sigma_static)

        #fixing alpha and beta
        #alpha_fix=0.001
        #beta_fix=0.001

        static_noise_ch1=white_noise_ch1*beta
        static_noise_ch2=white_noise_ch2*beta

        #static_noise_ch1_fix=white_noise_ch1*beta_fix
        #static_noise_ch2_fix=white_noise_ch2*beta_fix

        diff_noise_ch1=laterev_ch1_ssn*alpha
        diff_noise_ch2=laterev_ch2_ssn*alpha

        #diff_noise_ch1_fix=laterev_ch1_ssn*alpha_fix
        #diff_noise_ch2_fix=laterev_ch2_ssn*alpha_fix

        #signalf_ch1_fix=reverb_sig_ch1+diff_noise_ch1_fix+static_noise_ch1_fix
        #signalf_ch2_fix=reverb_sig_ch2+diff_noise_ch2_fix+static_noise_ch2_fix

        signalf_ch1=reverb_sig_ch1+diff_noise_ch1+static_noise_ch1
        signalf_ch2=reverb_sig_ch2+diff_noise_ch2+static_noise_ch2
        np_save_mix[(vp-1)*2,:]=signalf_ch1
        np_save_mix[((vp*2)-1),:]=signalf_ch2

        #signalf_ch1=np.add(reverb_sig_ch1,diff_noise_ch1)
        #signalf_ch1=np.add(signalf_ch1,static_noise_ch1)
        #signalf_ch2=np.add(reverb_sig_ch2,diff_noise_ch2)
        #signalf_ch2=np.add(signalf_ch2,static_noise_ch2)

        #Doing this so that we can write in stereo files
        #signalf_ch1_=np.reshape(signalf_ch1,(-1,1))
        #signalf_ch2_=np.reshape(signalf_ch2,(-1,1))
        #abc=np.concatenate((signalf_ch1_,signalf_ch2_),axis=1)

        diff_noise_f=np.concatenate((diff_noise_ch1,diff_noise_ch2),axis=0)
        #print(diff_noise_f.shape)
        static_noise_f=np.concatenate((static_noise_ch1,static_noise_ch2),axis=0)

        #diff_noise_f_fix=np.concatenate((diff_noise_ch1_fix,diff_noise_ch2_fix),axis=0)
        #static_noise_f_fix=np.concatenate((static_noise_ch1_fix,static_noise_ch2_fix),axis=0)


        snr_f=10*np.log10((np.var(np.concatenate((signalf_ch1,signalf_ch2),axis=0)))/(np.var(diff_noise_f)+np.var(static_noise_f)))
        #snr_fix=10*np.log10((np.var(np.concatenate((signalf_ch1_fix,signalf_ch2_fix),axis=0)))/(np.var(np.add(diff_noise_f_fix,static_noise_f_fix))))

        snr_f2=10*np.log10((np.var(np.concatenate((signalf_ch1,signalf_ch2),axis=0)))/(np.var(diff_noise_f)))
        snr_f3=10*np.log10((np.var(np.concatenate((signalf_ch1,signalf_ch2),axis=0)))/(np.var(static_noise_f)))

        adc.append(snr_f)
        kbc.append(snr_f2)
        alc.append(snr_f3)
        #kbc.append(snr_fix)
        #rir_enr.append(acoustics.Signal(np.concatenate((ch_1,ch_2),axis=0),16000).energy())
        #conv_enr.append(acoustics.Signal(np.concatenate((reverb_sig_ch1,reverb_sig_ch2),axis=0),16000).energy())

        #print(snr_f)
        #print(snr_f,snr_f2,snr_f3)
        #euc_dist.append(distance.euclidean(rir_data['receiver_config']['room_'+str(room)]['barycenter'][(vp-1),:],rir_data['source_config']['room_'+str(room)]['source_pos'][(vp-1),:]))
        #print("example",i)
    room_id.create_dataset("nsmix_f",(10,48000),data=np_save_mix)
    room_id.create_dataset("nsmix_snr_f",5,data=adc[-5:])
    room_id.create_dataset("nsmix_snr_diff", 5, data=kbc[-5:])
    room_id.create_dataset("nsmix_snr_static", 5, data=alc[-5:])

''' 
        f, Pxx, Sxx=signal.spectrogram(signalf_ch1,16000)
        f2, Pxx_2, Sxx_2=signal.spectrogram(reverb_sig_ch1,16000)
        f3, Pxx_3, Sxx_3=signal.spectrogram(diff_noise_ch1,16000)
        f4, Pxx_4, Sxx_4=signal.spectrogram(ssn_noise,16000)
        #f5, Pxx_5, Sxx_5=signal.spectrogram(ch1, 16000)
    
        fig=plt.figure()
        gs=gridspec.GridSpec(2,2,wspace=0.5,hspace=0.5)
        ax=plt.subplot(gs[0,0])
        ax1=plt.subplot(gs[0,1])
        ax2=plt.subplot(gs[1,0])
        ax3=plt.subplot(gs[1,1])
    
        ax.pcolormesh(Pxx,f,np.log10(abs(Sxx)),shading='gouraud')
        ax.set_title("Final Noisy Signal")
        ax1.pcolormesh(Pxx_2,f2,np.log10(abs(Sxx_2)), shading='gouraud')
        ax1.set_title("Reverb Signal")
        ax2.pcolormesh(Pxx_3,f3,np.log10(abs(Sxx_3)), shading='gouraud')
        ax2.set_title("Diffuse Noise")
        ax3.pcolormesh(Pxx_4,f4,np.log10(abs(Sxx_4)), shading='gouraud')
        ax3.set_title("Speech Shape Noise")
    
        
    
        #plt.semilogy(f,np.sqrt(Pxx),label="Noisy Mixture")
        #plt.semilogy(f2,np.sqrt(Pxx_2),c='y',label="Clean Rev Signal")
        #plt.xlabel("Frequency")
        #plt.ylabel("Linear Spectrum [V RMS]")
        #plt.legend()
        #print("reverb signal",np.sqrt(Pxx_2.max()))
    
    
        #plt.semilogy(f,np.sqrt(Pxx))
        
        fig.add_subplot(ax)
        fig.add_subplot(ax1)
        fig.add_subplot(ax2)
        fig.add_subplot(ax3)
        
        
        fig.savefig("psd_noisy_"+str(i)+".jpeg")
        
        fig.clf()
        #plt.semilogy(f2, np.sqrt(Pxx_2))
        #plt.savefig("rever_"+str(i)+".jpeg")
    
        #plt.plot(np.arange(48000),signalf_ch1)
        #plt.savefig("ref_noise_"+str(i)+".jpeg")
        '''

        #write("ns_mix_"+str(room)+str(i+1)+".wav",16000,abc)
        #sf.write("ns_mix_"+"ch_2_"+str(i)+".flac",signalf_ch2,16000)
        #sf.write("ns_simple_"+str(i)+".flac",reverb_sig_ch1,16000)
    
    

#fig,ax1=plt.subplots()
#cs=['b','b','b','b','b','r','r','r','r','r','g','g','g','g','g','y','y','y','y','y','orange','orange','orange','orange','orange','violet','violet','violet','violet','violet']
#for l in range(30):


#ax1.scatter(euc_dist,adc,color='b')

#for i,txt in enumerate(n):
#    ax1.annotate(txt,(euc_dist[i],adc[i]))


#ax1.set_ylabel('SNR dB',c='b')
#ax1.set_xlabel('Euclidian Distance',c='b')

#print(euc_dist)
#print(adc)

'''
ax2=ax1.twinx()
ax2.plot(np.arange(10),euc_dist,c='r',marker='s',label="Euclidian Distance")
ax2.set_ylabel('Euc distance',c='r')
'''

#fig.tight_layout()
#plt.legend()
#plt.savefig("/home/psrivastava/baseline/scripts/haikus_project/snr_plot_fix_snr_static_room_200samples.jpeg")
#fig.clf()


fig,ax2=plt.subplots()
ax2.hist(adc)
ax2.set_ylabel("No of samples")
ax2.set_xlabel("Negative log of variance -log(np.var(sp_signal))")
fig.tight_layout()
plt.savefig("hist_static_final_lwh.jpeg")

fig,ax3=plt.subplots()
ax3.hist(kbc)
ax3.set_xlabel("SNR dB Diffuse Noise")
plt.savefig("hist_diffnoise_final_lwh.jpeg")

fig,ax4=plt.subplots()
ax4.hist(alc)
ax4.set_xlabel("SNR dB Static Noise")
plt.savefig("hist_staticnoise_final_lwh.jpeg")




'''
fig2,ax2=plt.subplots()

for n in range(30):
    ax2.scatter(euc_dist[n],kbc[n],color=cs[n])
ax2.set_ylabel('SNR dB',c='b')
ax2.set_xlabel('Euclidian Distance',c='b')
print(euc_dist)
print(kbc)
fig2.tight_layout()
plt.legend()
plt.savefig("/home/psrivastava/baseline/scripts/haikus_project/snr_plot_fix_alpha_beta_room.jpeg")

fig3,ax3=plt.subplots()
for n in range(30):
    ax3.scatter(euc_dist[n],rir_enr[n],color=cs[n])
ax3.set_ylabel('Energy',c='b')
ax3.set_xlabel('Euclidian Distance',c='b')
fig3.tight_layout()
plt.legend()
plt.savefig("/home/psrivastava/baseline/scripts/haikus_project/energy_decay_rir.jpeg")

fig4,ax4=plt.subplots()
for n in range(30):
    ax4.scatter(euc_dist[n],conv_enr[n],color=cs[n])
ax4.set_ylabel("Energy",c='b')
ax4.set_xlabel('Euclidian Distance',c='b')
fig4.tight_layout()
plt.legend()
plt.savefig("/home/psrivastava/baseline/scripts/haikus_project/energy_decay_reverbspeech.jpeg")

    
'''














'''
for i in os.listdir(root_speech_test_data):
    #speech_files[i]={j:[] for j in os.listdir(root_ssn_data+"/"+i)}
    for j in os.listdir(root_speech_test_data+"/"+i):
        #js=[]
        for k in os.listdir(root_speech_test_data+"/"+i+"/"+j):
            if ".flac" in k:
                data,sr=sf.read(root_speech_test_data+"/"+i+"/"+j+"/"+k)
                if data.shape[0] > (sr*2 + 1000):
                    speech_files.append(k)
                    num+=1
        #speech_files[i][j]=js



np.save("speech_test_files_.npy",speech_files)
#print(speech_files)
print(num)
'''
