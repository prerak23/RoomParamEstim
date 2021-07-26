import numpy as np
import yaml
import h5py
from acoustics import room
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.io.wavfile import read
from acoustics.signal import bandpass
from scipy import stats
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#Calculate RT60 annotation's using schroder curve equation for t:20 dB , the median of all the 5 vp's is taken as the RT60 of the room for each octave band.


noisy_mixture=h5py.File('/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/generated_rirs_lwh_correct.hdf5','r')


def t60_impulse(raw_signal, fs, bands, rt='t20'):  # pylint: disable=too-many-locals
    """
    Reverberation time from a WAV impulse response.

    :param file_name: name of the WAV file containing the impulse response.
    :param bands: Octave or third bands as NumPy array.
    :param rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.
    :returns: Reverberation time :math:`T_{60}`

    """
    #fs, raw_signal = wavfile.read(file_name)
    band_type = _check_band_type(bands)

    if band_type == 'octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == 'third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
        factor = 6.0

    t60 = np.zeros(bands.size)

    for band in range(bands.size):
        # Filtering signal
        filtered_signal = bandpass(raw_signal, low[band], high[band], fs, order=8)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

        # Schroeder integration
        sch = np.cumsum(abs_signal[::-1]**2)[::-1]
        sch_db = 10.0 * np.log10(sch / np.max(sch))

        # Linear regression
        sch_init = sch_db[np.abs(sch_db - init).argmin()]
        sch_end = sch_db[np.abs(sch_db - end).argmin()]

        init_sample = np.where(sch_db == sch_init)[0][0]
        end_sample = np.where(sch_db == sch_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = sch_db[init_sample:end_sample + 1]
        slope, intercept = stats.linregress(x, y)[0:2]

        # Reverberation time (T30, T20, T10 or EDT)
        db_regress_init = (init - intercept) / slope
        db_regress_end = (end - intercept) / slope
        t60[band] = factor * (db_regress_end - db_regress_init)
    return t60


'''
abc=np.load("sample_room_final_test.npy")
b=t60_impulse(abc[:,0,5],16000,np.array([125 * pow(2,a) for a in range(6)]))
print(b)
b=t60_impulse(abc[:,1,5],16000,np.array([125 * pow(2,a) for a in range(6)]))
print(b)

b=t60_impulse(abc[:,2,5],16000,np.array([125 * pow(2,a) for a in range(6)]))
print(b)
b=t60_impulse(abc[:,3,5],16000,np.array([125 * pow(2,a) for a in range(6)]))
print(b)
b=t60_impulse(abc[:,4,5],16000,np.array([125 * pow(2,a) for a in range(6)]))
print(b)
b=t60_impulse(abc[:,5,5],16000,np.array([125 * pow(2,a) for a in range(6)]))
print(b)
b=t60_impulse(abc[:,6,5],16000,np.array([125 * pow(2,a) for a in range(6)]))
print(b)
'''








rt60_file=h5py.File("rt60_anno_room_20_median.hdf5","w")
room_save=rt60_file.create_group("room_nos")
for r in range(20000):
    abc=np.zeros((1,6))
    print("room_no",r)
    for vp in range(5):
        rt60=t60_impulse(noisy_mixture['rirs']['room_'+str(r)]['rir'][vp*2,:],16000,np.array([125 * pow(2,a) for a in range(6)])).reshape(1,-1)
        abc=np.concatenate((abc,rt60),axis=0)
    room_id=room_save.create_group("room_"+str(r))
    room_id.create_dataset("rt60",(1,6),data=np.median(abc[1:,:],axis=0))





'''
def eyring60(room_d,f_bands,ab_coeff):
    #length,width,height
    #2*(length*width(Floor,celling),width*height,length*height)
    room_s=[room_d[0]*room_d[1],room_d[1]*room_d[2],room_d[0],room_d[2]]
    t60=[]
    for j in range(6):

        band_ab_coeff=[ab_coeff[((x*6)+j)] for x in range(6)]

        alp_freq=(room_s[0]*band_ab_coeff[0]+room_s[0]*band_ab_coeff[5]+room_s[1]*band_ab_coeff[1]+room_s[1]*band_ab_coeff[2]+room_s[2]*band_ab_coeff[3]+room_s[2]*band_ab_coeff[4])/np.sum([2*room_d[0]*room_d[1],2*room_d[1]*room_d[2],2*room_d[0]*room_d[2]])
        surf_area=np.sum([2*room_d[0]*room_d[1],2*room_d[1]*room_d[2],2*room_d[0]*room_d[2]])
        t60_band= -0.163*((room_d[0]*room_d[1]*room_d[2])/(surf_area*(np.log(1-alp_freq))))
        t60.append(t60_band)
    return np.array(t60)

file_load=h5py.File('generated_rirs.hdf5','r')
#print(file_load['rirs']['room_1']['rir'][0,:][5000])
#plt.plot(np.arange(55125),file_load['rirs']['room_0']['rir'][0,:],linewidth=0.4)
#plt.show()
bands=np.array([125 * pow(2, a) for a in range(6)])
print(bands)
no_of_rooms=9
no_of_channels=10
big_eye=[]
big_avg_rt=[]
big_er_avg=[]
for r in range(no_of_rooms):
    t60_eye=eyring60(file_load['room_config']['room_'+str(r)]['dimension'][()],np.array([125 * pow(2, a) for a in range(6)]),file_load['room_config']['room_'+str(r)]['absorption'][()])
    #print("t60eye",t60_eye)
    er_avg=np.zeros(6)
    avg_rt=np.zeros(6)
    big_eye.append(t60_eye)
    for c in range(no_of_channels):
        write('example.wav',file_load.attrs['fs'],file_load['rirs']['room_'+str(r)]['rir'][c,:])
        t60_rir=t60_impulse('example.wav',bands)
        #print("t60rir",t60_rir)
        error=t60_eye-t60_rir
        avg_rt=avg_rt+t60_rir
        er_avg=er_avg+error

    big_avg_rt.append(avg_rt/10)
    big_er_avg.append(er_avg/10)



fig = plt.figure()
gs = gridspec.GridSpec(2, 2)
ax = plt.subplot(gs[0, 0])
ax1=plt.subplot(gs[0,1])
ax2=plt.subplot(gs[1,0])
ax3=plt.subplot(gs[1,1])

ax.plot(np.arange(6), big_eye[0], marker='o', c='r',label='Eyet60')
ax.plot(np.arange(6), big_er_avg[0], marker='o', c='b',label='Error Avg')
ax.plot(np.arange(6), big_avg_rt[0], marker='o', c='g',label='RT60')

ax.set_title('Room 0')
ax.set_xticklabels([0,125,250,500,1000,2000,4000])

ax1.plot(np.arange(6), big_eye[1], marker='o', c='r',label='Eyet60')
ax1.plot(np.arange(6), big_er_avg[1], marker='o', c='b',label='Error Avg')
ax1.plot(np.arange(6), big_avg_rt[1], marker='o', c='g',label='RT60')
ax1.set_title('Room 1')
ax1.set_xticklabels([0,125,250,500,1000,2000,4000])
ax2.plot(np.arange(6), big_eye[2], marker='o', c='r',label='Eyet60')
ax2.plot(np.arange(6), big_er_avg[2], marker='o', c='b',label='Error Avg')
ax2.plot(np.arange(6), big_avg_rt[2], marker='o', c='g',label='RT60')
ax2.set_title('Room 2')
ax2.set_xticklabels([0,125,250,500,1000,2000,4000])
ax3.plot(np.arange(6), big_eye[3], marker='o', c='r',label='Eyet60')
ax3.plot(np.arange(6), big_er_avg[3], marker='o', c='b',label='Error Avg')
ax3.plot(np.arange(6), big_avg_rt[3], marker='o', c='g',label='RT60')
ax3.set_title('Room 3')
ax3.set_xticklabels([0,125,250,500,1000,2000,4000])
fig.add_subplot(ax)
fig.add_subplot(ax1)
fig.add_subplot(ax2)
fig.add_subplot(ax3)

fig.legend(['Eyet60', 'Error Avg', 'RT60'], loc='upper left')
plt.show()
'''
