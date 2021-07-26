import numpy as np
import h5py 

#This script calculates std and variance of the rt60 values/freq band in the training set.

abc=np.load("train_random_ar.npy")

#Annoted labels of rt60,abs,vol,surface area.
dc=h5py.File("rt60_anno_room_20_median.hdf5","r")
kc=h5py.File("absorption_surface_calcul.hdf5","r")
#print(dc["room_nos"]["room_1"][()].shape)

ab=0
abl=np.zeros((1,6))
for a in range(len(abc)):
    room=abc[a][0]
    #room_no=int(room.split("_")[1])
    vp=int(abc[a][1])
    print(room,vp)
    tot=dc["room_nos"][room]['rt60'][()].reshape(1,6)
    abl=np.concatenate((abl,tot),axis=0) #Concatenate rt60 per freq band on axis=0
    '''
    if room_no > ab:
        print(room)
        abso=kc["room_nos"][room]['absorption'][()]
        #print(abso)
        vol=kc["room_nos"][room]['volume'][()]
        #print(vol)
        surf=kc["room_nos"][room]['surface_area'][()]
        #print(surf)
        tot_=np.concatenate((abso,vol,surf),axis=0).reshape(1,8)
        abl=np.concatenate((abl,tot_),axis=0)
        ab=room_no
    '''

    
print(abl.shape)
print(np.std(abl[1:,0]),np.var(abl[1:,0]))
print(np.std(abl[1:,1]),np.var(abl[1:,1]))
print(np.std(abl[1:,2]),np.var(abl[1:,2]))
print(np.std(abl[1:,3]),np.var(abl[1:,3]))
print(np.std(abl[1:,4]),np.var(abl[1:,4]))
print(np.std(abl[1:,5]),np.var(abl[1:,5]))
#print(np.std(abl[1:,6]),np.var(abl[1:,6]))
#print(np.std(abl[1:,7]),np.var(abl[1:,7]))

