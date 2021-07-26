import h5py 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()
abc=h5py.File("absorption_surface_calcul.hdf5","r")
surface_area=[]
hz_125=[]
hz_250=[]
hz_500=[]
hz_1000=[]
hz_2000=[]
hz_4000=[]
for r in range(20000):
    surface_area.append(abc['room_nos']['room_'+str(r)]['surface_area'])
    hz_125.append(abc['room_nos']['room_'+str(r)]['absorption'][0])
    hz_250.append(abc['room_nos']['room_'+str(r)]['absorption'][1])
    hz_500.append(abc['room_nos']['room_'+str(r)]['absorption'][2])
    hz_1000.append(abc['room_nos']['room_'+str(r)]['absorption'][3])
    hz_2000.append(abc['room_nos']['room_'+str(r)]['absorption'][4])
    hz_4000.append(abc['room_nos']['room_'+str(r)]['absorption'][5])

plt.hist(np.array(hz_125),color="yellow")
plt.savefig("plt_absorption_125.png")
plt.clf()
plt.hist(np.array(hz_250),color="yellow")
plt.savefig("plt_absorption_250.png")
plt.clf()
plt.hist(np.array(hz_500),color="yellow")
plt.savefig("plt_absorption_500.png")
plt.clf()
plt.hist(np.array(hz_1000),color="yellow")
plt.savefig("plt_absorption_1000.png")
plt.clf()
plt.hist(np.array(hz_2000),color="yellow")
plt.savefig("plt_absorption_2000.png")
plt.clf()
plt.hist(np.array(hz_4000),color="yellow")
plt.savefig("plt_absorption_4000.png")
