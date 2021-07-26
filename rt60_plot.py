import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

sns.set_theme()

abc=h5py.File("rt60_anno_room_20.hdf5","r")
cdf=h5py.File("rt60_anno_room_10.hdf5","r")
kcc=h5py.File("rt60_anno_room.hdf5","r")
rt_125=np.zeros((1))
rt_250=np.zeros((1))
rt_500=np.zeros((1))
rt_1000=np.zeros((1))
rt_2000=np.zeros((1))
rt_4000=np.zeros((1))
rt_125_10=np.zeros((1))
rt_250_10=np.zeros((1))
rt_500_10=np.zeros((1))
rt_1000_10=np.zeros((1))
rt_2000_10=np.zeros((1))
rt_4000_10=np.zeros((1))
rt_125_20=np.zeros((1))
rt_250_20=np.zeros((1))
rt_500_20=np.zeros((1))
rt_1000_20=np.zeros((1))
rt_2000_20=np.zeros((1))
rt_4000_20=np.zeros((1))

for r in range(20000):
  rt_125=np.concatenate((rt_125,kcc['room_nos']['room_'+str(r)]['rt60'][:,0]),axis=0)
  rt_125_10=np.concatenate((rt_125_10,cdf['room_nos']['room_'+str(r)]['rt60'][:,0]),axis=0)
  rt_125_20=np.concatenate((rt_125_20,abc['room_nos']['room_'+str(r)]['rt60'][:,0]),axis=0)
  
  rt_250=np.concatenate((rt_250,kcc['room_nos']['room_'+str(r)]['rt60'][:,1]),axis=0)
  rt_250_10=np.concatenate((rt_250_10,cdf['room_nos']['room_'+str(r)]['rt60'][:,1]),axis=0)
  rt_250_20=np.concatenate((rt_250_20,abc['room_nos']['room_'+str(r)]['rt60'][:,1]),axis=0)
  
  rt_500=np.concatenate((rt_500,kcc['room_nos']['room_'+str(r)]['rt60'][:,2]),axis=0)
  rt_500_10=np.concatenate((rt_500_10,cdf['room_nos']['room_'+str(r)]['rt60'][:,2]),axis=0)
  rt_500_20=np.concatenate((rt_500_20,abc['room_nos']['room_'+str(r)]['rt60'][:,2]),axis=0)
  
  rt_1000=np.concatenate((rt_1000,kcc['room_nos']['room_'+str(r)]['rt60'][:,3]),axis=0)
  rt_1000_10=np.concatenate((rt_1000_10,cdf['room_nos']['room_'+str(r)]['rt60'][:,3]),axis=0)
  rt_1000_20=np.concatenate((rt_1000_20,abc['room_nos']['room_'+str(r)]['rt60'][:,3]),axis=0)
  
  rt_2000=np.concatenate((rt_2000,kcc['room_nos']['room_'+str(r)]['rt60'][:,4]),axis=0)
  rt_2000_10=np.concatenate((rt_2000_10,cdf['room_nos']['room_'+str(r)]['rt60'][:,4]),axis=0)
  rt_2000_20=np.concatenate((rt_2000_20,abc['room_nos']['room_'+str(r)]['rt60'][:,4]),axis=0)
  
  rt_4000=np.concatenate((rt_4000,kcc['room_nos']['room_'+str(r)]['rt60'][:,5]),axis=0)
  rt_4000_10=np.concatenate((rt_4000_10,cdf['room_nos']['room_'+str(r)]['rt60'][:,5]),axis=0)
  rt_4000_20=np.concatenate((rt_4000_20,abc['room_nos']['room_'+str(r)]['rt60'][:,5]),axis=0)
  
  
  '''
  rt_1000=np.concatenate((rt_1000,abc['room_nos']['room_'+str(r)]['rt60'][:,3]),axis=0)
  rt_2000=np.concatenate((rt_2000,abc['room_nos']['room_'+str(r)]['rt60'][:,4]),axis=0)
  rt_4000=np.concatenate((rt_4000,abc['room_nos']['room_'+str(r)]['rt60'][:,5]),axis=0)
  '''


n=np.arange(6,step=0.1)


fig, axs = plt.subplots(3, 1)
fig.tight_layout(pad=2.0)
axs[0].hist(rt_125,bins=n)
axs[0].set_title("t 30db")
axs[1].hist(rt_125_10,bins=n,color='orange')
axs[1].set_title("t 10db")
axs[2].hist(rt_125_20,bins=n,color='green')
axs[2].set_title("t 20db")

#plt.title("RT 60 Box Plot")
plt.xlabel("Time In Seconds")
plt.savefig("rt60_hist_125.png",bbox_inches='tight')

plt.clf()


fig, axs = plt.subplots(3, 1)
fig.tight_layout(pad=2.0)

axs[0].hist(rt_250,bins=n)
axs[0].set_title("t 30db")
axs[1].hist(rt_250_10,bins=n,color='orange')
axs[1].set_title("t 10db")
axs[2].hist(rt_250_20,bins=n,color='green')
axs[2].set_title("t 20db")

#plt.title("RT 60 Box Plot")
plt.xlabel("Time In Seconds")
plt.savefig("rt60_hist_250.png",bbox_inches='tight')

plt.clf()


fig, axs = plt.subplots(3, 1)
fig.tight_layout(pad=2.0)
axs[0].hist(rt_500,bins=n)
axs[0].set_title("t 30db")
axs[1].hist(rt_500_10,bins=n,color='orange')
axs[1].set_title("t 10db")
axs[2].hist(rt_500_20,bins=n,color='green')
axs[2].set_title("t 20db")

#plt.title("RT 60 Box Plot")
plt.xlabel("Time In Seconds")
plt.savefig("rt60_hist_500.png",bbox_inches='tight')

plt.clf()


fig, axs = plt.subplots(3, 1)
fig.tight_layout(pad=2.0)
axs[0].hist(rt_1000,bins=n)
axs[0].set_title("t 30db")
axs[1].hist(rt_1000_10,bins=n,color='orange')
axs[1].set_title("t 10db")
axs[2].hist(rt_1000_20,bins=n,color='green')
axs[2].set_title("t 20db")

#plt.title("RT 60 Box Plot")
plt.xlabel("Time In Seconds")
plt.savefig("rt60_hist_1000.png",bbox_inches='tight')

plt.clf()


fig, axs = plt.subplots(3, 1)
fig.tight_layout(pad=2.0)
axs[0].hist(rt_2000,bins=n)
axs[0].set_title("t 30db")
axs[1].hist(rt_2000_10,bins=n,color='orange')
axs[1].set_title("t 10db")
axs[2].hist(rt_2000_20,bins=n,color='green')
axs[2].set_title("t 20db")

#plt.title("RT 60 Box Plot")
plt.xlabel("Time In Seconds")
plt.savefig("rt60_hist_2000.png",bbox_inches='tight')

plt.clf()

fig, axs = plt.subplots(3, 1)
fig.tight_layout(pad=2.0)
axs[0].hist(rt_4000,bins=n)
axs[0].set_title("t 30db")
axs[1].hist(rt_4000_10,bins=n,color='orange')
axs[1].set_title("t 10db")
axs[2].hist(rt_4000_20,bins=n,color='green')
axs[2].set_title("t 20db")

#plt.title("RT 60 Box Plot")
plt.xlabel("Time In Seconds")
plt.savefig("rt60_hist_4000.png",bbox_inches='tight')




'''
plt.title("RT 60 Box Plot")
plt.xlabel("Time In Seconds")
plt.hist(rt_250,bins=n)

plt.savefig("rt60_hist_20db_250.png",bbox_inches='tight')
plt.clf()

plt.title("RT 60 Box Plot")
plt.xlabel("Time In Seconds")
plt.hist(rt_500,bins=n)
plt.savefig("rt60_hist_20db_500.png",bbox_inches='tight')
plt.clf()
#plt.boxplot([rt_125,rt_250,rt_500,rt_1000,rt_2000,rt_4000],showmeans=True,vert=True)
plt.title("RT 60 Box Plot")
plt.xlabel("Time In Seconds")
plt.hist(rt_1000,bins=n)
plt.savefig("rt60_hist_20db_1000.png",bbox_inches='tight')
#plt.xticks([1,2,3,4,5,6],('125','250','500','1000','2000','4000'))
plt.clf()

plt.title("RT 60 Box Plot")
plt.xlabel("Time In Seconds")
plt.hist(rt_2000,bins=n)
plt.savefig("rt60_hist_20db_2000.png",bbox_inches='tight')
plt.clf()

plt.title("RT 60 Box Plot")
plt.xlabel("Time In Seconds")
plt.hist(rt_4000,bins=n)
plt.savefig("rt60_hist_20db_4000.png",bbox_inches='tight')
'''
