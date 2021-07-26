import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
#Do the train test split of the rooms present in the noisy mixture dataset, primarly used by the dataloader.
tmp_=[]
for i in range(19999): #It starts from room 1 and goes until room 19999, so basically train-test split misses two room's : room_0, room_20000
    for j in range(5):
        tmp_.append(("room_"+str(i+1),j+1))


total_samples=19999*5

train,val,test=(total_samples*80)/100,(total_samples*10)/100,(total_samples*10)/100

#train_ar,val_ar,test_ar=random_split(ad,(int(train),int(val),int(test)))
#No room's a repeated in any of the train, dev and test set's

train_ar=tmp_[:int(train)]
print(len(train_ar),train_ar[-1])

val_ar=tmp_[int(train):int(train)+int(val)]
print(len(val_ar),val_ar[0],val_ar[-1])

test_ar=tmp_[int(train)+int(val):int(train)+int(val)+int(test)]
print(len(test_ar),test_ar[0],test_ar[-1])

np.save("train_random_ar.npy",train_ar,allow_pickle=True)
np.save("val_random_ar.npy",val_ar)
np.save("test_random_ar.npy",test_ar)


