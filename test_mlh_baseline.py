import torch
from mlh_baseline import Model,pred,NLLloss

net=Model()
chkp=torch.load("/home/psrivastava/baseline/scripts/pre_processing/mlh_tas_save_best_sh.pt")
optimizer=chkp['optimizer_dic']
model_dict=chkp['model_dict']
net.eval()

test_data=dl.binuaral_dataset('test_random_ar.npy','/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_lwh_.hdf5','/home/psrivastava/baseline/sofamyroom_2/conf_room_setup.yml')

#train_dl=DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
#val_dl=DataLoader(val_data, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
test_dl=DataLoader(test_data,batch_size=128,shuffle=True,num_workers=0,drop_last=True)



#epch=chkp['epoch']
#loss=chkp['loss']
#print(epoch,loss)


