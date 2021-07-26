#Calculate and Store the following annotations from the noisy mixtures original HDF5 file : mean surface absorption coeffcients,surface and volume.
import h5py
import numpy as np


abc=h5py.File("/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/generated_rirs_lwh_correct.hdf5","r") 
mean_absorp_surface=h5py.File("absorption_surface_calcul.hdf5","w") #Create New File 
room_save=mean_absorp_surface.create_group("room_nos")
for r in range(20000):
    #Retrive the absorption coeffcients every room consist of 6 abs val/wall 
    absorption_coeff=abc['room_config']['room_'+str(r)]['absorption'][()].reshape(6,6)   
    dimension=abc['room_config']['room_'+str(r)]['dimension'][()] #Retrive the dimension 
    lw=dimension[0]*dimension[1]
    wh=dimension[1]*dimension[2]
    lh=dimension[0]*dimension[2]
    surface_area=2*(lw+wh+lh) # Cal Surface area for specfic room 
    volume=dimension[0]*dimension[1]*dimension[2] # Cal Volume for specfic room 
    print("room no ",r)    
    val_absorp=[]
    #Calculate weighted mean absorption coefficent for every room and for every frequency band.
    for i in range(6):
        absorp_band=absorption_coeff[:,i] 
        mean_band=absorp_band[0]*lw+absorp_band[1]*wh+absorp_band[2]*lh+absorp_band[3]*lw+absorp_band[4]*wh+absorp_band[5]*lh
        mean_absorp_coeff_band=mean_band/surface_area
        val_absorp.append(mean_absorp_coeff_band)

    room_id=room_save.create_group("room_"+str(r))
    room_id.create_dataset("absorption",6,data=np.array(val_absorp)) #Save frequency band dependent mean absorption coeff 
    room_id.create_dataset("surface_area",1,data=surface_area) #Save surface area 
    room_id.create_dataset("volume",1,data=volume) #Save volume 

