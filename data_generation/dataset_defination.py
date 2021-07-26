import numpy as np
import yaml
import random
import h5py
from scipy.spatial import distance
from tqdm import tqdm 

#Generate room configuration.
#These room configurations is used by pysofamyroom to generate RIR's.

class util_room:
    def height_width_length(self):
        #Room Dimension Range: Length (3,10), Width(3,10), Height(2.5,4)
        values = [round(random.uniform(3,12),1),round(random.uniform(3,12),1), round(random.uniform(2.5, 4.5), 1)]
        return values

    def get_diffusion_coeff(self):
        #Diffusion Coeff Range: [0.2,1]
        coeff = round(random.uniform(0.2, 1), 2)
        return [coeff for x in range(36)]

    def get_absorption_coeff(self):
        #Coin-flip to get the different absorption profiles

        walls_profile = random.randint(0, 1)
        celling_profile = random.randint(0, 1)
        floor_profile = random.randint(0, 1)

        if walls_profile:
            wall_value = round(random.uniform(0.01, 0.12), 2)
            four_wall_freq = [wall_value for x in range(6 * 4)]

        elif walls_profile == 0:
            wall_value_freq = [round(random.uniform(0.01, 0.50), 2), round(random.uniform(0.01, 0.50), 2),
                               round(random.uniform(0.01, 0.30), 2), round(random.uniform(0.01, 0.12), 2),
                               round(random.uniform(0.01, 0.12), 2), round(random.uniform(0.01, 0.12), 2)]
            four_wall_freq = ([] + wall_value_freq) * 4

        if celling_profile:
            celling_value = round(random.uniform(0.01, 0.12), 2)
            celling_freq = [celling_value for x in range(6)]

        elif celling_profile == 0:
            celling_freq = [round(random.uniform(0.01, 0.70), 2), round(random.uniform(0.15, 1.00), 2),
                            round(random.uniform(0.40, 1.00), 2), round(random.uniform(0.40, 1.00), 2),
                            round(random.uniform(0.40, 1.00), 2), round(random.uniform(0.30, 1.00), 2)]

        if floor_profile:
            floor_value = round(random.uniform(0.01, 0.12), 2)
            floor_freq = [floor_value for x in range(6)]

        elif floor_profile == 0:
            floor_freq = [round(random.uniform(0.01, 0.20), 2), round(random.uniform(0.01, 0.30), 2),
                          round(random.uniform(0.05, 0.50), 2), round(random.uniform(0.15, 0.60), 2),
                          round(random.uniform(0.25, 0.75), 2), round(random.uniform(0.30, 0.80), 2)]

        return celling_freq + four_wall_freq + floor_freq


#Generate receivers position

class util_receiver():
    def __init__(self):
        self.dic_receiver={}

    def mic_defination_array(self,rotation_matrix,barycenter):
        distance_bw_mics=22.5 # Distance between mics in the array
        distance_wrt_bc=(distance_bw_mics/2)/100 # Distance wrt to the barycenter

        mic_1=np.matmul(rotation_matrix,np.array([distance_wrt_bc,0,0])) # Multiply with the rotation matrix to get the translated microphone coordinates
        mic_2=np.matmul(rotation_matrix,np.array([-distance_wrt_bc,0,0]))

        return mic_1+barycenter,mic_2+barycenter # Add those with the barycenter coordinates to get the rotated microphone coordiantes wrt barycenter.

    def generate_receivers_rooms(self,room_dimension,different_no_receivers,saftey_distance,room_id):
        a=different_no_receivers
        mic_pos_1_arr=np.empty((a,3)) # 5 Viewpoints per room so 5,3 matrix
        mic_pos_2_arr = np.empty((a, 3))
        li_bc=np.empty((a,3)) # list for the bary center
        li_ypr=np.empty((a,3)) # list for the ypr of the barycenter

        for x in range(different_no_receivers):

            #Random barycenter coordinates selection , within the saftey distance from the walls of the room

            barycenter=np.array([round(random.uniform(0.3,(room_dimension[0]-saftey_distance)),3),round(random.uniform(0.3,(room_dimension[1]-saftey_distance)),3),round(random.uniform(0.3,(room_dimension[2]-saftey_distance)),3)])

            barycenter_ypr=[random.randint(0,360),random.randint(0,360),random.randint(0,360)]
            y,p,r=barycenter_ypr[0],barycenter_ypr[1],barycenter_ypr[2]
            rotation_mat_1=np.array([np.cos(np.pi*(y)/180)*np.cos(np.pi*(p)/180),
                                     np.cos(np.pi*(y)/180)*np.sin(np.pi*(p)/180)*np.sin(np.pi*(r)/180)-np.sin(np.pi*(y)/180)*np.cos(np.pi*(r)/180),
                                     np.cos(np.pi * (y) / 180) * np.sin(np.pi * (p) / 180) * np.cos(np.pi * (r) / 180) + np.sin(np.pi * (y) / 180) * np.sin(np.pi * (r) / 180)
                                    ])
            rotation_mat_2=np.array([np.sin(np.pi*(y)/180)*np.cos(np.pi*(p)/180),
                                     np.sin(np.pi*(y)/180)*np.sin(np.pi*(p)/180)*np.sin(np.pi*(r)/180)+np.cos(np.pi*(y)/180)*np.cos(np.pi*(r)/180),
                                     np.sin(np.pi * (y) / 180) * np.sin(np.pi * (p) / 180) * np.cos(np.pi * (r) / 180) - np.cos(np.pi * (y) / 180) * np.sin(np.pi * (r) / 180)
                                    ])
            rotation_mat_3=np.array([ -np.sin(np.pi*(p)/180),
                                     np.cos(np.pi*(p)/180)*np.sin(np.pi*(r)/180),
                                     np.cos(np.pi*(p)/180)*np.cos(np.pi*(r)/180)
                                     ])
            rotation_mat=np.array([rotation_mat_1,rotation_mat_2,rotation_mat_3]) #3*3 rotation matrice.

            mic_pos_1,mic_pos_2=self.mic_defination_array(rotation_mat,barycenter)
            mic_pos_1_arr[x,:]=mic_pos_1
            mic_pos_2_arr[x,:]=mic_pos_2

            li_bc[x,:]=barycenter
            li_ypr[x,:]=barycenter_ypr

        return  mic_pos_1_arr,mic_pos_2_arr,li_bc,li_ypr




#Generate real and fake sources for the simulation.

class util_source():
    def __init__(self):
        self.description=['omnidirectional','cardioid']
    def generate_source_room(self,room_dimension,different_no_sources,saftey_distance,barycenter):
        #Random source in the room
        ll_source=np.empty((different_no_sources,3)) #5 viewpoints per room so 5*3 matrice
        ypr_source=np.empty((different_no_sources,3))

        description_source=[]

        x_safteydistance = room_dimension[0] - saftey_distance

        y_safteydistance= room_dimension[1] - saftey_distance

        z_safteydistance=room_dimension[2] - saftey_distance

        for i in range(different_no_sources):
            tmp_cord = np.array([round(random.uniform(0.3, x_safteydistance), 3),
                                 round(random.uniform(0.3, y_safteydistance), 3),
                                 round(random.uniform(0.3, z_safteydistance), 3)])
            tmp_distance=distance.euclidean(tmp_cord,barycenter[i,:])

            # until the distance between the source and the barycenter is greater than 0.3, we keep on finding the new coordinates for the source.

            while tmp_distance < 0.3 :
                tmp_cord = np.array([round(random.uniform(0.3, x_safteydistance), 3),
                                     round(random.uniform(0.3, y_safteydistance), 3),
                                     round(random.uniform(0.3, z_safteydistance), 3)])
                tmp_distance = distance.euclidean(tmp_cord, barycenter[i, :])

            ll_source[i, :] = tmp_cord
            ypr_source[i,:] = [random.randint(-180, 180), 0, 0]
            description_source.append(random.sample(self.description,1))

        return ll_source,ypr_source,description_source

    def fake_source_rooom(self,room_dimension,different_no_sources,saftey_distance,barycenter):
        #Generate fake source for same reciver positions, basically will be used to calculate the late reverberant RIR's which will be help us in diffuse noise model.

        ll_source = np.empty((different_no_sources, 3))

        ypr_source = np.empty((different_no_sources, 3))

        description_source = []

        x_safteydistance = room_dimension[0] - saftey_distance

        y_safteydistance = room_dimension[1] - saftey_distance

        z_safteydistance = room_dimension[2] - saftey_distance
        for i in range(different_no_sources):
            tmp_cord = np.array([round(random.uniform(0.3, x_safteydistance), 3),
                                 round(random.uniform(0.3, y_safteydistance), 3),
                                 round(random.uniform(0.3, z_safteydistance), 3)])
            tmp_distance=distance.euclidean(tmp_cord,barycenter[i,:])

            while tmp_distance < 0.5 :
                tmp_cord = np.array([round(random.uniform(0.3, x_safteydistance), 3),
                                     round(random.uniform(0.3, y_safteydistance), 3),
                                     round(random.uniform(0.3, z_safteydistance), 3)])
                tmp_distance = distance.euclidean(tmp_cord, barycenter[i, :])

            ll_source[i,:] = tmp_cord
            ypr_source[i,:] = [random.randint(-180, 180), 0, 0]
            description_source.append(random.sample(self.description,1))

        return ll_source,ypr_source,description_source


class conf_files(util_room,util_receiver):

    def __init__(self,number_of_rooms,name_of_the_dataset):
        self.number_rooms=number_of_rooms
        self.receiver_file=util_receiver()
        self.source_file=util_source()
        self.util_room=util_room()
        self.params_file()
        self.room_file()

    def params_file(self):
        dict_file = {'simulation_params':
            {
                'fs': 48000,
                'responseduration': 1.25,
                'bandsperoctave': 1,
                'referencefrequency': 125,
                'airabsorption': True,
                'distanceattenuation': True,
                'subsampleaccuracy': False,
                'highpasscutoff': 0,
                'verbose': True,
                'simulatespecular': True,
                'reflectionorder': [10, 10, 10],
                'simulatediffuse': True,
                'numberofrays': 2000,
                'diffusetimestep': 0.01,
                'rayenergyfloordB': -80.0,
                'uncorrelatednoise': True
            }
        }
        with open('conf_sim_params.yml', 'w') as file:
            documents = yaml.dump(dict_file, file)


    def room_file(self):
        dict_file = {}
        dict_file_receiver={}
        dict_file_source={}
        dict_file_noise_source={}
        humidity = 0.42
        temprature = 20.0
        reference_freq = 125
        no_of_reciver_per_room=5
        saftey_distance=0.7
        for x in tqdm(range(self.number_rooms)):


            return_dimension = self.util_room.height_width_length()
            dict_file['room_' + str(x)] = {'surface': {}}
            dict_file['room_' + str(x)]['dimension'] = return_dimension
            dict_file['room_' + str(x)]['humidity'] = humidity
            dict_file['room_' + str(x)]['temperature'] = temprature
            dict_file['room_' + str(x)]['surface']['frequency'] = [reference_freq * pow(2, a) for a in range(6)]
            dict_file['room_' + str(x)]['surface']['absorption'] = self.util_room.get_absorption_coeff()
            dict_file['room_' + str(x)]['surface']['diffusion'] = self.util_room.get_diffusion_coeff()


            li_rec_1,li_rec_2,li_bc,li_ypr=self.receiver_file.generate_receivers_rooms(return_dimension,no_of_reciver_per_room,saftey_distance,'room_' + str(x))
            li_sc,li_sc_ypr,li_sc_ds=self.source_file.generate_source_room(return_dimension,no_of_reciver_per_room,saftey_distance,li_bc)
            li_nsc,li_nsc_ypr,li_nsc_ds=self.source_file.fake_source_rooom(return_dimension,no_of_reciver_per_room,saftey_distance,li_bc)
            dict_file_receiver['room_' + str(x)] = {'barycenter':li_bc.tolist(),'barycenter_ypr':li_ypr.tolist(),'mic_pos_1':li_rec_1.tolist(),'mic_pos_2':li_rec_2.tolist()}

            dict_file_source['room_' + str(x)]={'source_pos':li_sc.tolist(),'source_ypr':li_sc_ypr.tolist(),'source_description':li_sc_ds}
            dict_file_noise_source['room_'+str(x)]={'source_pos':li_nsc.tolist(),'source_ypr':li_nsc_ypr.tolist(),'source_description':li_nsc_ds}


        with open('conf_room_setup_2.yml', 'w') as file:
            documents = yaml.dump(dict_file, file)
        with open('conf_receivers_2.yml','w') as file_1:
            documents = yaml.dump(dict_file_receiver,file_1)
        with open('conf_source_2.yml','w') as file_2:
            documents = yaml.dump(dict_file_source,file_2)
        with open('conf_noise_source_2.yml','w') as file_3:
            documents = yaml.dump(dict_file_noise_source, file_3)


conf_files(20000,"test")















#Extra Chunk of code comments
'''
More microphone on the array mic_defination_array
mic_3=np.matmul(rotation_matrix,np.array([(distance_wrt_bc+distance_wrt_bc),0,0]))
mic_4=np.matmul(rotation_matrix,np.array([-(distance_wrt_bc+distance_wrt_bc),0,0]))
mic_5 = np.matmul(rotation_matrix, np.array([(distance_wrt_bc)*3, 0, 0]))
mic_6 = np.matmul(rotation_matrix, np.array([-(distance_wrt_bc)*3, 0, 0]))
mic_7 = np.matmul(rotation_matrix, np.array([(distance_wrt_bc)*4, 0, 0]))
mic_8 = np.matmul(rotation_matrix, np.array([-(distance_wrt_bc)*4, 0, 0]))
mic_9 = np.matmul(rotation_matrix, np.array([(distance_wrt_bc)*5, 0, 0]))
mic_10 = np.matmul(rotation_matrix, np.array([-(distance_wrt_bc)*5, 0, 0]))
mic_3+barycenter,mic_4+barycenter,mic_5+barycenter,mic_6+barycenter,mic_7+barycenter,mic_8+barycenter,mic_9+barycenter,mic_10+barycenter

In generate receiver_room 
 
mic_pos_3_arr = np.empty((a,3))
mic_pos_4_arr = np.empty((a,3))
mic_pos_5_arr = np.empty((a, 3))
mic_pos_6_arr = np.empty((a, 3))
mic_pos_7_arr = np.empty((a, 3))
mic_pos_8_arr = np.empty((a, 3))
mic_pos_9_arr = np.empty((a, 3))
mic_pos_10_arr = np.empty((a, 3))
        
mic_pos_3,mic_pos_4,mic_pos_5,mic_pos_6,mic_pos_7,mic_pos_8,mic_pos_9,mic_pos_10
mic_pos_3_arr[x,:]=mic_pos_3
mic_pos_4_arr[x,:]=mic_pos_4
mic_pos_5_arr[x, :] = mic_pos_5
mic_pos_6_arr[x, :] = mic_pos_6
mic_pos_7_arr[x, :] = mic_pos_7
mic_pos_8_arr[x, :] = mic_pos_8
mic_pos_9_arr[x, :] = mic_pos_9
mic_pos_10_arr[x, :] = mic_pos_10
mic_pos_3_arr,mic_pos_4_arr,mic_pos_5_arr,mic_pos_6_arr,mic_pos_7_arr,mic_pos_8_arr,mic_pos_9_arr,mic_pos_10_arr
            
#self.dic_receiver[room_id]={'barycenter':[barycenter,barycenter_ypr],'ll_receiver':li_receiver}
#self.dic_receiver[room_id]['ll_receiver']=[mic_pos_1,mic_pos_2]
#dic_receiver[room_id]['orientation']=[[0,0,0],[0,0,0]]
#dic_receiver[room_id]['description']='omnidirectional'

HDF5 file 
#self.file=h5py.File(name_of_the_dataset+".hdf5",'w')
 self.file.attrs['fs'] = 44100
        self.file.attrs['response_duration'] = 1.25
        self.file.attrs['bandsperoctave'] = 1
        self.file.attrs['referencefrequency'] = 125
        self.file.attrs['airabsorption'] = True
        self.file.attrs['distanceattenuation'] = True
        self.file.attrs['subsampleaccuracy'] = False
        self.file.attrs['highpasscutoff'] = 0
        self.file.attrs['verbose'] = True
        self.file.attrs['simulatespecular'] = True
        self.file.attrs['reflectionorder'] = [10,10,10]
        self.file.attrs['simulatediffuse'] = True
        self.file.attrs['numberofrays'] = 2000
        self.file.attrs['diffusetimestep'] = 0.01
        self.file.attrs['rayenergyfloordB'] = -80.0
        self.file.attrs['uncorrelatednoise'] = True

 #room=self.file.create_group("room_config")
#receiver=self.file.create_group("receiver_config")
#source=self.file.create_group("source_config")
#room.attrs['saftey_distance'] = 0.3
#room.attrs['reference_freq'] = 125
#room.attrs['no_of_receiver_per_room'] = 5
  
room_id=room.create_group('room_' + str(x))
receiver_id=receiver.create_group('room_'+str(x))
source_id=source.create_group('room_'+str(x))
 
room_id.create_dataset('dimension',3,data=return_dimension)
room_id.create_dataset('humidity',1,data=humidity)
room_id.create_dataset('temprature',1,data=temprature)
room_id.create_dataset('frequency',6,data=[reference_freq * pow(2, a) for a in range(6)])
room_id.create_dataset('absorption',36,data=self.util_room.get_absorption_coeff())
room_id.create_dataset('diffusion',36,data=self.util_room.get_diffusion_coeff())
         
li_rec_3,li_rec_4,li_rec_5,li_rec_6,li_rec_7,li_rec_8,li_rec_9,li_rec_10,
            ,'mic_pos_3':li_rec_3.tolist(),
'mic_pos_4':li_rec_4.tolist(),'mic_pos_5':li_rec_5.tolist(),'mic_pos_6':li_rec_6.tolist(),'mic_pos_7':li_rec_7.tolist(), ,'mic_pos_8':li_rec_8.tolist(),'mic_pos_9':li_rec_9.tolist(),'mic_pos_10':li_rec_10.tolist()
            
 
receiver_id.create_dataset('barycenter',(a,3),data=li_bc)
receiver_id.create_dataset('barycenter_ypr', (a, 3), data=li_ypr)
receiver_id.create_dataset('mic_pos_1',(a,3),data=li_rec_1)
receiver_id.create_dataset('mic_pos_2',(a,3),data=li_rec_2)
receiver_id.create_dataset('mic_pos_3', (a, 3), data=li_rec_3)
receiver_id.create_dataset('mic_pos_4', (a, 3), data=li_rec_4)
source_id.create_dataset('source_pos',(a,3),data=li_sc)
source_id.create_dataset('source_ypr', (a, 3), data=li_sc_ypr)
source_id.create_dataset('source_description', (a, 1), data=li_sc_ds)
#dict_file['room_'+str(x)]=room_dict['room_'+str(x)]


'''
