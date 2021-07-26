import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection







from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy as np
from itertools import product, combinations



#ax.plot3D((0,0),(0,0),(0,0), color="b")
#ax.plot3D((3,4),(0,4),(0,0),color="b")
#ax.plot3D((0,0),(0,0),(0,5),color="b")

'''
[0,0,0]
[x,0,0]
[0,y,0]
[0,0,z]
[x,y,0]
[0,y,z]
[x,0,z]
[x,y,z]

[0,0,0]
[4,0,0]
[0,4,0]
[0,0,4]
[4,4,0]
[4,0,4]
[0,4,4]
[4,4,4]

[0,0,0]
[3,0,0]
[3,4,0]
[0,4,0]
[0,0,0]


        ax.plot3D([0,x,0,0,x,0,x,x],[0,0,y,0,y,y,0,y],[0,0,0,z,0,z,z,z],c='r')

        v = np.array([[0, 0, 0], [x, 0, 0], [0, y, 0],  [0, 0, z], [x, y, 0],[0,y,z],[x,0,z],[x,y,z]])
        ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])

        print(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7])
        # generate list of sides' polygons of our pyramid
        verts = [ [v[0],v[1],v[2]],[v[4],v[1],v[2]],[v[0],v[3],v[2]],[v[5],v[3],v[2]],[v[0],v[1],v[3]]]

        # plot sides
        ax.add_collection3d(Poly3DCollection(verts,
         facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))



[3,0,5]





'''

import yaml








with open('conf_receivers.yml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    bc = yaml.load(file, Loader=yaml.FullLoader)

with open('conf_room_setup.yml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    ab = yaml.load(file, Loader=yaml.FullLoader)
with open('conf_source.yml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    cd = yaml.load(file, Loader=yaml.FullLoader)
for i in range(10):
    for j in range(4):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        '''
        print(bc['room_'+str(i)]['barycenter'][j])
        print("-------------------------------")
        print(ab['room_'+str(i)]['dimension'])
        print('-------------------------------')
        print(cd['room_'+str(i)]['source_pos'][j])
        print(bc['room_' + str(i)]['mic_pos_1'][j])
        print(bc['room_' + str(i)]['mic_pos_2'][j])
        print(bc['room_' + str(i)]['mic_pos_3'][j])
        print(bc['room_' + str(i)]['mic_pos_4'][j])
        print("=======================================")
        '''

        x,y,z=ab['room_'+str(i)]['dimension']

        bc_x,bc_y,bc_z=bc['room_'+str(i)]['barycenter'][j]
        mc1_x,mc1_y,mc1_z=bc['room_'+str(i)]['mic_pos_1'][j]
        mc2_x,mc2_y,mc2_z=bc['room_'+str(i)]['mic_pos_2'][j]
        mc3_x,mc3_y,mc3_z=bc['room_'+str(i)]['mic_pos_3'][j]
        mc4_x,mc4_y,mc4_z=bc['room_'+str(i)]['mic_pos_4'][j]

        mc5_x, mc5_y, mc5_z = bc['room_' + str(i)]['mic_pos_5'][j]
        mc6_x, mc6_y, mc6_z = bc['room_' + str(i)]['mic_pos_6'][j]
        mc7_x, mc7_y, mc7_z = bc['room_' + str(i)]['mic_pos_7'][j]
        mc8_x, mc8_y, mc8_z = bc['room_' + str(i)]['mic_pos_8'][j]
        mc9_x, mc9_y, mc9_z = bc['room_' + str(i)]['mic_pos_9'][j]
        mc10_x, mc10_y, mc10_z = bc['room_' + str(i)]['mic_pos_10'][j]

        print(distance.euclidean(bc['room_'+str(i)]['mic_pos_1'][j],bc['room_'+str(i)]['mic_pos_2'][j]))
        print(distance.euclidean(bc['room_' + str(i)]['mic_pos_3'][j], bc['room_' + str(i)]['mic_pos_4'][j]))

        print(distance.euclidean(bc['room_' + str(i)]['mic_pos_5'][j], bc['room_' + str(i)]['mic_pos_6'][j]))
        print(distance.euclidean(bc['room_' + str(i)]['mic_pos_7'][j], bc['room_' + str(i)]['mic_pos_8'][j]))

        sc_x,sc_y,sc_z=cd['room_'+str(i)]['source_pos'][j]


        ax.plot3D([0,x,x,0,0,0,x,x,0,0,0,0,x,x,x,x],
          [0,0,y,y,0,0,0,y,y,y,0,y,y,y,0,0],
          [0,0,0,0,0,z,z,z,z,0,z,z,0,z,0,z])




        ax.scatter(bc_x,bc_y,bc_z,c='b')
        #ax.text(bc_x, bc_y, bc_z, '%s' % ('bc'), size=10, zorder=1, color='k')
        ax.scatter(mc1_x,mc1_y,mc1_z,c='g')
        #ax.text(mc1_x, mc1_y, mc1_z, '%s' % ('mic_1'), size=10, zorder=1, color='k')
        ax.scatter(mc2_x,mc2_y,mc2_z,c='g')
        #ax.text(mc2_x, mc2_y, mc2_z, '%s' % ('mic_2'), size=10, zorder=1, color='k')
        ax.scatter(mc3_x,mc3_y,mc3_z,c='g')
        #ax.text(mc2_x, mc2_y, mc2_z, '%s' % ('mic_3'), size=10, zorder=1, color='k')
        ax.scatter(mc4_x,mc4_y,mc4_z,c='g')
        ax.scatter(mc5_x, mc5_y, mc5_z, c='g')
        ax.scatter(mc6_x, mc6_y, mc6_z, c='g')
        ax.scatter(mc7_x, mc7_y, mc7_z, c='g')
        ax.scatter(mc8_x, mc8_y, mc8_z, c='g')
        ax.scatter(mc9_x, mc9_y, mc9_z, c='g')
        ax.scatter(mc10_x,mc10_y,mc10_z,c='g')

        #ax.text(mc2_x, mc2_y, mc2_z, '%s' % ('mic_4'), size=10, zorder=1, color='k')




        ax.scatter(sc_x,sc_y,sc_z,c='y')
        ax.text(sc_x, sc_y, sc_z, '%s' % ('src'), size=10, zorder=1, color='k')

        plt.show()