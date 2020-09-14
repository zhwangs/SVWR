
#%%
import os
import sys
# get path of this script
#path_this_script = os.getcwd()
path_this_script = os.path.realpath(__file__)

# add the ./src/ path to the search path
path_this_script_splitted = os.path.split(path_this_script)
this_script_filename = path_this_script_splitted[1]
path_this_script_splitted = os.path.split(path_this_script_splitted[0])
path_to_src = os.path.join(path_this_script_splitted[0], 'test/Exp')
sys.path.append(path_to_src)  # I could have used sys.path.append('../src/'), but it didn't work with the debugger

from WD_fuc import * 


rad_test=100#2000




# %%


import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.special import spherical_jn, spherical_yn

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

# get path of this script
#path_this_script = os.getcwd()
path_this_script = os.path.realpath(__file__)

# add the ./src/ path to the search path
path_this_script_splitted = os.path.split(path_this_script)
this_script_filename = path_this_script_splitted[1]
path_this_script_splitted = os.path.split(path_this_script_splitted[0])
path_this_script_splitted = os.path.split(path_this_script_splitted[0])

path_to_src = os.path.join(path_this_script_splitted[0], 'src')
sys.path.append(path_to_src)   
path_to_cache = os.path.join(path_this_script_splitted[0], 'cache')
from Mesh_inter import *

#%%  create mesh 


gmsh_file = path_this_script_splitted[0]+'/cache/sample_mesh/mesh_5.msh'

# Class 
s=Mesh_inter(gmsh_file)
# interpolation (triangle mesh)
Data_g=s.mesh_interpo(20)
Data_x=Data_g[0]
Data_y=Data_g[1]
Data_z=Data_g[2]



# Orignial mesh

Org_data=Data_g[3]
 
# Create theta phi mesh 
s.angle_radius_mesh()
Est=s.G_quad_mesh_N_esti()
k_theta=Est[0] # theta unique 
k_phi=Est[2] # phi unique 
k_theta_mean=Est[1] # theta mean 
k_phi_mean=Est[3] # phi mean 

s_theta=np.linspace(0,1,len(k_theta)-1)
s_phi=np.linspace(0,1,len(k_phi)-1)
 
K=s.L_grid(20)
theta=K[3]
phi=K[4]
radius=K[2]

Rec=s.generate_rec_mesh(K[3], K[4], K[2])



#%% Mesh 
K_mag=0.01
K_mesh=K[0]
Geo_mesh=Rec
geo_mag=100
Geo_mesh_polar=np.array([K[2],K[3],K[4]])
K_mesh_size=int(np.sqrt(K_mesh.shape[1]))
K_mesh_recover=K_mesh.reshape(3,K_mesh_size,K_mesh_size)

K_x=K_mesh_recover[0,:,:]
K_y=K_mesh_recover[1,:,:]
K_z=K_mesh_recover[2,:,:]





#%%
A=Emap_K(Geo_mesh,geo_mag,Geo_mesh_polar,K_mesh,K_mag,n_max=5,polar_psi=0,polar_chi=0,E0=1)

#%% recover K mesh 3D
lbd=2*pi/K_mesh_recover.max()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

n_max=6

fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)

ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=15, azim=-90)#np.pi/4)

x_r=K_x
y_r=K_y
z_r=K_z



E_mag_r=A[1].real#np.sqrt(np.abs(E_x_r)**2+np.abs(E_y_r)**2+np.abs(E_z_r)**2)
norm = Normalize()
colors =(E_mag_r) # auto-adjust true radius into [0,1] for color mapping

cmap = cm.get_cmap("coolwarm")
ax.plot_surface(K_x, K_y, K_z, linewidth=0, facecolors=cmap(colors), shade=False, alpha=1)

# the surface is not mappable, we need to handle the colorbar manually
mappable = cm.ScalarMappable(cmap=cmap)
mappable.set_array(E_mag_r)
fig.colorbar(mappable, shrink=0.2, aspect=5,label='Field strength E')
 # auto-adjust true radius into [0,1] for color mapping

#ax.quiver(x_r, y_r, z_r, M_mn_x.real, M_mn_y.real, M_mn_z.real, length=0.1*x.max(),linewidths=0.6*x.max())

ax.set_title('Plane_wave rec \n n_max='+str(n_max), fontsize=30)
ax.set_xlabel('x (meter)', fontsize=20)
ax.set_ylabel('y (meter)', fontsize=20)
ax.set_zlabel('z (meter)', fontsize=20)





# %%
EEx=E_mag_r.reshape(-1,) 
xxx=K_x.reshape(-1,)
plt.scatter(xxx,EEx)
# %%
cmap = cm.get_cmap("coolwarm")
fig = plt.figure()
fig.suptitle('Plane_wave Generation:' +'\n ratio: wavelength/radius= '+str(round(lbd/radius.max(),2)), fontsize=30)
ax_1 = fig.add_subplot(221, projection='3d')
ax_2 = fig.add_subplot(222, projection='3d')
ax_3 = fig.add_subplot(223, projection='3d')
#ax_4 = fig.add_subplot(224, projection='3d')


fig.set_figheight(15)
fig.set_figwidth(15)



ax_1.view_init(elev=15, azim=45)
ax_1.set_title('Field Strength in x direction (E_x)', fontsize=20)
ax_1.set_xlabel('x (meter)', fontsize=15)
ax_1.set_ylabel('y (meter)', fontsize=15)
ax_1.set_zlabel('z (meter)', fontsize=15)

normx = Normalize()
colorsx =(A[0].real)
ax_1.plot_surface(K_x, K_y, K_z, linewidth=0, facecolors=cmap(colorsx), shade=False, alpha=1)
mappable = cm.ScalarMappable(cmap=cmap)
mappable.set_array(colorsx)
fig.colorbar(mappable, shrink=2, aspect=5,label='MSE',ax=ax_1)

# the surface is not mappable, we need to handle the colorbar manually
#mappable = cm.ScalarMappable(cmap=cmap)
#mappable.set_array(E_mag_r)
#fig.colorbar(mappable, shrink=0.2, aspect=5,label='Field Strength E_x')



ax_2.view_init(elev=15, azim=45)
ax_2.set_title('Field Strength in y direction (E_y)', fontsize=20)
ax_2.set_xlabel('x (meter)', fontsize=15)
ax_2.set_ylabel('y (meter)', fontsize=15)
ax_2.set_zlabel('z (meter)', fontsize=15)
normy = Normalize()
colorsy =(A[1].real)
ax_2.plot_surface(K_x, K_y, K_z, linewidth=0, facecolors=cmap(colorsy), shade=False, alpha=1)
mappable = cm.ScalarMappable(cmap=cmap)
mappable.set_array(colorsy)
fig.colorbar(mappable, shrink=2, aspect=5,label='MSE',ax=ax_2)

ax_3.view_init(elev=15, azim=45)
ax_3.set_title('Field Strength in z direction (E_z)', fontsize=20)
ax_3.set_xlabel('x (meter)', fontsize=15)
ax_3.set_ylabel('y (meter)', fontsize=15)
ax_3.set_zlabel('z (meter)', fontsize=15)
normz = Normalize()
colorsz =(A[2].real)
ax_3.plot_surface(K_x, K_y, K_z, linewidth=0, facecolors=cmap(colorsz), shade=False, alpha=1)
mappable = cm.ScalarMappable(cmap=cmap)
mappable.set_array(colorsz)
fig.colorbar(mappable, shrink=2, aspect=5,label='MSE',ax=ax_3)
#ax_4.set_axis_off()
#fig.colorbar(mappable, shrink=10, aspect=10,label='Field Strength E_x')

dir_x=np.zeros(2)
dir_y=np.zeros(2)
dir_z=np.zeros(2)

'''
ax_4.view_init(elev=15, azim=30)
ax_4.set_title('Direction'+str(np.round(unit_K,2))+' \n polarization'+str(np.round(K2.real,2)), fontsize=20)
ax_4.set_xlabel('x (meter)', fontsize=15)
ax_4.set_ylabel('y (meter)', fontsize=15)
ax_4.set_zlabel('z (meter)', fontsize=15)
ax_4.set_xlim3d(-1, 1)
ax_4.set_ylim3d(-1,1)
ax_4.set_zlim3d(-1,1)

ax_4.quiver(dir_x, dir_y, dir_z, K_matrix[:,0], K_matrix[:,1], K_matrix[:,2], length = 0.7, normalize = True,arrow_length_ratio=0.1,linewidths=2)
ax_4.quiver(0, 0, 0, unit_K[0],unit_K[1],unit_K[2], length = 1, normalize = True,colors='r',linewidths=4)

polar=K2.real
ax_4.quiver(0, 0, 0, polar[0],polar[1],polar[2], length = 0.8, normalize = True,colors='g',linewidths=4)

'''

# %%
