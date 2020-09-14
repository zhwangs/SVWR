
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
radius=K[2]*rad_test
 
Rec=s.generate_rec_mesh(theta, phi, radius)
 
x=Rec[0]
y=Rec[1]
z=Rec[2]
 
radius_tensor=torch.tensor(radius, requires_grad=True)
theta_tensor=torch.tensor(theta, requires_grad=True)
phi_tensor=torch.tensor(phi, requires_grad=True)

# %%
Mu_vacuum = 1.25663706212e-6    # vacuum magnetic permeability
Eps_vacuum = 8.8541878128e-12   # vacuum electric permittivity
ZVAC = 376.73031346177          # impedance of free space
c = 3e8#299792458                   # the speed of light, m/s
pi = np.pi
 
Eps=Eps_vacuum
Mu = Mu_vacuum

polar_psi=pi/4#np.pi/4#np.pi/2#np.pi/3
polar_chi=0#np.pi/3

E0=1
#2*pi/span 
#x=np.linspace(0,10,3)
#y=np.linspace(0,2*np.pi,10)
#z=np.linspace(0,2*np.pi,10)

x=Rec[0,:] 
y=Rec[1,:] 
z=Rec[2,:] 
kk=0.05
K=kk*np.array([1,0,-1])
K_mag=np.sqrt(np.sum(K**2))
unit_K=K/K_mag

lbd=2*np.pi/K_mag
K2=plane_wave_dir(polar_psi,polar_chi,E0,K)
#RR=angle_from_ek(k)
KKK=plane_wave_dir(polar_psi,polar_chi,E0,K,direction=True)
K_a=np.sqrt(np.sum(KKK[0]**2))
unit_K_a=KKK[0]/K_a
K_b=np.sqrt(np.sum(KKK[1]**2))
unit_K_b=KKK[1]/K_b

K_matrix=np.array([unit_K_a,unit_K_b])
Omega=c*K_mag
RB=plane_wave_grid(x,y,z,K2,K,omega=Omega,t0=0)
E_x_r=RB[0] # x-y plane 
E_y_r=RB[1]
E_z_r=RB[2]


 # %%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D



 
'''
EEx=E_y_r.real.reshape(-1,) 
xxx=x.reshape(-1,)
plt.scatter(xxx,EEx)
kr=kk*rad_test
print(kr)
'''

# %%

cmap = cm.get_cmap("coolwarm")
fig = plt.figure()
fig.suptitle('Plane_wave Generation:' +'\n ratio: wavelength/radius= '+str(round(lbd/radius.max(),2)), fontsize=30)
ax_1 = fig.add_subplot(221, projection='3d')
ax_2 = fig.add_subplot(222, projection='3d')
ax_3 = fig.add_subplot(223, projection='3d')
ax_4 = fig.add_subplot(224, projection='3d')


fig.set_figheight(15)
fig.set_figwidth(15)

ax_1.view_init(elev=15, azim=45)
ax_1.set_title('Field Strength in x direction (E_x)', fontsize=20)
ax_1.set_xlabel('x (meter)', fontsize=15)
ax_1.set_ylabel('y (meter)', fontsize=15)
ax_1.set_zlabel('z (meter)', fontsize=15)

normx = Normalize()
colorsx =normx(E_x_r.real)
ax_1.plot_surface(x, y, z, linewidth=0, facecolors=cmap(colorsx), shade=False, alpha=1)

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
colorsy =normy(E_y_r.real)
ax_2.plot_surface(x, y, z, linewidth=0, facecolors=cmap(colorsy), shade=False, alpha=1)

ax_3.view_init(elev=15, azim=45)
ax_3.set_title('Field Strength in z direction (E_z)', fontsize=20)
ax_3.set_xlabel('x (meter)', fontsize=15)
ax_3.set_ylabel('y (meter)', fontsize=15)
ax_3.set_zlabel('z (meter)', fontsize=15)
normz = Normalize()
colorsz =normz(E_z_r.real)
ax_3.plot_surface(x, y, z, linewidth=0, facecolors=cmap(colorsz), shade=False, alpha=1)

#ax_4.set_axis_off()
#fig.colorbar(mappable, shrink=10, aspect=10,label='Field Strength E_x')

dir_x=np.zeros(2)
dir_y=np.zeros(2)
dir_z=np.zeros(2)

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


# %%

# %%
