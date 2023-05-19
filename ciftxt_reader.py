#!/usr/bin/env python
# coding: utf-8

# # CODICE STRUTTURA

# In[20]:


# %pip install numpy
# %pip install  pymatgen
# %pip install matplotlib
# %pip install scipy
# %pip install ipympl


# ## Parte 1

# In[21]:


# get_ipython().run_line_magic('matplotlib', 'widget')
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.chemenv.utils import coordination_geometry_utils as cgu
import matplotlib.pyplot as plt
import scipy 

Material = "Si"
# Load the crystal structure of silicon
path_cif = Material+".cif"
structure = Structure.from_file(path_cif)
# print(structure)

a_lattice, b_lattice, c_lattice = [structure.lattice.abc[i] for i in range(0,len(structure.lattice.abc))]
alfa_lattice, beta_lattice, gamma_lattice = structure.lattice.angles


# Replicate the unit cell to obtain a larger cell
supercell = structure.make_supercell([2, 2, 2])

# Extract the positions of the atoms
positions = structure.cart_coords

# Round the coordinates of each position to two decimal places
for position in positions:
    for coordinate in position:
        coordinate = round(coordinate,2)
    #print(position)
#print(positions)

# Plot the positions of the atoms in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:,0], positions[:,1], positions[:,2])
ax.set_title(Material+" Structure")
ax.set_xlabel('X, $\AA$')
ax.set_ylabel('Y, $\AA$')
ax.set_zlabel('Z, $\AA$')
#plt.show()()


# In[22]:


import pymatgen.analysis.chemenv.utils.coordination_geometry_utils as pmg

asse = r"1 1 1"
print (Material)
print ("asse ", np.fromstring(asse, dtype=int, sep=" "))

prova = np.fromstring(asse, dtype=int, sep=" ")

a_p = prova[0]
b_p = prova[1]
c_p = prova[2]
d_p = 0
# d_p = - (sum(prova))*3 -c_p  #* a_lattice

plane = pmg.Plane.from_coefficients(a_p,b_p,c_p,d_p)
# print(plane)
# print(sum(prova))
# print (a_p,b_p,c_p,d_p)

# print(plane.coefficients.round(2))
a_p = plane._coefficients[0]
b_p = plane._coefficients[1]
c_p = plane._coefficients[2]
d_p = plane._coefficients[3]

# print ("absolute ", a_p.round(2),b_p.round(2),c_p.round(2),d_p.round(2))
# print ("coeff/a ",(a_p/a_lattice).round(2),(b_p/a_lattice).round(2),(c_p/a_lattice).round(2),(d_p/a_lattice).round(2))
# print ("a/coeff ",(a_lattice/a_p).round(2),(a_lattice/b_p).round(2),(a_lattice/c_p).round(2),(a_lattice/d_p).round(2))



# Generiamo un array di valori x, y e z
if c_p == 0 :
    y = np.linspace(0, 9, 50)
    x = np.linspace(0, 9, 50)
    if a_p != 0 and b_p ==0: #100
        Z, Y = np.meshgrid(x, y)
        X = (0*Z + 0*Y)
    if a_p == 0 and b_p !=0: #010
        X, Z = np.meshgrid(x, y)
        Y = (0*X + 0*Z) 
    if a_p != 0 and b_p !=0: #110
        Y, Z = np.meshgrid(x, y)
        X = -Y 
else: #111
    y = np.linspace(0, 9, 50)
    x = np.linspace(0, 9, 50)
    X, Y = np.meshgrid(x, y)
    Z = (-a_p*X - b_p*Y - d_p)/c_p

# Creiamo un'istanza dell'oggetto Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plottiamo il piano come una superficie
ax.plot_surface(X, Y, Z)
ax.scatter(positions[:,0], positions[:,1], positions[:,2])
# Aggiungiamo le etichette degli assi
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Visualizziamo il grafico
# plt.show()




# In[ ]:





# In[23]:


#selection of the axis for plane/axis projection
# asse = r"1 1 1"
# print ("asse ", np.fromstring(asse, dtype=int, sep=" "))

## Creates the axis vector to project all the points on the plane (normalizing it)##
norm_vector = np.fromstring(asse, dtype=int, sep=" ")
norma = np.linalg.norm(norm_vector)
norm_vector = norm_vector/norma

vector_projections = []
for position in positions:
    pr_scalar = np.dot(position, norm_vector) * norm_vector
    #print(" position", position)
    #print(" pr_scalar", pr_scalar)
    #projection = pr_scalar/norma
    vector_projection = np.array(position) - (pr_scalar) # projection *
    vector_projections.append(vector_projection)

#print(vector_projections)

soglia = 1e-10

# Iterate over the vector projections and convert them all to np.arrays
for prj in vector_projections:
    # Set values smaller than the threshold to zero
    prj[np.abs(prj)< soglia] = 0.0
    #print(prj)

# Convert the vector projections to a numpy array
ndarray_projs = np.array(vector_projections)

# Plot the vector projections in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.scatter(ndarray_projs[:,0], ndarray_projs[:,1], ndarray_projs[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# plt.show()



# In[24]:


# Define three base vectors for x, y and z
x_base = np.array([1, 0, 0])
y_base = np.array([0, 1, 0])
z_base = np.array([0, 0, 1])

# Calculate the rotation axis for a given normal vector
rotation_axis = np.cross(norm_vector, z_base)
# print(rotation_axis)

# Calculate the rotation angle for a given normal vector
angle = np.arccos(np.dot(norm_vector,z_base)/(np.sqrt(norm_vector[0]**2 + norm_vector[1]**2 + norm_vector[2]**2)*np.sqrt(z_base[0]**2 + z_base[1]**2 + z_base[2]**2)))
# print(np.degrees(angle))

# Define the rotation matrix for a given rotation axis and angle
c = np.cos(angle)
def rotation_matrix(axis, theta):
    """
    Returns the rotation matrix for a rotation around a specified axis by a specified angle in radians 
    using Rodrigues' formula.
    """
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis * np.sin(theta/2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

# Calculate the rotated projection vectors
R = rotation_matrix(rotation_axis, angle)
r_ndarray_projs = []
for p in ndarray_projs:
    r_ndarray_proj = np.dot(R, p)
    #print("rotated",r_ndarray_proj)
    r_ndarray_projs.append(r_ndarray_proj)

# Apply a threshold to each of the rotated projection vectors
for prj in r_ndarray_projs:
    prj[np.abs(prj) < soglia] = 0.0

# Plot the rotated projection vectors in 3D space
r_ndarray_projs = np.array(r_ndarray_projs)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(r_ndarray_projs[:,0], r_ndarray_projs[:,1], r_ndarray_projs[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.savefig(Material+'_<'+asse.replace(" ","")+">"+"_plane_projection.pdf")


# In[25]:


x_positions = []
y_positions = []
atom_positions = []
for a in range(0,len(r_ndarray_projs)):
    atom_positions.append((round(r_ndarray_projs[a][0],3),round(r_ndarray_projs[a][1],3)))
    x_positions.append(round(r_ndarray_projs[a][0],3))
    y_positions.append(round(r_ndarray_projs[a][1],3))
#print(atom_positions)
atom_positions = list(set(atom_positions))
#print("at_pos", atom_positions)
# print("Number of projected atoms:",len(atom_positions))

posizioni_proiettate = []

for position in positions:
    p_scalare = np.dot(position, norm_vector)

    # calculates the length of the projection vector
    norma_asse = np.linalg.norm(norm_vector)

    # calcola la proiezione della posizione lungo l'asse
    d_proiezione = round(p_scalare / norma_asse, 3)

    posizioni_proiettate.append(d_proiezione)

posizioni_proiettate = list(set(posizioni_proiettate))
#print(posizioni_proiettate)

planar_distances = [round(posizioni_proiettate[n+1]-posizioni_proiettate[n],2) for n in range(0, len(posizioni_proiettate)-1)]
d_pl = []
if planar_distances.count(planar_distances[0]) == len(planar_distances):
    d_pl.append(planar_distances[0])
else:
    d_pl = list(set(planar_distances))
d_plm = sum(d_pl)/len(d_pl) 
# print("d_pl = ",d_pl, 'd_plm = ', d_plm)

################################## d_at CALCULATION ##################################

def punto_retta_scalare(v1,r):
    r_norm = np.linalg.norm(r)
    v1_norm = np.linalg.norm(v1)
    # print (round(v1_norm,2))
    # print (round(r_norm,2))
    pr_scalar=np.dot(v1, r)
    # print (pr_scalar, " -- ", v1_norm*r_norm)
    if abs(pr_scalar - v1_norm*r_norm)<0.01:
        return True
    else:
        return False
    
pos_ax =[]
for position in positions:
    test2 = punto_retta_scalare(position,norm_vector*norma)
    if test2 == True:
        #print (position)
        pos_ax.append(position) 


for at in pos_ax: #to make all np.arrays
    # print (atoms)
    at[np.abs(at)< soglia] = 0.0
    

pos_ax=np.sort(np.array(pos_ax), axis=0)
axial_projections = [np.sqrt((at[0])**2+(at[1])**2+(at[2])**2) for at in pos_ax] # è un nome a caso

axial_distances = [round(axial_projections[n+1]-axial_projections[n],2) for n in range(0, len(axial_projections)-1)]
d_at = []
if axial_distances.count(axial_distances[0]) == len(axial_distances):
    d_at.append(axial_distances[0])
else:
    d_at = list(set(axial_distances))
d_atm = sum(d_at)/len(d_at) 
# print("d_at = ",d_at, 'd_atm = ', d_atm)

################################## d_s CALCULATION ##################################
import math 

atom_dist_projection = []
for i in range(0,len(atom_positions)):
    for j in range(i+1,len(atom_positions)):
        p2 = atom_positions[i]
        p1 = atom_positions[j]
        dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) # + (p2[2]-p1[2])**2)
        atom_dist_projection.append(dist)
        # print(dist)
d_s = min(atom_dist_projection)
# print ('d_s',round(d_s,3))


'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos_ax[:,0], pos_ax[:,1], pos_ax[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()'''


# In[26]:


# plot_in_plane = []
# for position in positions:
#     # if (np.abs(np.dot(plane.normal_vector, position) + plane._coefficients[3])) <= 0.1:
#     if plane.is_in_plane(position, 0.5):
#         plot_in_plane.append(position)
#         # print(position)     


def point_in_plane(asse,points,dist,tollerance): #axis as a vector (fromstring)
    a_plan = asse[0]
    b_plan = asse[1]
    c_plan = asse[2]
    # d_plan = 0
    d_plan = - (sum(asse))*3 -c_plan -dist #* a_lattice
    plane = pmg.Plane.from_coefficients(a_plan,b_plan,c_plan,d_plan)
    # print(plane)
    # print(sum(prova))
    # print (a_p,b_p,c_p,d_p) 
    point_in_plane =[]
    for point in points:
        if plane.is_in_plane(point, tollerance):
            point_in_plane.append(point)
    # print(len(point_in_plane), ' points found at d= ', dist, 'A')
    return point_in_plane

toll = min(d_pl)/2
axis = np.fromstring(asse, dtype=int, sep=" ")
# plot_in_plane = np.array(point_in_plane(axis,positions,0,toll))
# plot_dt = np.array(point_in_plane(axis,positions,d_pl[0],toll))
# plot_dt2 = np.array(point_in_plane(axis,positions,d_pl[-1],toll))

plot_values=[]
labels=[]
for at in positions: #to make all np.arrays
    # print (atoms)
    at[np.abs(at)< soglia] = 0.0

for i in range (0,10):
    value = point_in_plane(axis,positions,i*sum(d_pl)/len(d_pl),toll)
    if len(value) > 0:
        plot_values.append(np.array(value))
        labels.append(i*sum(d_pl)/len(d_pl))

# print (plot_in_plane)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plottiamo il piano come una superficie
# ax.plot_surface(X, Y, Z)
# for plot in plot_values:
for index,plot in enumerate(plot_values):
    # print(' i ', index)
    ax.scatter(plot[:,0], plot[:,1], plot[:,2], label = labels[index])
# print(plot_values[1])
# ax.scatter(plot_in_plane[:,0], plot_in_plane[:,1], plot_in_plane[:,2])
# ax.scatter(plot_dt[:,0], plot_dt[:,1], plot_dt[:,2])
# ax.scatter(plot_dt2[:,0], plot_dt2[:,1], plot_dt2[:,2])
# Aggiungiamo le etichette degli assi
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Visualizziamo il grafico
#plt.show()()



# In[27]:


# import math
# longest_list = max(plot_values, key=len)
# print('longest list len ',len(longest_list))

# # longest_list= sorted(longest_list, key = lambda p:longest_list[0])
# distances = []
# for i in range(0,len(longest_list)):
#     for j in range(i+1,len(longest_list)):
#         p2 = longest_list[i]
#         p1 = longest_list[j]
#         dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)
#         distances.append(dist)
#         # print(dist)
# d_s = min(distances)
# print ('d_s',round(d_s,3))


# In[28]:


#################################################################### Silicon
if "Si" in path_cif:
    coeff_a = [1.876, 2.617, 0.8604, 0.3903, 0.07769]
    coeff_b = [62.11, 18.68, 3.935, 0.7690, 0.0823]
    coeff_B = [b/(4*(np.pi)**(2)) for b in coeff_b]
    u = 0.075 #preso da sytov thesis
    N = 4.995E-2 #atomic concentration
    Z = 14
################################################################### Germanium
elif "Ge" in path_cif:
    coeff_a = [0.2259E+01, 0.2746E+01, 0.1533E+01, 0.6738E+00, 0.1563E+00]
    coeff_b = [0.5883E+02, 0.1588E+02, 0.2957E+01, 0.6679E+00, 0.7455E-01]
    coeff_B = [b/(4*(np.pi)**(2)) for b in coeff_b]
    u = 0.085 #from Baier Katkov
    N = 4.42E-2 #atomic concentration
    Z = 32
################################################################### Germanium
elif "W" in path_cif:
    coeff_a = [0.3467E-01, 0.1775E+00, 0.2189E+00, 0.1268E+00, -0.4894E-01]
    coeff_b = [0.1895, 0.3829E+01, 0.8004E+00, 0.1958E+00, 0.1958E+00]
    coeff_B = [b/(4*(np.pi)**(2)) for b in coeff_b]
    u = 0.050 #from Baier Katkov
    N = 0.168 #atomic concentration
    Z = 74
############################## COEFF MOLIERE
alfa = [0.1, 0.55, 0.35]
beta = [6.0, 1.2, 0.3]

z = 1
a_0 = scipy.constants.hbar/(scipy.constants.m_e*scipy.constants.c*scipy.constants.alpha)*10**10  #Angstrom
e = scipy.constants.e
a_TF = (9*np.pi**2/(128*Z))**(1/3)*a_0

def U_DT_2D(x, y, x_0, y_0):
    # d_atm = sum(d_at)/len(d_at) #mean of the atomic distances
    U1 = 2/d_atm*Z*z*np.sqrt(1/137)*1973.2**2/0.511E6
    k1 = [(coeff_a[i]/(coeff_B[i]+2*u**2))*np.exp(-((x-x_0)**2+(y-y_0)**2)/(coeff_B[i]+2*u**2)) for i in range(0,len(coeff_a))]
    k = sum(k1)
    return U1*k

def U_DT_axial(x, y, x_0, y_0):
    # d_atm = sum(d_at)/len(d_at) #mean of the atomic distances
    U1 = 2/d_atm*Z*z*(scipy.constants.hbar/scipy.constants.e)**2/0.511E6
    k1 = [(coeff_a[i]/(coeff_B[i]+2*u**2))*np.exp(-((x-x_0)**2+(y-y_0)**2)/(coeff_B[i]+2*u**2)) for i in range(0,len(coeff_a))]
    k = sum(k1)
    return U1*k

def U_DT_axial1(x, y, x_0, y_0):#Kyryllin
    # d_atm = sum(d_at)/len(d_at) #mean of the atomic distances
    U1 = 8*np.pi**2*(1973.2)**2/(0.511E6*d_atm)
    k1 = [(coeff_a[i]/(coeff_b[i]+8*np.pi**2*u**2))*np.exp(-4*np.pi**2*((x-x_0)**2+(y-y_0)**2)/(coeff_b[i]+8*np.pi**2*u**2)) for i in range(0,len(coeff_a))]
    k = sum(k1)
    return U1*k

def U_Lin_axial(x, y, x_0, y_0):
    # d_atm = sum(d_at)/len(d_at) #mean of the atomic distances
    U1 = Z*z*e**2/(4*np.pi*scipy.constants.epsilon_0*12*d_atm)/e*10**10*np.log(1+3*a_TF**2/((x-x_0)**2+(y-y_0)**2))
    #k1 = [(coeff_a[i]/(coeff_b[i]+8*np.pi**2*u**2))*np.exp(-4*np.pi**2*((x-x_0)**2+(y-y_0)**2)/(coeff_b[i]+8*np.pi**2*u**2)) for i in range(0,len(coeff_a))]
    #k = sum(k1)
    return U1

################ Make the Potential calculation
x = np.linspace(min(x_positions), max(x_positions), 1000)
y = np.linspace(min(y_positions), max(y_positions), 1000)
xx, yy = np.meshgrid(x, y, indexing="ij")

x_0 = [0] * 100
y_0 = [0] * 100
xx_0, yy_0 = np.meshgrid(x_0,y_0)

U_2D= np.zeros_like(xx)
for atom in atom_positions:
    U_2D += U_DT_axial1(xx,yy,atom[0],atom[1])

U_max, U_min = [], []
for U in U_2D:
    U_max.append(max(U))
    U_min.append(min(U))
    
U_top, U_bottom = max(U_max), min(U_min)
# print(U_top, U_bottom)

#U_2D -= U_bottom


# In[29]:


# U_max, U_min = [], []
# for U in U_2D:
#     U_max.append(max(U))
#     U_min.append(min(U))
    
# U_top, U_bottom = max(U_max), min(U_min)

#x_plot = np.linspace(min(x_positions)/2, max(x_positions)/2, 1000)
#y_plot = np.linspace(min(y_positions)/2, max(y_positions)/2, 1000)
#xx_plot, yy_plot = np.meshgrid(x_plot, y_plot)
i_max, i_low = 800, 200
# Make the graph
levels = np.linspace(U_bottom, U_top, 100)
fig_U_ax, ax=plt.subplots()
# plt.contourf(xx[i_low:i_max,i_low:i_max], yy[i_low:i_max,i_low:i_max], U_2D[i_low:i_max,i_low:i_max], levels, cmap='jet') #levels,
plt.contourf(xx, yy, U_2D, levels, cmap='jet') #levels,
plt.colorbar()
plt.title("U "+ Material + "<"+ asse +">") 
plt.xlabel('x, $\AA$')
plt.ylabel('y, $\AA$')
#plt.show()()
fig_U_ax.savefig(Material+'_<'+asse.replace(" ","")+">"+"_axial2D.pdf")


# In[30]:


from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig_U_ax_surf, ax = plt.subplots(subplot_kw={"projection": "3d"})


surf = ax.plot_surface(xx[i_low:i_max,i_low:i_max], yy[i_low:i_max,i_low:i_max], U_2D[i_low:i_max,i_low:i_max], cmap=cm.coolwarm, linewidth=0, antialiased=False, rcount=100, ccount=100) #cmap=cm.coolwarm,
#surf2 = ax.plot_surface(xx, yy, U_top*np.ones_like(xx) ,linewidth=0, antialiased=False)
fig_U_ax_surf.savefig(Material+'_<'+asse.replace(" ","")+">"+"_axial3D.pdf")


# In[31]:


def U_pl_DT(x): 
    # d_plm = sum(d_pl)/len(d_pl)
    U = 2*np.sqrt(np.pi)*N*d_plm*Z*z*1973.2**2/(0.511E6)*np.sqrt(1/137) #a_0*e**2=(hbar*c)^2/m_e
    k1 = [U*(coeff_a[i]/np.sqrt(coeff_B[i]+2*u**2))*np.exp(-x**2/(coeff_B[i]+2*u**2)) for i in range(0,len(coeff_a))]
    k = sum(k1)
    return k

#coeff_a = [2.1293, 2.5333, 0.8349, 0.3216]
#coeff_b = [57.7748, 6.4756, 2.8796, 0.3860]
# print(a_lattice)
def U_pl_DT2(x,i): 
    # d_plm = sum(d_pl)/len(d_pl)
    # d_atm = sum(d_at)/len(d_at)
    if i == 0:
        if ("Si" in Material or "Ge" in Material):
            if asse == '1 0 0':
                # print ('1 0 0 : ', asse)
                print (Material, asse)
                d_axis = a_lattice #100
                # print('igor ds = ', round((a_lattice**2/2)/d_axis,3), ' - ', round(d_s,3) , ' our d_s - ',round(a_lattice/2,3),' a/2')#100
                # print('igor ds = ', round((a_lattice*np.sqrt(2)/4),3), ' - ', round(d_s,3) , ' our d_s - ',round(a_lattice/2,3),' a/2')#100
            elif asse == '1 1 0':    
                # print ('1 1 0 : ',asse)
                print (Material, asse)
                d_axis = a_lattice*np.sqrt(2)/2 #110
                # print('igor ds = ', round((a_lattice**2*np.sqrt(2)/4)/d_axis,3), ' - ', round(d_s,3) , ' our d_s - ' ,round(a_lattice/2,3),' a/2')#110
            elif asse == '1 1 1':
                # print ('1 1 1 :',asse)
                print (Material, asse)
                d_axis = a_lattice*np.sqrt(3)/2 #111
                # print('igor ds = ', round((a_lattice**2*np.sqrt(3)/4)/d_axis,3), ' - ', round(d_s,3) , ' our d_s - ',round(a_lattice/2,3),' a/2')#111
                # print('igor ds = ', round((a_lattice*np.sqrt(6)/6),3), ' - ', round(d_s,3) , ' our d_s - ',round(a_lattice/2,3),' a/2')#111
            else:
                d_axis = float("NaN") 
            # print(i)
        else:
            d_axis = float("NaN")     
        # print ('igor d = ',round(d_axis,2), ' - ', d_atm , ' our d (d_atm)') #!! d_igor = d_atm_nostro = d_axis!!

    U = 4*(np.pi)**(3/2)*(1973.2)**2/(0.511E6*d_s*d_atm)
    # U = 4*(np.pi)**(3/2)*(1973.2)**2/(0.511E6*d_axis*d_s)
    k1 = [(coeff_a[i]/np.sqrt(coeff_b[i]+8*np.pi**2*u**2))*np.exp(-4*np.pi**2*x**2/(coeff_b[i]+8*np.pi**2*u**2)) for i in range(0,len(coeff_a))]
    k = sum(k1)

    # print('dds',d_s*d_axis,' vs ', a_lattice**2*np.sqrt(2)/4    
    return U*k
# print(U_pl_DT2(0))

x_DT = np.linspace(posizioni_proiettate[posizioni_proiettate.index(0.0)], posizioni_proiettate[posizioni_proiettate.index(0.0)+len(d_at)] , 1000)
posizioni_proiettate=[p-posizioni_proiettate[int(len(posizioni_proiettate)/2)] for p in posizioni_proiettate]
#print(posizioni_proiettate)
#U_harm = U_h_100(x_DT-d_pl/2)
'''#Moliere
U_Mol = [U_M(x_DT+n*d_pl) for n in range(-nel,nel)]
U_sum_Mol = sum(U_Mol)/e
U_sum_Mol = U_sum_Mol - min(U_sum_Mol)'''
#calculate for a number of planes
U_DT=[]
for n in range(0, len(posizioni_proiettate)):
    U_DT.append(U_pl_DT2(x_DT-posizioni_proiettate[n],n))
U_sum_DT = sum(U_DT) #/e
#subtraction of the minimum energy
U_sum_DT = [U_sum_DT1 - min(U_sum_DT) for U_sum_DT1 in U_sum_DT]

figure_U_pl, ax = plt.subplots(figsize=(7,5))

ax.plot(x_DT, U_sum_DT, label="DT")
ax.set_title("U "+ Material + "("+asse+")")
ax.set_ylabel("U, eV")
ax.set_xlabel("x, $\AA$")
ax.legend()
ax.grid()
#plt.show()()
figure_U_pl.savefig(Material+'_<'+asse.replace(" ","")+">"+"_planar.pdf")


# In[32]:


# print(max(U_sum_DT),min(U_sum_DT[0:150]))


# ## ELECTRIC FIELD AND ELECTRON DENSITY

# In[33]:


def gradient(x,y,x_0,y_0):
    d_atm = sum(d_at)/len(d_at)
    dU_dx = -2/d_atm*Z*z*e*a_0*sum([-2*(x-x_0)*coeff_a[i]/(coeff_B[i]+2*u**2)**2 * np.exp(-((x-x_0)**2+(y-y_0)**2)/(coeff_B[i]+2*u**2)) for i in range(len(coeff_a))])/e
    dU_dy = -2/d_atm*Z*z*e*a_0*sum([-2*(y-y_0)*coeff_a[i]/(coeff_B[i]+2*u**2)**2 * np.exp(-((x-x_0)**2+(y-y_0)**2)/(coeff_B[i]+2*u**2)) for i in range(len(coeff_a))])/e
    return dU_dx, dU_dy

E_tot= np.zeros_like(xx)
for atom in atom_positions:
    # dU_dx, dU_dy = gradient(xx,yy,atom[0],atom[1]) 
    #print(dU_dx, dU_dy)
    x0, y0 = atom[0], atom[1]
    dU_dx, dU_dy = gradient(xx, yy, x0, y0)
    
    d2U_dx2, d2U_dxdy = gradient(dU_dx, yy, x0, y0)
    d2U_dydx, d2U_dy2 = gradient(xx, dU_dy, x0, y0)

    hessian = np.array([[d2U_dx2, d2U_dxdy], [d2U_dydx, d2U_dy2]])
    eigenvalues = np.linalg.eigvals(hessian)
    
    # if (eigenvalues.all() > 0):
    #     print(f"({x0}, {y0}) is a local minimum") #, eigenvalues)
    #     # pass
    # elif (eigenvalues.all() < 0):
    #     print(f"({x0}, {y0}) is a local maximum") #, eigenvalues)
    #     # pass
    # else:
    #     print(f"({x0}, {y0}) is a saddle point") #, eigenvalues)
    E_tot += np.sqrt(dU_dx**2 + dU_dy**2)

    

E_max, E_min = [], []
for E in E_tot:
    E_max.append(max(E))
    E_min.append(min(E))
    
E_top, E_bottom = max(E_max), min(E_min)
# print(E_top, E_bottom)
# Make the graph
levels = np.linspace(E_bottom, E_top, 100)
fig_E_ax, ax=plt.subplots()
plt.contourf(-xx, -yy, E_tot, levels, cmap='jet') #levels,
plt.colorbar()
plt.title("E "+ Material + "<"+ asse +">") 
plt.xlabel('x, $\AA$')
plt.ylabel('y, $\AA$')
#plt.show()()
fig_E_ax.savefig(Material+'_<'+asse.replace(" ","")+">"+"_E_axial.pdf")


# In[34]:


def E_pl_DT_dx(x):
    d_plm = sum(d_pl)/len(d_pl)
    E = 2*np.sqrt(np.pi)*N*d_plm*Z*z*e*a_0/e
    k1 = [-(2*x*coeff_a[i]/(coeff_B[i]+2*u**2)**(3/2))*np.exp(-x**2/(coeff_B[i]+2*u**2)) for i in range(0,len(coeff_a))]
    k = sum(k1)
    return E*k

E_tot=[]
for n in range(0, len(posizioni_proiettate)):
    E_tot.append(E_pl_DT_dx(x_DT-posizioni_proiettate[n]))
E_tot = sum(E_tot)

fig_E_pl, ax = plt.subplots(figsize=(7,5))
ax.plot(x_DT, E_tot, label="DT")
ax.set_title("E "+ Material + "("+asse+")")
ax.set_ylabel("E, eV/$\AA$")
ax.set_xlabel("x, $\AA$")
ax.legend()
ax.grid()
#plt.show()()
fig_E_pl.savefig(Material+'_<'+asse.replace(" ","")+">"+"_E_planar.pdf")


# In[35]:


##Nuclear density
def n_nuc_pl(x):
    d_plm = sum(d_pl)/len(d_pl)
    dens = N*d_plm/(np.sqrt(2*np.pi*u**2))*np.exp(-(x**2/(2*u**2))) 
    return dens
# print(d_pl)
N_pl = []
for n in range(0, len(posizioni_proiettate)):
    N_pl.append(n_nuc_pl(x_DT-posizioni_proiettate[n])/1e-24) # to cm-3
n_pl_tot = sum(N_pl)
# print(np.shape(n_pl_tot))

fig_N_pl, ax = plt.subplots(figsize=(7,5))
ax.plot(x_DT, n_pl_tot, label="DT")
ax.set_title("Nuclear density "+ Material + "("+asse+")")
ax.set_ylabel("$n_N$, $cm^{-3}$")
ax.set_xlabel("x, $\AA$")
ax.legend()
ax.grid()
#plt.show()()
fig_N_pl.savefig(Material+'_<'+asse.replace(" ","")+">"+"_N_planar.pdf")


# In[36]:


##Nuclear density
def n_nuc_ax(xp,yp,x_0,y_0):
    d_atm = sum(d_at)/len(d_at)
    dens = (1/(2*np.pi*d_atm*u**2))*np.exp(-((xp-x_0)**2+(yp-y_0)**2)/(2*u**2))/1e-24 # to cm-3
    # print (x_0,y_0, dens)
    return dens
# print(d_at)

N_tot= np.zeros_like(xx)
for atom in atom_positions:
    N_tot += n_nuc_ax(xx,yy,atom[0],atom[1]) 
    #print(dU_dx, dU_dy)

N_max, N_min = [], []
for Na in N_tot:
    N_max.append(max(Na))
    N_min.append(min(Na))
    
N_top, N_bottom = max(N_max), min(N_min)
# print(N_top, N_bottom)
# Make the graph
levels = np.linspace(N_bottom, N_top, 1000)
fig_N_ax, ax=plt.subplots()
plt.contourf(xx, yy, N_tot, levels, cmap='jet') #levels,
plt.colorbar()
plt.title("Nuclear density "+ Material + "<"+asse+">")
plt.xlabel('x, $\AA$')
plt.ylabel('y, $\AA$')
#plt.zlabel('Nuclear density axial, $$')
#plt.show()()
fig_N_ax.savefig(Material+'_<'+asse.replace(" ","")+">"+"_N_axial.pdf")


# ## Electron density distribution 

# In[37]:


#definition of U_pl"
def fd2U_pl_DT_dx2(xpos):
    d_plm = sum(d_pl)/len(d_pl)
    E = 2*np.sqrt(np.pi)*N*d_plm*Z*z*e*a_0
    k1 = [E*((4*xpos**2-2*(coeff_B[i]+2*u**2))*coeff_a[i]/((coeff_B[i]+2*u**2)**(5/2)))*np.exp(-xpos**2/(coeff_B[i]+2*u**2))/10**(-24) for i in range(0,len(coeff_a))]
    k = sum(k1)
    return k

def fn_e_pl(xpos):
    return fd2U_pl_DT_dx2(xpos)/(4*np.pi*e**2)-Z*n_nuc_pl(xpos)

n_e_pl = []
for n in range(0, len(posizioni_proiettate)):
    n_e_pl.append(fn_e_pl(x_DT-posizioni_proiettate[n]))
    #n_e_pl.append(fd2U_pl_DT_dx2(x_DT-posizioni_proiettate[n]))
n_e_pl_tot = sum(n_e_pl)

figure, ax = plt.subplots(figsize=(7,5))
ax.plot(x_DT, n_e_pl_tot)
ax.set_title("Electron density distribution "+ Material + "("+asse+")")
ax.set_ylabel("$n_e$, $cm^{-3}$")
ax.set_xlabel("x, $\AA$")
#ax.legend()
ax.grid()
#plt.show()()


# In[38]:


## axial electronic distribution:
def gradient(x,y,x_0,y_0):
    d_atm = sum(d_at)/len(d_at)
    dU_dx = -2/d_atm*Z*z*e*a_0*sum([-2*(x-x_0)*coeff_a[i]/(coeff_B[i]+2*u**2)**2 * np.exp(-((x-x_0)**2+(y-y_0)**2)/(coeff_B[i]+2*u**2)) for i in range(len(coeff_a))])/e
    dU_dy = -2/d_atm*Z*z*e*a_0*sum([-2*(y-y_0)*coeff_a[i]/(coeff_B[i]+2*u**2)**2 * np.exp(-((x-x_0)**2+(y-y_0)**2)/(coeff_B[i]+2*u**2)) for i in range(len(coeff_a))])/e
    return dU_dx, dU_dy

def fLap_U_ax(x, y, x_0, y_0):
    d_atm = sum(d_at)/len(d_at)
    d2U_dx2 = -2/d_atm*Z*z*e*a_0*sum([2*coeff_a[i]/(coeff_B[i]+2*u**2)**4*((coeff_B[i]+2*u**2)-2*(x-x_0)**2) * np.exp(-((x-x_0)**2+(y-y_0)**2)/(coeff_B[i]+2*u**2)) for i in range(len(coeff_a))])/e
    d2U_dy2 = -2/d_atm*Z*z*e*a_0*sum([2*coeff_a[i]/(coeff_B[i]+2*u**2)**4*((coeff_B[i]+2*u**2)-2*(y-y_0)**2) * np.exp(-((x-x_0)**2+(y-y_0)**2)/(coeff_B[i]+2*u**2)) for i in range(len(coeff_a))])/e
    return d2U_dx2 + d2U_dy2

def fn_e_ax(x, y, x_0, y_0):
    return fLap_U_ax(x, y, x_0, y_0)/(4*np.pi*scipy.constants.e**2)-Z*n_nuc_ax(x,y,x_0,y_0)

n_e_ax = np.zeros_like(xx)
for atom in atom_positions:
    n_e_ax += fn_e_ax(xx,yy,atom[0],atom[1]) 

n_e_ax_max, n_e_ax_min = [], []
for neax in n_e_ax:
    n_e_ax_max.append(max(neax))
    n_e_ax_min.append(min(neax))
    
n_e_ax_top, n_e_ax_bottom = max(n_e_ax_max), min(n_e_ax_max)
#print(np.amin(n_e_ax),  np.amax(n_e_ax))
# print(n_e_ax_bottom,  n_e_ax_top)
# Make the graph
levels = np.linspace(n_e_ax_bottom,  n_e_ax_top, 1000)
fig, ax=plt.subplots()
plt.contourf(xx, yy, n_e_ax, levels, cmap='jet') #levels,
plt.colorbar()
plt.title("Electron density distribution "+ Material + "<"+ asse +">") 
plt.xlabel('x, $\AA$')
plt.ylabel('y, $\AA$')
#plt.show()()

print("Done")
exit()