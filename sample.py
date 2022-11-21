

############for desnity derivative
from phi.flow import *
from kernal_function import *

fluid_coords = pack_dims(math.meshgrid(x=1, y=3), 'x,y', instance('particles')) * (0.2/1, 0.12/3) + (0.003,0.003)
wall_coords = pack_dims(math.meshgrid(x=1, y=3), 'x,y', instance('particles')) * ( (0.018/1), (0.204/3) ) + (-0.015, 0.003)  

coords=math.concat([fluid_coords,wall_coords],'particles')
distance_matrix_vec= coords - math.rename_dims(coords, 'particles', 'others') # contains both the x and y component of separation between particles
distance_matrix = math.vec_length(distance_matrix_vec)
#math.print(distance_matrix)
alpha=2
alpha_ab=math.where( (distance_matrix==0), 0, alpha)# REMOVE THE SELF PARTICLES (Diagonals are set to 0)
#print('inter')
#math.print(alpha_ab)
alpha_ab=math.where((distance_matrix>0.04) , 0, alpha_ab)
#print('final')
#math.print(alpha_ab)

alpha_ab_fluid=alpha_ab.particles[:fluid_coords.particles.size].others[:]

#math.print(alpha_ab_fluid)

#breakpoint()

distance_matrix = math.where(distance_matrix > 0.04, 0, distance_matrix) 

distance_matrix_fluid_elements=distance_matrix.particles[:fluid_coords.particles.size].others[:]

fluid_particle_density= tensor([100,1000,10], instance('particles'))

fluid_coords = pack_dims(math.meshgrid(x=1, y=3), 'x,y', instance('particles')) * (0.2/1, 0.12/3) + (0.003,0.003)
wall_coords = pack_dims(math.meshgrid(x=1, y=3), 'x,y', instance('particles')) * ( (0.018/1), (0.204/3) ) + (-0.015, 0.003)  
fluid_pressure=math.zeros(instance(fluid_coords))  

wall_particles = PointCloud(Sphere(wall_coords, radius=0.002))
fluid_particles = PointCloud(Sphere(fluid_coords, radius=0.002))

coords=math.concat([fluid_coords,wall_coords],'particles')

fluid_velocity = fluid_particles * (0, 0)
    
#math.print(fluid_velocity)

#ACTUALLY IT IS PARTICLE VELOCITY
fluid_velocity= tensor([(1, 0), (10, 2), (0, 12),(1,1),(3,4),(5,5)], instance('particles') & channel(vector='x,y'))

#print(fluid_velocity)

#math.print(fluid_velocity.fluid_vel[1]['x'])

fluid_relative_velocity=fluid_velocity - math.rename_dims(fluid_velocity,'particles', 'others')

distance_matrix_vec=coords-math.rename_dims(coords,'particles','others')
distance_matrix = math.vec_length(distance_matrix_vec)

# considering all neighbours of fluid particles (both fluid and wall)
fluid_relative_dist=distance_matrix.particles[:fluid_coords.particles.size].others[:]

#wall_particle_density= tensor([0, 0, 0], instance('others'))
wall_particle_density= math.rename_dims(math.ones(instance(wall_coords)),'particles','others')
#wall_particle_density=math.rename_dims(wall_particle_density,'particles','others')

fluid_particle_density= tensor([100,1000,10], instance('others'))
wall_particle_mass= tensor([1, 1, 1], instance('others'))
fluid_particle_mass= tensor([100,200,50], instance('others'))

particle_density= math.concat([fluid_particle_density,wall_particle_density], dim='others') #1D Scalar array of densities of all particles
particle_mass= math.concat([fluid_particle_mass,wall_particle_mass], dim='others')
#math.print(particle_density)

#create a matrix of densities
#a=instance(fluid_coords)+instance(wall_coords)
#math.print(a)
#particle_density=expand(particle_density, instance(particles=instance(fluid_coords)+ instance(wall_coords))) #################HOw to get 6 ? NOT EVEN REQUIRED!"!" 
#math.print(particle_density)
#breakpoint()

particle_density = math.where(distance_matrix==0,0,particle_density)
particle_density=math.where(distance_matrix > 0.04, 0, particle_density) #0 for places which are not neighbours for particle under consideration. Stacks the particle density vertically. i.e. dupicates the densities
particle_mass=math.where(distance_matrix > 0.04, 0, particle_mass)

distance_matrix_vec= math.where(distance_matrix > 0.04, 0, distance_matrix_vec)  #Only including the nearest neighbours
distance_matrix = math.where(distance_matrix > 0.04, 0, distance_matrix)

# Slicing the distance matrix to get the portion of ALL neighbors of fluid particles 
h=0.006
q=distance_matrix.particles[:fluid_coords.particles.size].others[:]/1  # distance 

#math.print(distance_matrix)

print('q')
math.print(q)

#math.print(distance_matrix_vec['y'])

drx=distance_matrix_vec['x'].particles[:fluid_coords.particles.size].others[:]
dry=distance_matrix_vec['y'].particles[:fluid_coords.particles.size].others[:] 

mod_dist=(distance_matrix.particles[:fluid_coords.particles.size].others[:])
print('modified dist')
math.print(mod_dist)

Fab_x= math.where(mod_dist!=0, math.divide_no_nan(drx,mod_dist)*q,drx)  #Shoud be multiplied by DWab (derivative of kernal function instead of q)
print('Fabx')
math.print(Fab_x)

Fab_y= math.where(mod_dist!=0, math.divide_no_nan(dry,mod_dist)*q,dry)
print('Faby')
math.print(Fab_y)

dvx=fluid_relative_velocity['x'].particles[:fluid_coords.particles.size].others[:]
dvy=fluid_relative_velocity['y'].particles[:fluid_coords.particles.size].others[:]
# print('vel_x original')
# math.print(dvx)

dvx=math.where(fluid_relative_dist>0.04,0,dvx) # relative x-velocity between a fluid particle and its neighbour
dvy=math.where(fluid_relative_dist>0.04,0,dvy)

print('dvx')
math.print(dvx)
print('dvy')
math.print(dvy)

#breakpoint()

fluid_initial_density=1
fluid_Xi=0
fluid_adiabatic_exp=7
fluid_p_0=10 # 
#wall_density=math.zeros(instance(wall_coords)) 

dx=0.006

#Need to calculate density and mass of only wall particles which are neighbours of fluid particles
print('\n Particle density stacked matirx after doing elimination among the nearest neighbours:')
math.print(particle_density)

fluid_particle_neighbour_density=particle_density.particles[:fluid_coords.particles.size].others[:]
fluid_particle_neighbour_mass=particle_mass.particles[:fluid_coords.particles.size].others[:]

print('\n Fluid particle density with all neighbours (sliced matrix of particle density)')
math.print(fluid_particle_neighbour_density)

#rho_b = fluid_initial_density * ((wall_density - fluid_Xi) / fluid_p_0 + 1)**(1/fluid_adiabatic_exp)   # calculate density from pressure for wall particles 
#m_b = fluid_initial_density *dx*dx

fluid_particle_fluid_neighbour_density=fluid_particle_neighbour_density.particles[:].others[:fluid_coords.particles.size]
fluid_particle_wall_neighbour_density=fluid_particle_neighbour_density.particles[:].others[fluid_coords.particles.size:]
fluid_particle_fluid_neighbour_mass=fluid_particle_neighbour_mass.particles[:].others[:fluid_coords.particles.size]
fluid_particle_wall_neighbour_mass=fluid_particle_neighbour_mass.particles[:].others[fluid_coords.particles.size:]


print('\n Fluid particle fluid neighbour density ')
math.print(fluid_particle_fluid_neighbour_density)
print('\n Fluid particle wall neighbour density ')
math.print(fluid_particle_wall_neighbour_density)

print('\n Fluid pressure: ')
fluid_pressure=tensor([0.2,0.5,0.3], instance('particles'))
math.print(fluid_pressure)

fluid_particle_wall_neighbour_density_term=fluid_initial_density * ((fluid_pressure - fluid_Xi) / fluid_p_0 + 1)**(1/fluid_adiabatic_exp)

#ALL boundary particle neighbours for a given fluid particle will have same density 
fluid_particle_wall_neighbour_density=math.where(fluid_particle_wall_neighbour_density!=0,fluid_particle_wall_neighbour_density_term,fluid_particle_wall_neighbour_density)
fluid_particle_wall_neighbour_mass =math.where(fluid_particle_wall_neighbour_mass!=0,(fluid_initial_density*dx*dx),fluid_particle_wall_neighbour_mass)

print('fluid_particle_fluid_neighbour_density')
math.print(fluid_particle_fluid_neighbour_density)
print('fluid_particle_wall_neighbour_density')
math.print(fluid_particle_wall_neighbour_density)

#joining back the densities to form fluid_particle_neighbour_density. Horizontal concatenation to get (rho_b) and m_b
fluid_particle_neighbour_density=math.concat([fluid_particle_fluid_neighbour_density, fluid_particle_wall_neighbour_density], 'others')
fluid_particle_neighbour_mass =math.concat([fluid_particle_fluid_neighbour_mass, fluid_particle_wall_neighbour_mass], 'others')
print('fluid_particle_ALL_neighbour_mass')
math.print(fluid_particle_neighbour_mass)

neighbour_mass_by_density_ratio=math.divide_no_nan(fluid_particle_neighbour_mass,fluid_particle_neighbour_mass) #m_b/rho_b

print('ratio')
math.print(neighbour_mass_by_density_ratio)

print('dot_product_result')
dot_product_result=(dvx*Fab_x)+(dvy*Fab_y)
math.print(dot_product_result)

#fluid_particle_density=expand(fluid_particle_density, instance(fluid_coords)+instance(wall_coords))
print('rho_a')
math.print(fluid_particle_density)

#drho_by_dt=math.sum((neighbour_mass_by_density_ratio*dot_product_result).others*fluid_particle_density.particles,'others')
temp=neighbour_mass_by_density_ratio*dot_product_result
print('temp')
#print(f"{temp.particles[0].others[:]}")
math.print(temp)

print('fd')
math.print(fluid_particle_density)
fluid_particle_density= math.rename_dims(fluid_particle_density,'others','particles')
print('addition')

res=math.where(dvy!=0,fluid_particle_density+dvy,dvy)

math.print(res)

print('fd_inverse')
math.print(1/fluid_particle_density)

#math.print(dvy**2)
#print(f"{temp.particles[0].others[:]}")
#math.print(temp*fluid_particle_density)

print('initial_visc_art_factt')
visc_art_fact=math.zeros(instance(distance_matrix.particles[:fluid_coords.particles.size].others[:]))
math.print(visc_art_fact)

print('dvy')
math.print(dvy)

print('2d randos')
term=math.random_normal(instance(distance_matrix.particles[:fluid_coords.particles.size].others[:]))
math.print(term)

visc_art_fact = math.where(dvy<0, term, visc_art_fact)

print('final ')
math.print(visc_art_fact)
#Goal is to multiply row-1 by rho_a first element, row-2 by rho_a second element,...




#print('density derivative')
#math.print(drho_by_dt)


"""

alpha_d = 0.004661441847880 / (h * h)

W= math.where((q<3) & (q>2) ,  ((3-q)**5),q)

#print(f"{W:full}")
print("\n")

fluid_pressure=tensor((1,2,3),instance('particles'))
#fluid_pressure=math.ones(instance(fluid_coords))
sum_pW=W.others*fluid_pressure.particles
#print(f"{res:full}")
print("Sum_pW")
math.print(sum_pW)
print('\n')


g=-9.81

dry=distance_matrix_vec['y'].particles[fluid_coords.particles.size:].others[:fluid_coords.particles.size]*g 
drx=distance_matrix_vec['x'].particles[fluid_coords.particles.size:].others[:fluid_coords.particles.size]*0 

sum_rhorW=(dry.others*W.particles).others*fluid_pressure.particles
#mm=(dry.others*W.particles)
print("Sum_rhorW")
math.print(sum_rhorW)
print('\n')

print('sum_W')
sum_W=math.sum(W,'others')
math.print(sum_W)
print('\n')



# wall particle fluid neighbour
print("Wall pressure")
wall_pressure= math.where(sum_W==0,0,sum_W)
wall_pressure= math.where(sum_W!=0,((sum_rhorW+sum_pW)/sum_W),sum_W)
math.print(wall_pressure)
"""


#hERE LIES THE CODE FOR DENSITY DERIVATIVE
""" fluid_coords = pack_dims(math.meshgrid(x=1, y=3), 'x,y', instance('particles')) * (0.2/1, 0.12/3) + (0.003,0.003)
wall_coords = pack_dims(math.meshgrid(x=1, y=3), 'x,y', instance('particles')) * ( (0.018/1), (0.204/3) ) + (-0.015, 0.003)  
fluid_pressure=math.zeros(instance(fluid_coords))  

wall_particles = PointCloud(Sphere(wall_coords, radius=0.002))
fluid_particles = PointCloud(Sphere(fluid_coords, radius=0.002))

coords=math.concat([fluid_coords,wall_coords],'particles')

fluid_velocity = fluid_particles * (0, 0)
    
math.print(fluid_velocity)

fluid_velocity= tensor([(1, 0), (10, 2), (0, 12),(1,1),(3,4),(5,5)], instance('particles') & channel(vector='x,y'))

#print(fluid_velocity)

#math.print(fluid_velocity.fluid_vel[1]['x'])

fluid_relative_velocity=fluid_velocity - math.rename_dims(fluid_velocity,'particles', 'others')

distance_matrix_vec=coords-math.rename_dims(coords,'particles','others')
distance_matrix = math.vec_length(distance_matrix_vec)


# considering all neighbours of fluid particles (both fluid and wall)
fluid_relative_dist=distance_matrix.particles[:fluid_coords.particles.size].others[:]

#wall_particle_density= tensor([0, 0, 0], instance('others'))
wall_particle_density= math.rename_dims(math.zeros(instance(wall_coords)),'particles','others')
#wall_particle_density=math.rename_dims(wall_particle_density,'particles','others')

fluid_particle_density= tensor([100,200,50], instance('others'))
wall_particle_mass= tensor([0, 0, 0], instance('others'))
fluid_particle_mass= tensor([100,200,50], instance('others'))

particle_density= math.concat([fluid_particle_density,wall_particle_density], dim='others') #1D Scalar array of densities of all particles
particle_mass= math.concat([fluid_particle_mass,wall_particle_mass], dim='others')
#math.print(particle_density)

#create a matrix of densities
#a=instance(fluid_coords)+instance(wall_coords)
#math.print(a)
#particle_density=expand(particle_density, instance(particles=instance(fluid_coords)+ instance(wall_coords))) #################HOw to get 6 ? NOT EVEN REQUIRED!"!" 
#math.print(particle_density)
#breakpoint()


particle_density=math.where(distance_matrix > 0.04, 0, particle_density) #0 for places which are not neighbours for particle under consideration. Stacks the particle density vertically. i.e. dupicates the densities
particle_mass=math.where(distance_matrix > 0.04, 0, particle_mass)

distance_matrix_vec= math.where(distance_matrix > 0.04, 0, distance_matrix_vec)  #Only including the nearest neighbours
distance_matrix = math.where(distance_matrix > 0.04, 0, distance_matrix)

# Slicing the distance matrix to get the portion of ALL neighbors of fluid particles 
h=0.006
q=distance_matrix.particles[:fluid_coords.particles.size].others[:]/1  # distance 

#math.print(distance_matrix)

print('q')
math.print(q)

#math.print(distance_matrix_vec['y'])

drx=distance_matrix_vec['x'].particles[:fluid_coords.particles.size].others[:]
dry=distance_matrix_vec['y'].particles[:fluid_coords.particles.size].others[:] 

mod_dist=(distance_matrix.particles[:fluid_coords.particles.size].others[:])
print('modified dist')
math.print(mod_dist)

Fab_x= math.where(mod_dist!=0, math.divide_no_nan(drx,mod_dist)*q,drx)  #Shoud be multiplied by DWab (derivative of kernal function instead of q)
print('Fabx')
math.print(Fab_x)

Fab_y= math.where(mod_dist!=0, math.divide_no_nan(dry,mod_dist)*q,dry)
print('Faby')
math.print(Fab_y)

dvx=fluid_relative_velocity['x'].particles[:fluid_coords.particles.size].others[:]
dvy=fluid_relative_velocity['y'].particles[:fluid_coords.particles.size].others[:]
# print('vel_x original')
# math.print(dvx)

dvx=math.where(fluid_relative_dist>0.04,0,dvx) # relative x-velocity between a fluid particle and its neighbour
dvy=math.where(fluid_relative_dist>0.04,0,dvy)

print('dvx')
math.print(dvx)
print('dvy')
math.print(dvy)

#breakpoint()

fluid_initial_density=1
fluid_Xi=0
fluid_adiabatic_exp=7
fluid_p_0=10 # 
#wall_density=math.zeros(instance(wall_coords)) 

dx=0.006

#Need to calculate density and mass of only wall particles which are neighbours of fluid particles
print('\n Particle density stacked matirx after doing elimination among the nearest neighbours:')
math.print(particle_density)

fluid_particle_neighbour_density=particle_density.particles[:fluid_coords.particles.size].others[:]
fluid_particle_neighbour_mass=particle_mass.particles[:fluid_coords.particles.size].others[:]

print('\n Fluid particle density with all neighbours (sliced matrix of particle density)')
math.print(fluid_particle_neighbour_density)

#rho_b = fluid_initial_density * ((wall_density - fluid_Xi) / fluid_p_0 + 1)**(1/fluid_adiabatic_exp)   # calculate density from pressure for wall particles 
#m_b = fluid_initial_density *dx*dx

fluid_particle_fluid_neighbour_density=fluid_particle_neighbour_density.particles[:].others[:fluid_coords.particles.size]
fluid_particle_wall_neighbour_density=fluid_particle_neighbour_density.particles[:].others[fluid_coords.particles.size:]
fluid_particle_fluid_neighbour_mass=fluid_particle_neighbour_mass.particles[:].others[:fluid_coords.particles.size]
fluid_particle_wall_neighbour_mass=fluid_particle_neighbour_mass.particles[:].others[fluid_coords.particles.size:]


print('\n Fluid particle fluid neighbour density ')
math.print(fluid_particle_fluid_neighbour_density)
print('\n Fluid particle wall neighbour density ')
math.print(fluid_particle_wall_neighbour_density)

print('\n Fluid pressure: ')
fluid_pressure=tensor([0.2,0.5,0.3], instance('particles'))
math.print(fluid_pressure)

fluid_particle_wall_neighbour_density_term=fluid_initial_density * ((fluid_pressure - fluid_Xi) / fluid_p_0 + 1)**(1/fluid_adiabatic_exp)

#ALL boundary particle neighbours for a given fluid particle will have same density 
fluid_particle_wall_neighbour_density=math.where(fluid_particle_wall_neighbour_density!=0,fluid_particle_wall_neighbour_density_term,fluid_particle_wall_neighbour_density)
fluid_particle_wall_neighbour_mass =math.where(fluid_particle_wall_neighbour_mass!=0,(fluid_initial_density*dx*dx),fluid_particle_wall_neighbour_mass)

print('fluid_particle_fluid_neighbour_density')
math.print(fluid_particle_fluid_neighbour_density)
print('fluid_particle_wall_neighbour_density')
math.print(fluid_particle_wall_neighbour_density)

#joining back the densities to form fluid_particle_neighbour_density. Horizontal concatenation to get (rho_b) and m_b
fluid_particle_neighbour_density=math.concat([fluid_particle_fluid_neighbour_density, fluid_particle_wall_neighbour_density], 'others')
fluid_particle_neighbour_mass =math.concat([fluid_particle_fluid_neighbour_mass, fluid_particle_wall_neighbour_mass], 'others')
print('fluid_particle_ALL_neighbour_mass')
math.print(fluid_particle_neighbour_mass)

neighbour_mass_by_density_ratio=math.divide_no_nan(fluid_particle_neighbour_mass,fluid_particle_neighbour_mass) #m_b/rho_b

print('ratio')
math.print(neighbour_mass_by_density_ratio)

print('dot_product_result')
dot_product_result=(dvx*Fab_x)+(dvy*Fab_y)
math.print(dot_product_result)

fluid_particle_density=expand(fluid_particle_density, instance(fluid_coords)+instance(wall_coords))
print('rho_a_extended')
math.print(fluid_particle_density)

drho_by_dt=math.sum((neighbour_mass_by_density_ratio*dot_product_result).others*fluid_particle_density.particles,'others')

print('density derivative')
math.print(drho_by_dt)

 """


#here lies the code of boundary update testing
"""
####boundary update
fluid_coords = pack_dims(math.meshgrid(x=1, y=3), 'x,y', instance('particles')) * (0.2/1, 0.12/3) + (0.003,0.003)
wall_coords = pack_dims(math.meshgrid(x=1, y=3), 'x,y', instance('particles')) * ( (0.018/1), (0.204/3) ) + (-0.015, 0.003)  
fluid_pressure=math.zeros(instance(fluid_coords))  
#print(f"{fluid_coords:full}")
#print(f"{wall_coords:full}")
wall_particles = PointCloud(Sphere(wall_coords, radius=0.002))
fluid_particles = PointCloud(Sphere(fluid_coords, radius=0.002))


coords=math.concat([fluid_coords,wall_coords],'particles')


distance_matrix_vec= coords - math.rename_dims(coords, 'particles', 'others')
distance_matrix = math.vec_length(distance_matrix_vec)
#print("\n")

#print("\mod")
distance_matrix_vec= math.where(distance_matrix > 0.04, 0, distance_matrix_vec)
#print(numpy.asarray(distance_matrix_vec['y']))

distance_matrix = math.where(distance_matrix > 0.04, 0, distance_matrix)
#print(numpy.asarray(distance_matrix))

fluid_pressure=math.zeros(instance(fluid_coords))


# print("\n")
# print(numpy.asarray(distance_matrix))

# Slicing the distance matrix to get the portion of fluid neighbors of wall particles 
h=0.006
q=distance_matrix.particles[fluid_coords.particles.size:].others[:fluid_coords.particles.size]/1  # distance 

#nonzero_indices = math.nonzero(rename_dims(distance_matrix, instance, spatial('particles, others')))
#print(f"{distance_matrix:full}")
#print(f"{nonzero_indices:full}")
#print(f"{q:full}") 


alpha_d = 0.004661441847880 / (h * h)



W= math.where((q<3) & (q>2) ,  ((3-q)**5),q)

#print(f"{W:full}")
print("\n")
#print(f"{math.transpose(fluid_pressure,0):full}")

#fp=numpy.asarray(fluid_pressure)

fluid_pressure=tensor((1,2,3),instance('particles'))
#fluid_pressure=math.ones(instance(fluid_coords))
sum_pW=W.others*fluid_pressure.particles
#print(f"{res:full}")
print("Sum_pW")
math.print(sum_pW)
print('\n')
#fp=numpy.array([[1],[2],[3]])
#W_np=numpy.asarray(W)
#sum_pW=W_np.dot(fp)   #1d vector with pW corresponding to each diagonal 

g=-9.81

dry=distance_matrix_vec['y'].particles[fluid_coords.particles.size:].others[:fluid_coords.particles.size]*g 
drx=distance_matrix_vec['x'].particles[fluid_coords.particles.size:].others[:fluid_coords.particles.size]*0 

sum_rhorW=(dry.others*W.particles).others*fluid_pressure.particles
#mm=(dry.others*W.particles)
print("Sum_rhorW")
math.print(sum_rhorW)
print('\n')
#dry=numpy.asarray(dry)
#drx=numpy.asarray(drx)
#sum_rhoW=np.matmul(np.matmul(dry,W_np),(fp))

#sum_rhoW=np.matmul(dry,W_np)

#print(sum_rhoW)

#math.print(W)
#print('g')
print('sum_W')
sum_W=math.sum(W,'others')
math.print(sum_W)
print('\n')

#print('Combined term')
#math.print(sum_rhorW+sum_pW/sum_W)
#print('\n')


# wall particle fluid neighbour
print("Wall pressure")
wall_pressure= math.where(sum_W==0,0,sum_W)
wall_pressure= math.where(sum_W!=0,((sum_rhorW+sum_pW)/sum_W),sum_W)
math.print(wall_pressure)
"""