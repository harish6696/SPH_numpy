
from kernal_function import *
#from phi.flow import *
from phi.flow import *
#math.set_global_precision(32)

def calculate_density_derivative(fluid_particles, wall_particles, fluid_particle_density,fluid_pressure,fluid_particle_mass,fluid_initial_density,fluid_Xi,fluid_adiabatic_exp,fluid_p_0, h, d, r_c, dx):
    #Note: No wall_particle_density and wall_particle_mass in input arguments. It is expected to be zero initially

    fluid_coords=fluid_particles.points   #fluid_particles and boundary_particles are a point cloud objects
    wall_coords=wall_particles.points
    particle_coords=math.concat([fluid_coords,wall_coords],'particles')  # concatenating fluid coords and then boundary coords
    
    #wall_particle_velocity= wall_particles *(0,0) ####QQQQQQQQ DOES THIS HAVE A DIMENSION NAMED PARTICLES ? (same q for fluid_particle_velocity)

    fluid_particle_velocity=math.expand(fluid_particles.values, instance(fluid_particles)) # adding particle dimension to fluid_particles.values
    wall_particle_velocity = math.expand(wall_particles.values, instance(wall_particles))

    particle_velocity=math.concat([fluid_particle_velocity,wall_particle_velocity], dim='particles') 

    wall_particle_density=math.rename_dims(math.zeros(instance(wall_coords)),'particles','others')
    wall_particle_mass=math.rename_dims(math.zeros(instance(wall_coords)),'particles','others')

    fluid_particle_density=math.rename_dims(fluid_particle_density,'particles','others') # fluid_particle_density and mass have initially particle dimension (i.e column vector)
    fluid_particle_mass=math.rename_dims(fluid_particle_mass,'particles','others')
    
    particle_density= math.concat([fluid_particle_density,wall_particle_density], dim='others') #1D Scalar array of densities of all particles (now it is a row vector)
    particle_mass= math.concat([fluid_particle_mass,wall_particle_mass], dim='others')
    
    #Compute distance between all particles 
    distance_matrix_vec= particle_coords - math.rename_dims(particle_coords, 'particles', 'others') # contains both the x and y component of separation between particles
    distance_matrix = math.vec_length(distance_matrix_vec) # contains magnitude of distance between ALL particles
    
    particle_neighbour_density = math.where(distance_matrix==0,0,particle_density)
    particle_neighbour_density=math.where(distance_matrix > r_c, 0, particle_neighbour_density) #0 for places which are not neighbours for particle under consideration. Stacks the particle density vertically. i.e. dupicates the densities
    
    particle_neighbour_mass = math.where(distance_matrix==0,0,particle_mass)
    particle_neighbour_mass=math.where(distance_matrix > r_c, 0, particle_neighbour_mass)

    #QQQQQQQQQQQWhy in the sample did u take fluid particle having 6 elements? fluid particles are only three 

    particle_relative_velocity=particle_velocity - math.rename_dims(particle_velocity,'particles', 'others')# 2d matrix of ALL particle velocity
    dvx=particle_relative_velocity['x'].particles[:fluid_coords.particles.size].others[:]# separating the x and y components
    dvy=particle_relative_velocity['y'].particles[:fluid_coords.particles.size].others[:]

    fluid_particle_relative_dist=distance_matrix.particles[:fluid_coords.particles.size].others[:]
    dvx=math.where(fluid_particle_relative_dist>r_c,0,dvx) # relative x-velocity between a fluid particle and its neighbour
    dvy=math.where(fluid_particle_relative_dist>r_c,0,dvy)
 
    #Do all the cut off things before the final cut off for distance matrix
    distance_matrix_vec= math.where(distance_matrix > r_c, 0, distance_matrix_vec)
    distance_matrix = math.where(distance_matrix > r_c, 0, distance_matrix)   # Stores the distance between neighbours which are inside cutoff radius

    #Slicing the distance matrix of ALL neighbour of fluid particles
    q=distance_matrix.particles[:fluid_coords.particles.size].others[:]/h 

    DWab = kernal_derivative(d, h, q) / h

    drx=distance_matrix_vec['x'].particles[:fluid_coords.particles.size].others[:]
    dry=distance_matrix_vec['y'].particles[:fluid_coords.particles.size].others[:] 

    mod_dist=(distance_matrix.particles[:fluid_coords.particles.size].others[:])
    Fab_x= math.where(mod_dist!=0, math.divide_no_nan(drx,mod_dist)*DWab,drx) 
    Fab_y= math.where(mod_dist!=0, math.divide_no_nan(dry,mod_dist)*DWab,dry)

    fluid_particle_neighbour_density=particle_neighbour_density.particles[:fluid_coords.particles.size].others[:]
    fluid_particle_neighbour_mass=particle_neighbour_mass.particles[:fluid_coords.particles.size].others[:]

    #Splitting the density and mass of fluid particle neighbours into two matrices....QQQQQQ CAN WE AVOID THIS? as we have to join it back anyways
    fluid_particle_fluid_neighbour_density=fluid_particle_neighbour_density.particles[:].others[:fluid_coords.particles.size]
    fluid_particle_wall_neighbour_density=fluid_particle_neighbour_density.particles[:].others[fluid_coords.particles.size:]
    fluid_particle_fluid_neighbour_mass=fluid_particle_neighbour_mass.particles[:].others[:fluid_coords.particles.size]
    fluid_particle_wall_neighbour_mass=fluid_particle_neighbour_mass.particles[:].others[fluid_coords.particles.size:]

    ######'fluid_pressure' is for each fluid particle 
    fluid_particle_wall_neighbour_density_term=fluid_initial_density * ((fluid_pressure - fluid_Xi) / fluid_p_0 + 1)**(1/fluid_adiabatic_exp)
    helper_matrix = distance_matrix.particles[:fluid_coords.particles.size].others[fluid_coords.particles.size:] 
    helper_matrix = math.where(helper_matrix!=0, 1 , helper_matrix)  # technically called adjacency matrix
    
    #ALL boundary particle neighbours for a given fluid particle will have same density 
    fluid_particle_wall_neighbour_density = helper_matrix * fluid_particle_wall_neighbour_density_term
    fluid_particle_wall_neighbour_mass =math.where(helper_matrix!=0,(fluid_initial_density*dx*dx),fluid_particle_wall_neighbour_mass)
    
    #joining the density and mass of the fluid and wall particle back together
    fluid_particle_neighbour_density=math.concat([fluid_particle_fluid_neighbour_density, fluid_particle_wall_neighbour_density], 'others')
    fluid_particle_neighbour_mass =math.concat([fluid_particle_fluid_neighbour_mass, fluid_particle_wall_neighbour_mass], 'others')

    neighbour_mass_by_density_ratio=math.divide_no_nan(fluid_particle_neighbour_mass,fluid_particle_neighbour_density) #m_b/rho_b
    dot_product_result=(dvx*Fab_x)+(dvy*Fab_y)

    #Fluid_particle_density is a vector, each row of result is multiplied by the corresponding term of fluid_particle_density and then Sum along each row to get the result(drho_by_dt) for each fluid particle 
    #######CHECK THIS below line WHILE RUNNING THE FULL CODE
    fluid_particle_density= math.rename_dims(fluid_particle_density,'others','particles') 

    drho_by_dt=math.sum((neighbour_mass_by_density_ratio*dot_product_result)*fluid_particle_density,'others')
    #drho_by_dt is calculated only for fluid particles
    return drho_by_dt

