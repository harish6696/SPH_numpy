############# MOVING WALL NOT IMPLEMENTED ##################
from kernal_function import *
from phi.flow import *

def boundary_update(fluid_particles,boundary_particles,fluid_pressure,fluid_velocity,boundary_initial_velocity,boundary_prescribed_velocity,d,r_c,h,g):

    fluid_coords=fluid_particles.points   #fluid_particles and boundary_particles are a point cloud objects
    boundary_coords=boundary_particles.points
    particle_coords=math.concat([fluid_coords,boundary_coords],'particles')  # concatenating fluid coords and then boundary coords
    
    distance_matrix_vec= particle_coords - math.rename_dims(particle_coords, 'particles', 'others') # contains both the x and y component of separation between particles
    distance_matrix = math.vec_length(distance_matrix_vec) # contains magnitude of distance between particles

    distance_matrix_vec= math.where(distance_matrix > r_c, 0, distance_matrix_vec)
    distance_matrix = math.where(distance_matrix > r_c, 0, distance_matrix)

    #Slicing the distance matrix of fluid neighbour of boundary particles
    q=distance_matrix.particles[fluid_coords.particles.size:].others[:fluid_coords.particles.size]/h   

    W=kernal_function(d,h,q)

    sum_pW = W.others*fluid_pressure.particles  # col-vector, each entry corresponding to each wall particle

    dry=distance_matrix_vec['y'].particles[fluid_coords.particles.size:].others[:fluid_coords.particles.size]*g 
    drx=distance_matrix_vec['x'].particles[fluid_coords.particles.size:].others[:fluid_coords.particles.size]*0 

    sum_rhorW =(dry.others*W.particles).others*fluid_pressure.particles # col-vector, each entry corresponding to each wall particle

    sum_W = math.sum(W,'others')  # col-vector, (row sum of W) each entry corresponding to each wall particle

    boundary_particle_pressure= math.where(sum_W==0,0,sum_W)
    boundary_particle_pressure= math.where(sum_W!=0,math.divide_no_nan((sum_rhorW+sum_pW),sum_W),sum_W)
        
    return boundary_particle_pressure





                
    

