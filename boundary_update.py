############# MOVING WALL NOT IMPLEMENTED i.e. wall_prescribed_velocity =0 ##################
from kernal_function import *
#from phi.flow import *
from phi.flow import *

def boundary_update(fluid_particles,wall_particles,fluid_pressure,fluid_density,d,r_c,h,g):

    fluid_coords=fluid_particles.points   #fluid_particles and wall_particles are a point cloud objects
    wall_coords=wall_particles.points

    #wall_particle_velocity2 = wall_particles.values
    
    particle_coords=math.concat([fluid_coords,wall_coords],'particles')  # concatenating fluid coords and then wall coords

    distance_matrix_vec= particle_coords - math.rename_dims(particle_coords, 'particles', 'others') # contains both the x and y component of separation between particles
    distance_matrix = math.vec_length(distance_matrix_vec) # contains magnitude of distance between particles

    #Apply cut off to the distance matrix at the end after all the cut-offs
    distance_matrix_vec= math.where(distance_matrix > r_c, 0, distance_matrix_vec)
    distance_matrix = math.where(distance_matrix > r_c, 0, distance_matrix)

    #Slicing the distance matrix of fluid neighbour of wall particles
    q=distance_matrix.particles[fluid_coords.particles.size:].others[:fluid_coords.particles.size]/h   

    W=kernal_function(d,h,q)

    sum_pW = W.others*fluid_pressure.particles  # col-vector, each entry corresponding to each wall particle

    dry=distance_matrix_vec['y'].particles[fluid_coords.particles.size:].others[:fluid_coords.particles.size]*g 
    drx=distance_matrix_vec['x'].particles[fluid_coords.particles.size:].others[:fluid_coords.particles.size]*0 

    sum_rhorW =(dry*W).others*fluid_density.particles # col-vector, each entry corresponding to each wall particle

    sum_W = math.sum(W,'others')  # col-vector, (row sum of W) each entry corresponding to each wall particle

    wall_particle_pressure= math.where(sum_W==0,0,sum_W)
    wall_particle_pressure= math.where(sum_W!=0,math.divide_no_nan((sum_rhorW+sum_pW),sum_W),sum_W)

    fluid_particle_velocity=math.expand(fluid_particles.values, instance(fluid_particles))
    wall_particle_velocity=math.expand(wall_particles.values, instance(wall_particles))

    sum_vWx = W.others*fluid_particle_velocity['x'].particles
    sum_vWy = W.others*fluid_particle_velocity['y'].particles
    
    #sum_W and wall_particle_velocity['x' or 'y'] have the same dimensions i.e. both are a column vector with particle dimension
    wall_vel_x = math.where(sum_W!=0, math.divide_no_nan(-sum_vWx,sum_W), wall_particle_velocity['x']) 
    wall_vel_y = math.where(sum_W!=0, math.divide_no_nan(-sum_vWy,sum_W), wall_particle_velocity['y'])
    
    wall_particle_velocity_temp = stack([wall_vel_x, wall_vel_y], channel(vector='x,y'))
        
    wall_particles = wall_particles.with_values(wall_particle_velocity_temp) # wall_particle_velocity is a point cloud
    
    return wall_particle_pressure, wall_particles





                
    

