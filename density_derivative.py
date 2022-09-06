import numpy as np
from sklearn import neighbors
from kernal_function import *

def density_derivative(particles, fluid_particle_indices, h, d, r_c,rho_0, p_0, Xi, gamma, dx):

    drho_by_dt= np.zeros(len(fluid_particle_indices))

    positions = list(zip(particles[:,0], particles[:,1])) # stores x,y in a tuple
    neighbor_ids, distances = neighbors.KDTree(positions).query_radius(positions,r_c,return_distance=True,sort_results=True) # Find the neighbours within the query radius
    positions=np.array(positions)

    fluid_particle_neighbours = neighbor_ids[:len(fluid_particle_indices)]# this array has the neighbours of all the fluid particles including self particle index.
    
    for n in range(len(fluid_particle_indices)): #loop over 1629 wall-particles 
        a=fluid_particle_indices[n] # w = 1,2,3..n:  where n is the number of fluid particles

        for m in range(1,len(fluid_particle_neighbours[n])): ###range should start from 1 because neighbours has the particle itself with zero distance
            b=fluid_particle_neighbours[n][m]

            flag_a = int(particles[a,3])
            flag_b = int(particles[b,3])

            # distance between particles
            drx = particles[a,0] - particles[b,0]
            dry = particles[a,1] - particles[b,1]
            rad = np.sqrt(drx**2 + dry**2)
            
            # nondimensional distance
            q = rad / h
            # kernel and derivative values (eq4)
            DWab = kernal_derivative(d, h, q) / h

            #### ????? kernel derivative with respect to x_a (eq4)
            Fab = [(drx/rad)*DWab,(dry/rad)*DWab]

            # densities, mass
            rho_a = particles[a,5]
            if (flag_b == 1): # if b boundary particle
                # calculate density of boundary particles, depending on interacting fluid particle
                rho_b = rho_0[flag_a] * ((particles[a,8] - Xi[flag_a]) / p_0[flag_a] + 1)**(1/gamma[flag_a])   # calculate density from pressure for wall particles 
                m_b = rho_0[flag_a] *dx*dx
                # velocity difference between particles(vel_wall from boundary_update)
                dvx = particles[a,6] - particles[b,6]
                dvy = particles[a,7] - particles[b,7]

            else: # straightforward if fluid particle
                rho_b = particles[b,5]
                m_b = particles[b,4]
                # velocity difference between particles
                dvx = particles[a,6] - particles[b,6]
                dvy = particles[a,7] - particles[b,7]
        
            # continuity equation
            V1= [dvx,dvy]       
            drho_by_dt[a] = drho_by_dt[a] + rho_a * (m_b / rho_b) * sum([x*y for x,y in zip(V1,Fab)])  # The sum represents the dot product                
           
    return drho_by_dt
