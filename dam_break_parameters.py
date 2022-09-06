
import numpy as np
from create_particles import *

def dam_break_case(dx,d,alph):

    g=-9.81
    height=0.3
    v_max=np.sqrt(2*abs(g)*height)

    rho_0=[0,0]     # stores base density of the particles 
    gamma=[0,0]     # adiabatic coefficient (pressure coefficient)
    c_0 = [0,0]     # artificial speed of sound, c_0
    p_0 = [0,0]   
    Xi =[0,0]       #back pressure
    mu = [0,0]
    alpha=[0,0]
    
    ###### Properties of Fluid Particles #######
    number_fluid_particles=5000  #useless if particles are generated uniformly
    flag=0          
    rho_0[flag] = 1000.0                                # density of water    
    gamma[flag] = 7                                     # pressure exponent
    c_0[flag]=10*v_max                                  # v_max = 2*abs(g)*height
    p_0[flag] = (rho_0[flag] * (c_0[flag]**2)) / (gamma[flag]) # reference pressure 
    Xi[flag] = 0.0 * p_0[flag]                          # background pressure
    mu[flag] = 0.01                                     # viscosity
    alpha[flag] = alph                                  # artificial visc factor

    xmin_fluid= 0; xmax_fluid= 0.6 ; ymin_fluid=0; ymax_fluid=0.3 ; zmin_fluid= 0; zmax_fluid=0 ; 
    fluid_coords = create_fluid_coords(xmin_fluid,xmax_fluid,ymin_fluid,ymax_fluid,zmin_fluid,zmax_fluid, number_fluid_particles,dx)
    
    particles=[];initial_fluid_velocity=[0,0]
    particles,fluid_particle_indices= fluid_initialization(particles,fluid_coords,flag,initial_fluid_velocity,rho_0[flag],dx,d)


    ###### Properties of Wall particles #####
    flag=1
    alpha[flag]=0.0
    width=dx*np.ceil(1.61/dx) #width=1.614 for dx= 0.006
    xmin_wall= 0 ; xmax_wall=width; ymin_wall=0 ; ymax_wall=0.8  ; zmin_wall=0; zmax_wall=0 

    boundary_coords= create_boundary_coords(xmin_wall,xmax_wall,ymin_wall,ymax_wall,zmin_wall,zmax_wall,dx)
    
    prescribed_wall_velocity=[0,0]; prescribed_wall_acceleration=[0,0]
    initial_wall_velocity=[0,0]; 

    particles,wall_particle_indices = boundary_initialization(particles,boundary_coords,flag,prescribed_wall_velocity, initial_wall_velocity)    
    #particles[:,0]
    """ for i in range(len(fluid_particle_indices)):

        #initial pressure of fluid particles, hydrostatic (rho*g*h)
        particles[fluid_particle_indices[i]][8]=rho_0[0]*abs(g)*(height-particles[fluid_particle_indices[i]][1])
       
        #initial density of fluid particles a
        particles[fluid_particle_indices[i]][5] = rho_0[0]*((particles[fluid_particle_indices[i]][8]-Xi[0])/p_0[0] +1)**(1/gamma[0])
    """
    
    particles=np.array(particles)
    particles[:number_fluid_particles,8]=rho_0[0]*abs(g)*(height-particles[:number_fluid_particles,1])
    particles[:number_fluid_particles,5]=rho_0[0]*(((particles[:number_fluid_particles,8]-Xi[0])/p_0[0]) +1)**(1/gamma[0])
    #print(particles[:10,5])
    # breakpoint()

    return particles,rho_0,gamma,c_0,p_0,Xi,mu,alpha, prescribed_wall_acceleration, fluid_particle_indices, wall_particle_indices, g, height