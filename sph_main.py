import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from adaptive_time_step import *
from dam_break_parameters import *
from boundary_update import *
from velocity_derivative import *
from density_derivative import *
import csv
import pandas as pd
from openpyxl import Workbook

def main():
    
    d=2 # dimension of the problem (indirectly dimension of the kernal function) 
    dx=0.006 #distance between particles
    h=dx #cut off radius
    r_c=3*h   #Quintic Spline
    alpha=0.02 #viscosity coefficient value of water
    
    #Generate paramters for dam break case
    fluid_particles,wall_particles, fluid_initial_density,wall_initial_density, \
    fluid_particle_density,wall_density, fluid_particle_pressure, wall_pressure,  \
    fluid_particle_velocity,wall_initial_velocity, wall_prescribed_velocity, fluid_particle_mass, \
    fluid_adiabatic_exp,wall_adiabatic_exp, fluid_c_0,wall_c_0,fluid_p_0,wall_p_0,fluid_Xi,wall_Xi,fluid_alpha,wall_alpha, \
    g, Height= dam_break_case(dx,d,alpha)

    
    t=0
    n_dt=0

    """ 
    
    particles are objects of the particle class contain both solid and fluid particles
    rho_0: Reference density of the particles
    gamma: Adiabatic constant =7 for water
    c_0: Artificial speed of sound = 10* sqrt(2*g*H)
    p_0: Reference pressure
    Xi: background pressure =0
    mu: 
    alpha: viscosity parameter
    a_wall: 
    int_fluid: array storing id of fluid particles
    int_boundary: array storing id of boundary particles 
    g:             Acc due to gravity [0,-9.81]
    Height:        Height of the dam problem (0.3) """

    #Acceleration
    
    #H = np.max(particles[:len(fluid_particle_indices),1])-np.min(particles[:len(fluid_particle_indices),1])+dx
    H=math.max(fluid_particles.points['y'])-math.min(fluid_particles.points['y'])+dx
    print('actual height of water column is: ' +str(H))
   
    if abs(H - Height) >= 1.0e-6:
        print("wrong height specified,please input dx again")
        exit()
    
    #reference_values
    
    v_ref = np.sqrt(2 *abs(g) * H )
    t_ref = H / v_ref 
    
    time_nondim = 5    # nondimensional running time (t/t_ref)
    t_end = time_nondim * t_ref
    print("Total simulation time is: "+str(t_end))
    #breakpoint()
    print("Total number of fluid particles: "+ str(len(fluid_particle_indices)))
    print("Total number of wall particles: "+ str(len(wall_particle_indices)))
    while t<=t_end:

        print("Simulation progress: "+ str((100*(t/t_end)))+ " Time step "+ str(n_dt))

        #DO SIMULATION   
        dt = time_step_size(fluid_c_0,particles,h,alpha,d,g)
        #print("Time step size is: "+str(dt))
        t=t+dt
        n_dt=n_dt+1
        #print(t)
        if n_dt == 1:
            #BOUNDARY UPDATE FUNCTION
            wall_particle_pressure=boundary_update(fluid_particles, wall_particles, fluid_particle_pressure,d,r_c,h,g)

            #VELOCITY DERIVATIVE FUNCTION
            a_x,a_y=calculate_acceleration(fluid_particles, wall_particles, fluid_particle_density,fluid_particle_pressure,wall_particle_pressure,fluid_particle_mass,fluid_particle_velocity,fluid_initial_density,fluid_Xi,fluid_adiabatic_exp,fluid_p_0, h, d, r_c, dx, fluid_alpha,fluid_c_0,g)

                
if __name__== "__main__":
    main()



