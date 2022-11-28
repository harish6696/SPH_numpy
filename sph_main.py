#import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)
from adaptive_time_step import *
from dam_break_parameters import *
from boundary_update import *
from velocity_derivative import *
from density_derivative import *
import csv
import pandas as pd
from openpyxl import Workbook
from phi import *
#math.set_global_precision(64)
def main():
    
    d=2 # dimension of the problem (indirectly dimension of the kernal function) 
    dx=0.006 #distance between particles
    h=dx #cut off radius
    r_c=3*h   #Quintic Spline
    alpha=0.02 #viscosity coefficient value of water

    ##############COMMENT LATER
    r_c = 0.04
    
    #Generate paramters for dam break case
    fluid_particles,wall_particles, fluid_initial_density,wall_initial_density, \
    fluid_particle_density,wall_particle_density, fluid_particle_pressure, wall_particle_pressure,  \
    fluid_particle_mass, \
    fluid_adiabatic_exp,wall_adiabatic_exp, fluid_c_0,wall_c_0,fluid_p_0,wall_p_0,fluid_Xi,wall_Xi,fluid_alpha,wall_alpha, \
    g, Height= dam_break_case(dx,d,alpha)

    #wall_particle_velocity = math.zeros(instance(wall_particles.points))
    #print(fluid_particle_velocity)
    #breakpoint()
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
    print('Actual height of water column is: ' +str(H))
   
    if abs(H - Height) >= 1.0e-6:
        print("wrong height specified,please input dx again")
        #exit()
    
    #reference_values
    
    v_ref = math.sqrt(2 *abs(g) * H )
    t_ref = H / v_ref 
    
    time_nondim = 5    # nondimensional running time (t/t_ref)
    t_end = time_nondim * t_ref
    print("Total simulation time is: "+str(t_end))

    print("Total number of fluid particles: "+ str(fluid_particles.elements.center.particles.size))
    print("Total number of wall particles: "+ str(wall_particles.elements.center.particles.size))

    while n_dt <=2 :  ##Replace with t < t_end once everything is working

        #print("Simulation progress: "+ str((100*(t/t_end)))+ " Time step "+ str(n_dt))
        
        #DO SIMULATION   
        #dt = time_step_size(fluid_c_0,particles,h,alpha,d,g)
        dt = time_step_size(fluid_c_0,fluid_particles, wall_particles,h,fluid_alpha,d,g)
        print("Time step size is: "+str(dt))
        
        
        t=t+dt
        n_dt=n_dt+1
        
        if n_dt == 1:
            #BOUNDARY UPDATE FUNCTION
            wall_particle_pressure, wall_particles =boundary_update(fluid_particles,wall_particles,fluid_particle_pressure,fluid_particle_density,d,r_c,h,g)
            #print('after n_dt ==1 boundary update')
            #math.print(wall_particle_pressure)
            #math.print(wall_particles.values)
            
            #VELOCITY DERIVATIVE FUNCTION
            fluid_particle_acceleration =calculate_acceleration(fluid_particles, wall_particles, fluid_particle_density,fluid_particle_pressure,wall_particle_pressure,fluid_particle_mass,fluid_initial_density,fluid_Xi,fluid_adiabatic_exp,fluid_p_0, h, d, r_c, dx, fluid_alpha,fluid_c_0,g)
            #print('after n_dt ==1 calculate acceleration')
            #math.print(fluid_particle_acceleration)
            #print(fluid_particle_acceleration)
            #breakpoint()
        """
        print('\n \n')
        print('positions 1')
        math.print(fluid_particles)       #pointcloud
        print('velocity 1')
        math.print(fluid_particle_velocity)    # pointcloud
        print('acceleration 1')
        print(fluid_particle_acceleration) # just a 2d vector
        """
        
        fluid_particle_velocity=math.expand(fluid_particles.values, instance(fluid_particles))
        #fluid_particle_velocity update
        fluid_particle_velocity = fluid_particle_velocity + (dt/2)*fluid_particle_acceleration #initial fluid_particle_velocity obtained from dam_break_case function
        #fluid_particle_position_update
        fluid_particle_position = fluid_particles.points + (dt/2)*fluid_particle_velocity
        
        fluid_particles = fluid_particles.with_values(fluid_particle_velocity)
        fluid_particles = fluid_particles.with_elements(fluid_particle_position)
        #print('after update')
        #math.print(fluid_particles)
        #print('wall_velocity befpre')
        #math.print(wall_particles)
   
        wall_particle_pressure, wall_particles = boundary_update(fluid_particles,wall_particles,fluid_particle_pressure,fluid_particle_density,d,r_c,h,g)

        
        drho_by_dt= calculate_density_derivative(fluid_particles, wall_particles, fluid_particle_density,fluid_particle_pressure,fluid_particle_mass,fluid_initial_density,fluid_Xi,fluid_adiabatic_exp,fluid_p_0, h, d, r_c, dx)
        
        

        #Density Update: 
        fluid_particle_density = fluid_particle_density + (dt*drho_by_dt)

        #Pressure Update:  p_0[0] * ((particles[:len(fluid_particle_indices),5]/rho_0[0])**gamma[0] - 1 ) + Xi[0] 
        fluid_particle_pressure = fluid_p_0 *((fluid_particle_density/fluid_initial_density)**fluid_adiabatic_exp -1) + fluid_Xi

        #Useless if there is no wall velocity
        fluid_particle_position = fluid_particles.points + (dt/2)*fluid_particle_velocity

        fluid_particles = fluid_particles.with_elements(fluid_particle_position)

        wall_particle_pressure, wall_particles= boundary_update(fluid_particles,wall_particles,fluid_particle_pressure,fluid_particle_density,d,r_c,h,g)
        #print('after boundary update 2 ')
        #math.print(wall_particles)
        #breakpoint()
        
        #Calculate Acceleration
        fluid_particle_acceleration = calculate_acceleration(fluid_particles, wall_particles, fluid_particle_density,fluid_particle_pressure,wall_particle_pressure,fluid_particle_mass,fluid_initial_density,fluid_Xi,fluid_adiabatic_exp,fluid_p_0, h, d, r_c, dx, fluid_alpha,fluid_c_0,g)
        #fluid_particle_velocity update
        fluid_particle_velocity = fluid_particle_velocity + (dt/2)*fluid_particle_acceleration
        fluid_particles = fluid_particles.with_values(fluid_particle_velocity)

        print('Time step number is: '+str(n_dt))
        print('in the end')
        math.print(fluid_particles)
        #breakpoint()
        

        """         
        print('\n \n')
        print('positions 2')
        math.print(fluid_particles)
        print('velocity 2')
        math.print(fluid_particle_velocity)
        print('acceleration 2')
        print(fluid_particle_acceleration) """

if __name__== "__main__":
    main()



