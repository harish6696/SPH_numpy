#import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)
from adaptive_time_step import *
#from dam_break_parameters import *
from free_fall_parameters import *
from boundary_update import *
from velocity_derivative import *
from density_derivative import *
import csv
import pandas as pd
from openpyxl import Workbook
#from phi.flow import *
from phi.jax.flow import *
import matplotlib
#math.set_global_precision(64)

def step(fluid_particles,wall_particles,fluid_particle_acceleration,fluid_particle_pressure,wall_particle_pressure,fluid_initial_density,fluid_particle_density,wall_particle_density, fluid_particle_mass, \
    fluid_adiabatic_exp, fluid_c_0,fluid_p_0,fluid_Xi,fluid_alpha, dt,n_dt,d,r_c,h,g,dx):

    fluid_particle_velocity=math.expand(fluid_particles.values, instance(fluid_particles))

    fluid_particle_velocity = fluid_particle_velocity + (dt/2)*fluid_particle_acceleration #initial fluid_particle_velocity obtained from dam_break_case function
    fluid_particle_position = fluid_particles.points + (dt/2)*fluid_particle_velocity

    #update the point cloud. positions are stored in elements of the pointcloudand velocities are stored in values of the pointcloud
    fluid_particles = fluid_particles.with_values(fluid_particle_velocity)
    fluid_particles = fluid_particles.with_elements(fluid_particle_position)
    

    wall_particle_pressure, wall_particles = boundary_update(fluid_particles,wall_particles,fluid_particle_pressure,fluid_particle_density,d,r_c,h,g)
    
    drho_by_dt= calculate_density_derivative(fluid_particles, wall_particles, fluid_particle_density,fluid_particle_pressure,fluid_particle_mass,fluid_initial_density,fluid_Xi,fluid_adiabatic_exp,fluid_p_0, h, d, r_c, dx)
    
    #Density Update: 
    fluid_particle_density = fluid_particle_density + (dt*drho_by_dt)

    #Pressure Update:  p_0[0] * ((particles[:len(fluid_particle_indices),5]/rho_0[0])**gamma[0] - 1 ) + Xi[0] 
    fluid_particle_pressure = fluid_p_0 *((fluid_particle_density/fluid_initial_density)**fluid_adiabatic_exp -1) + fluid_Xi

    fluid_particle_position = fluid_particles.points + (dt/2)*fluid_particle_velocity
    # print('fluid_pos before acc calc 2')    
    fluid_particles = fluid_particles.with_elements(fluid_particle_position)

    wall_particle_pressure, wall_particles= boundary_update(fluid_particles,wall_particles,fluid_particle_pressure,fluid_particle_density,d,r_c,h,g)

    #Calculate Acceleration
    fluid_particle_acceleration = calculate_acceleration(fluid_particles, wall_particles, fluid_particle_density,fluid_particle_pressure,wall_particle_pressure,fluid_particle_mass,fluid_initial_density,fluid_Xi,fluid_adiabatic_exp,fluid_p_0, h, d, r_c, dx, fluid_alpha,fluid_c_0,g,n_dt)
    
    fluid_particle_velocity = fluid_particle_velocity + (dt/2)*fluid_particle_acceleration
    fluid_particles = fluid_particles.with_values(fluid_particle_velocity)

    return fluid_particles,wall_particles, fluid_particle_acceleration, fluid_particle_pressure, fluid_particle_density, wall_particle_pressure

step = jit_compile(step, auxiliary_args='d')

def main():
    
    d=2 # dimension of the problem (indirectly dimension of the kernal function) 
    dx=0.006 #distance between particles
    h=dx #cut off radius
    r_c=3*h   #Quintic Spline
    alpha=0.02 #viscosity coefficient value of water
   
    #Generate paramters for dam break case
    # fluid_particles,wall_particles, fluid_initial_density,wall_initial_density, \
    # fluid_particle_density,wall_particle_density, fluid_particle_pressure, wall_particle_pressure,  \
    # fluid_particle_mass, \
    # fluid_adiabatic_exp,wall_adiabatic_exp, fluid_c_0,wall_c_0,fluid_p_0,wall_p_0,fluid_Xi,wall_Xi,fluid_alpha,wall_alpha, \
    # g, Height= dam_break_case(dx,d,alpha)

    fluid_particles,wall_particles, fluid_initial_density,wall_initial_density, \
    fluid_particle_density,wall_particle_density, fluid_particle_pressure, wall_particle_pressure,  \
    fluid_particle_mass, \
    fluid_adiabatic_exp,wall_adiabatic_exp, fluid_c_0,wall_c_0,fluid_p_0,wall_p_0,fluid_Xi,wall_Xi,fluid_alpha,wall_alpha, \
    g, Height= free_fall_case(dx,d,alpha)

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
    # if abs(H - Height) >= 1.0e-6:
    #     print("wrong height specified,please input dx again")
    #     exit()
    
    #reference_value
    v_ref = math.sqrt(2 *abs(g) * H )
    t_ref = H / v_ref 

    time_nondim = 5    # nondimensional running time (t/t_ref)
    t_end = time_nondim * t_ref
    print("Total simulation time is: "+str(t_end))
    print("Total number of fluid particles: "+ str(fluid_particles.elements.center.particles.size))
    print("Total number of wall particles: "+ str(wall_particles.elements.center.particles.size))
    
    fluid_traj=[]
    while t <=t_end :  ##Replace with t < t_end once everything is working

        print("Simulation progress: "+ str((100*(t/t_end)))+ " Time step "+ str(n_dt))
        
        dt = time_step_size(fluid_c_0,fluid_particles, wall_particles,h,fluid_alpha,d,g)
        print("Time step size is: "+str(dt))
        
        t=t+dt
        n_dt=n_dt+1

        if n_dt == 1:

            wall_particle_pressure, wall_particles =boundary_update(fluid_particles,wall_particles,fluid_particle_pressure,fluid_particle_density,d,r_c,h,g)
                        
            fluid_particle_acceleration =calculate_acceleration(fluid_particles, wall_particles, fluid_particle_density,fluid_particle_pressure,wall_particle_pressure,fluid_particle_mass,fluid_initial_density,fluid_Xi,fluid_adiabatic_exp,fluid_p_0, h, d, r_c, dx, fluid_alpha,fluid_c_0,g,n_dt)

            workbook_y = Workbook()
            workbook_y.save("y_pos.xlsx")

            workbook_x = Workbook()
            workbook_x.save("x_pos.xlsx")

            pos_x = numpy.asarray(math.concat([fluid_particles.points['x'],wall_particles.points['x']],'particles'))
            pos_y = numpy.asarray(math.concat([fluid_particles.points['y'],wall_particles.points['y']],'particles'))
            df_x=pd.DataFrame(pos_x)
            df_y=pd.DataFrame(pos_y)

            with pd.ExcelWriter("x_pos.xlsx",mode="a",engine="openpyxl",if_sheet_exists="overlay") as writer_x:
                df_x.to_excel(writer_x, sheet_name="Sheet",header=None, startcol=writer_x.sheets["Sheet"].max_column,index=False)
                writer_x.save()

            with pd.ExcelWriter("y_pos.xlsx",mode="a",engine="openpyxl",if_sheet_exists="overlay") as writer_y:
                df_y.to_excel(writer_y, sheet_name="Sheet",header=None, startcol=writer_y.sheets["Sheet"].max_column,index=False)
                writer_y.save()
        
        #*******************************************************************************************
        # Step-Function: Calculate the parameters for one step
        #*******************************************************************************************
        fluid_particles,wall_particles, fluid_particle_acceleration, fluid_particle_pressure, fluid_particle_density, wall_particle_pressure = step(fluid_particles,wall_particles,fluid_particle_acceleration,fluid_particle_pressure,wall_particle_pressure,fluid_initial_density,fluid_particle_density,wall_particle_density, fluid_particle_mass, \
        fluid_adiabatic_exp, fluid_c_0,fluid_p_0,fluid_Xi,fluid_alpha, dt,n_dt, d,r_c,h,g,dx)
        #*******************************************************************************************
        
        
        #fluid_traj.append(fluid_particles.points)
        
        if(n_dt%100==0):
            pos_x = numpy.asarray(math.concat([fluid_particles.points['x'],wall_particles.points['x']],'particles'))
            pos_y = numpy.asarray(math.concat([fluid_particles.points['y'],wall_particles.points['y']],'particles'))
            df_x=pd.DataFrame(pos_x)
            df_y=pd.DataFrame(pos_y)

            with pd.ExcelWriter("x_pos.xlsx",mode="a",engine="openpyxl",if_sheet_exists="overlay") as writer_x:
                df_x.to_excel(writer_x, sheet_name="Sheet",header=None, startcol=writer_x.sheets["Sheet"].max_column,index=False)
                writer_x.save()

            with pd.ExcelWriter("y_pos.xlsx",mode="a",engine="openpyxl",if_sheet_exists="overlay") as writer_y:
                df_y.to_excel(writer_y, sheet_name="Sheet",header=None, startcol=writer_y.sheets["Sheet"].max_column,index=False)
                writer_y.save()
    
    
    #**************************
    # Save animation
    #**************************
    #xs = math.stack(fluid_particles, batch('time'))
    #a:matplotlib.animation.FuncAnimation=vis.plot(fluid_traj ,animate ='list')
    #print('\n \n ')
    #print(a)
    
    #a.save('trial.mp4')

if __name__== "__main__":
    main()



