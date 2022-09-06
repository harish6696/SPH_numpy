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
    particles,rho_0,gamma,c_0,p_0,Xi,mu,alpha, a_wall, fluid_particle_indices, wall_particle_indices, g, Height = dam_break_case(dx,d,alpha)
    particles=np.array(particles)
    
    t=0
    n_dt=0

    """ 
    plt.scatter(fluid_coords[:,0],fluid_coords[:,1],s=5)
    
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

    H= np.max(particles[:len(fluid_particle_indices),1])-np.min(particles[:len(fluid_particle_indices),1])+dx
    print("actual height of water column from particles: "+ str(H))
    
    a_x = np.zeros(len(fluid_particle_indices))
    a_y = np.zeros(len(fluid_particle_indices))

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
        dt = time_step_size(c_0,particles,h,alpha,d,g)
        #print("Time step size is: "+str(dt))
        t=t+dt
        n_dt=n_dt+1
        #print(t)
        if n_dt == 1:
            #BOUNDARY UPDATE FUNCTION
            particles[len(fluid_particle_indices):,8],particles[:,6],particles[:,7]=boundary_update(particles, fluid_particle_indices,wall_particle_indices,d,r_c,h,g)
            #print(particles[5000:5000+len(wall_particle_indices),8])
            #print("/n")
            #print(len(particles[5000:5000+len(wall_particle_indices),8]))
            
            #pd.DataFrame(particles[len(fluid_particle_indices):len(fluid_particle_indices)+len(wall_particle_indices),8]).to_csv('new_densities_ndt=1.csv')
            #pd.DataFrame(particles[:,7]).to_csv('new_wall_vel_y_ndt=1.csv')
            
            
            
            #VELOCITY DERIVATIVE FUNCTION
            a_x,a_y=calculate_acceleration(particles, fluid_particle_indices,h, d, r_c,alpha, c_0, g, rho_0, p_0, Xi,gamma)
            #pd.DataFrame(a_x).to_csv('acc_x_new.csv')
            #pd.DataFrame(a_y).to_csv('acc_y_new.csv')
            
            #print("Printed accelerations")
            #breakpoint()
            #pd.DataFrame(particles[:,1]).to_csv('y_pos_initial.csv')
            #df=pd.DataFrame(particles[:,1])
            workbook_y = Workbook()
            workbook_y.save("y_pos.xlsx")

            workbook_x = Workbook()
            workbook_x.save("x_pos.xlsx")

            df_x=pd.DataFrame(particles[:,0])
            df_y=pd.DataFrame(particles[:,1])

            with pd.ExcelWriter("x_pos.xlsx",mode="a",engine="openpyxl",if_sheet_exists="overlay") as writer_x:
                df_x.to_excel(writer_x, sheet_name="Sheet",header=None, startcol=writer_x.sheets["Sheet"].max_column,index=False)
                writer_x.save()

            with pd.ExcelWriter("y_pos.xlsx",mode="a",engine="openpyxl",if_sheet_exists="overlay") as writer_y:
                df_y.to_excel(writer_y, sheet_name="Sheet",header=None, startcol=writer_y.sheets["Sheet"].max_column,index=False)
                writer_y.save()
        
        # Velocity Update of fluid particles
        particles[:len(fluid_particle_indices),6] = particles[:len(fluid_particle_indices),6] + ((dt/2) * a_x)
        particles[:len(fluid_particle_indices),7] = particles[:len(fluid_particle_indices),7] + ((dt/2) * a_y)

        # Position Update of fluid particles
        particles[:len(fluid_particle_indices),0] = particles[:len(fluid_particle_indices),0] + ((dt/2) * particles[:len(fluid_particle_indices),6])
        particles[:len(fluid_particle_indices),1] = particles[:len(fluid_particle_indices),1] + ((dt/2) * particles[:len(fluid_particle_indices),7])

        ## Density and Pressure update 
        # update boundary(wall) particles pressure, velocity 
        particles[len(fluid_particle_indices):,8],particles[:,6],particles[:,7] = boundary_update(particles,fluid_particle_indices,wall_particle_indices,d,r_c,h,g)
        drho_by_dt = density_derivative(particles, fluid_particle_indices, h, d, r_c,rho_0, p_0, Xi, gamma, dx)
        # update fluid particles density(5) and pressure (8)
        particles[:len(fluid_particle_indices),5] = particles[:len(fluid_particle_indices),5] + (dt * drho_by_dt )
        particles[:len(fluid_particle_indices),8] = p_0[0] * ((particles[:len(fluid_particle_indices),5]/rho_0[0])**gamma[0] - 1 ) + Xi[0]  #rho_0 is reference density, density of water   
                                                                                                        #p_0 is ref pressure = (rho_0* c^2) / gamma 
        # Position Update of fluid particles
        particles[:len(fluid_particle_indices),0] = particles[:len(fluid_particle_indices),0] + ((dt/2) * particles[:len(fluid_particle_indices),6])
        particles[:len(fluid_particle_indices),1] = particles[:len(fluid_particle_indices),1] + ((dt/2) * particles[:len(fluid_particle_indices),7])

        ### Velocity update boundary
        particles[len(fluid_particle_indices):,8],particles[:,6],particles[:,7] = boundary_update(particles, fluid_particle_indices, wall_particle_indices, d,r_c, h, g)
        #acceleration of next time step
        a_x, a_y = calculate_acceleration(particles, fluid_particle_indices, h, d, r_c,alpha, c_0, g, rho_0, p_0, Xi,gamma)
        #update velocities using acceleration of next time step of fluid particles
        particles[:len(fluid_particle_indices),6] = particles[:len(fluid_particle_indices),6] + ((dt/2) * a_x)
        particles[:len(fluid_particle_indices),7] = particles[:len(fluid_particle_indices),7] + ((dt/2) * a_y)
            
        if(n_dt%100==0):
            df_x=pd.DataFrame(particles[:,0])
            df_y=pd.DataFrame(particles[:,1])

            with pd.ExcelWriter("x_pos.xlsx",mode="a",engine="openpyxl",if_sheet_exists="overlay") as writer_x:
                df_x.to_excel(writer_x, sheet_name="Sheet",header=None, startcol=writer_x.sheets["Sheet"].max_column,index=False)
                writer_x.save()

            with pd.ExcelWriter("y_pos.xlsx",mode="a",engine="openpyxl",if_sheet_exists="overlay") as writer_y:
                df_y.to_excel(writer_y, sheet_name="Sheet",header=None, startcol=writer_y.sheets["Sheet"].max_column,index=False)
                writer_y.save()
                
if __name__== "__main__":
    main()



