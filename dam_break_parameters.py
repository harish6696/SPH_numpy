import numpy as np
from phi.field._point_cloud import distribute_points

from create_particles import *   #rendered useless
from phi.flow import *

def dam_break_case(dx, d, alph):
    width = dx * np.ceil(1.61 / dx)  # width=1.614 for dx= 0.006
    g = -9.81
    height = 0.3
    v_max = np.sqrt(2 * abs(g) * height)
    
    ###### Properties of Fluid Particles #######
    fluid_initial_density = 1000
    fluid_adiabatic_exp = 7  # adiabatic coefficient (pressure coefficient)
    fluid_c_0=10*v_max # artificial speed of sound c_0 and v_max = 2*abs(g)*height
    fluid_p_0=(fluid_initial_density*((fluid_c_0)**2))/fluid_adiabatic_exp  # reference pressure 
    fluid_Xi=0  # background pressure
    fluid_mu=0.01  # viscosity
    fluid_alpha=alph  # artificial visc factor

    fluid_coords = pack_dims(math.meshgrid(x=100, y=50), 'x,y', instance('particles')) * (0.6/100, 0.3/50) + (0.003,0.003)  # 5000 fluid particle coordinates created     
    fluid_particles = PointCloud(Sphere(fluid_coords, radius=0.002))  #"""is this radius only for visualization?????????????"""
    fluid_velocity = fluid_particles * (0, 0)
    
    #print(fluid_velocity.values)
    # breakpoint()

    fluid_particle_mass = fluid_initial_density * dx**d
    fluid_pressure=math.zeros(instance(fluid_coords))  

    ###### Properties of Wall particles #####
    boundary_initial_density =0 
    boundary_adiabatic_exp = 0  # adiabatic coefficient (pressure coefficient)
    boundary_c_0=0 # artificial speed of sound c_0
    boundary_p_0=0  # reference pressure 
    boundary_Xi=0  # background pressure
    boundary_mu=0  # viscosity
    boundary_alpha=0  # artificial visc factor

    left_boundary_coords = pack_dims(math.meshgrid(x=3, y=134), 'x,y', instance('particles')) * ( (0.018/3), (0.804/134) ) + (-0.015, 0.003)  
    #print(f"{left_boundary_coords:full:shape}")
    right_boundary_coords = pack_dims(math.meshgrid(x=3, y=134), 'x,y', instance('particles')) * ( (0.018/3), (0.804/134) ) + (1.617, 0.003)  
    #print(f"{right_boundary_coords:full:shape}")
    center_boundary_coords = pack_dims(math.meshgrid(x=275, y=3), 'x,y', instance('particles')) * ( (1.65/275), (0.018/3) ) + (-0.015, -0.015)  
    #print(f"{center_boundary_coords:full:shape}")
    
    boundary_coords=math.concat([left_boundary_coords, right_boundary_coords,center_boundary_coords], 'particles') #1629 wall particles
    boundary_particles = PointCloud(Sphere(boundary_coords, radius=0.002))

    boundary_prescribed_velocity = boundary_particles * (0, 0)
    boundary_initial_velocity=boundary_particles * (0, 0)
    boundary_pressure=math.zeros(instance(boundary_coords)) 
    boundary_density=math.zeros(instance(boundary_coords))  

    #particles[:number_fluid_particles,8]=rho_0[0]*abs(g)*(height-particles[:number_fluid_particles,1])
    fluid_pressure = fluid_initial_density * abs(g) * (height - fluid_particles.points['y']) 
  
    #particles[:number_fluid_particles, 5] = rho_0[0] * (((particles[:number_fluid_particles, 8] - Xi[0]) / p_0[0]) + 1) ** (1 / gamma[0])
    fluid_density = fluid_initial_density * (((fluid_pressure-fluid_Xi)/fluid_p_0)+1)**(1/fluid_adiabatic_exp)

    return fluid_particles,boundary_particles, fluid_initial_density,boundary_initial_density, \
    fluid_density, boundary_density,fluid_pressure, boundary_pressure,  \
    fluid_velocity,boundary_initial_velocity, boundary_prescribed_velocity, fluid_particle_mass, \
    fluid_adiabatic_exp,boundary_adiabatic_exp, fluid_c_0,boundary_c_0,fluid_p_0,boundary_p_0,fluid_Xi,boundary_Xi,fluid_alpha,boundary_alpha, \
    g, height




    #print(fluid_coords.native('x,y'))
    #print(f"{fluid_coords:full:shape}")