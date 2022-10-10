import numpy as np
from phi.field._point_cloud import distribute_points

from create_particles import *


def dam_break_case(dx, d, alph):
    width = dx * np.ceil(1.61 / dx)  # width=1.614 for dx= 0.006
    bounds = Box(x=width, y=.8)
    g = -9.81
    height = 0.3
    v_max = np.sqrt(2 * abs(g) * height)

    initial_fluid_density = 1000
    density_wall = 0
    adiabatic_exp = 7  # adiabatic coefficient (pressure coefficient)
    c_0 = [0, 0]  # artificial speed of sound, c_0
    p_0 = [0, 0]
    Xi = [0, 0]  # back pressure
    mu = [0, 0]
    alpha = [0, 0]

    ###### Properties of Fluid Particles #######
    # number_fluid_particles = 5000  # useless if particles are generated uniformly
    # c_0[flag] = 10 * v_max  # v_max = 2*abs(g)*height
    # p_0[flag] = (rho_0[flag] * (c_0[flag] ** 2)) / (gamma[flag])  # reference pressure
    # Xi[flag] = 0.0 * p_0[flag]  # background pressure
    # mu[flag] = 0.01  # viscosity
    # alpha[flag] = alph  # artificial visc factor

    fluid_coords = pack_dims(math.meshgrid(x=71, y=71), 'x,y', instance('particles')) * (.6, .3) / 71
    fluid_particles = PointCloud(Sphere(fluid_coords, radius=dx))
    fluid_velocity = fluid_particles * (0, 0)
    fluid_mass = density_fluid * dx**2

    ###### Properties of Wall particles #####
    boundary_coords = create_boundary_coords(xmin_wall, xmax_wall, ymin_wall, ymax_wall, zmin_wall, zmax_wall, dx)
    boundary_particles = PointCloud(Sphere(boundary_coords, radius=dx))
    boundary_velocity = boundary_particles * (0, 0)

    # particles, wall_particle_indices = boundary_initialization(particles, boundary_coords, flag, prescribed_wall_velocity, initial_wall_velocity)
    # particles[:,0]
    """ for i in range(len(fluid_particle_indices)):

        #initial pressure of fluid particles, hydrostatic (rho*g*h)
        particles[fluid_particle_indices[i]][8]=rho_0[0]*abs(g)*(height-particles[fluid_particle_indices[i]][1])
       
        #initial density of fluid particles a
        particles[fluid_particle_indices[i]][5] = rho_0[0]*((particles[fluid_particle_indices[i]][8]-Xi[0])/p_0[0] +1)**(1/gamma[0])
    """
    fluid_density = initial_fluid_density * abs(g) * (height - fluid_particles.points['y'])
    pressure = initial_fluid_density *
    particles[:number_fluid_particles, 5] = rho_0[0] * (((particles[:number_fluid_particles, 8] - Xi[0]) / p_0[0]) + 1) ** (1 / gamma[0])
    # print(particles[:10,5])
    # breakpoint()

    return particles, rho_0, gamma, c_0, p_0, Xi, mu, alpha, prescribed_wall_acceleration, fluid_particle_indices, wall_particle_indices, g, height
