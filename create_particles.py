from generate_random_points import *  # returns randomly generated points with a minimum separation of 0.06
import matplotlib.pyplot as plt
import numpy as np
from phi.flow import *


def create_fluid_coords(box: Box, num_fluid_particles: int, dx: float):
    ##uncomment below for generating random fluid
    # fluid_coords= return_points(xmin, xmax,ymin,ymax, zmin, zmax, num_fluid_particles)

    # create fluid particle coordinates
    no_rows = np.ceil(abs(ymax - ymin) / dx)  # rows of fluid particle
    no_cols = np.ceil(abs(xmax - xmin) / dx)  # cols of fluid

    no_rows = int(no_rows)
    no_cols = int(no_cols)

    fluid_coords = np.zeros((no_rows * no_cols, 3))  # Coordinate of fluid particle (2 array)

    # Fluid particle coordinate initialization
    for n in range(no_rows):
        for m in range(no_cols):
            # x-coordin:te of fluid particle. f_lowleft[1,1]=0.0 and f_lowleft[1,2]=0.0
            fluid_coords[m + (n) * no_cols, 0] = xmin + (m - 1 / 2 + 1) * dx
            # y-coordinate of fluid particle
            fluid_coords[m + (n) * no_cols, 1] = ymin + (n - 1 / 2 + 1) * dx
            # In this nested loop: Saves coordinates of fluid particles starting from lower left corner(0,0). """

    return fluid_coords


def create_boundary_coords(xmin_wall, xmax_wall, ymin_wall, ymax_wall, zmin_wall, zmax_wall, dx):
    # number of boundary particles for kernel support
    k = 3  # quintic spline 3 particles per wall

    # create boundary particles vector
    no_rows = int(np.ceil(abs(ymax_wall - ymin_wall) / dx))
    no_cols = int(np.ceil(abs(xmax_wall - xmin_wall) / dx + 2 * k))

    # print("Rows and Cols are: " + str(no_rows)+" "+str(no_cols)+ " \n")

    boundary_coords = np.zeros((2 * no_rows * k + k * no_cols, 3))  # stores the x,y,z coordinates of the boundary particles

    for n in range(no_rows):
        for m in range(2 * k):
            # x-coordinate
            if m < k:  # (m:0-->2)
                boundary_coords[(m + (n) * 2 * k), 0] = xmin_wall - (k * dx - dx / 2) + ((m) * dx)
            else:  # (m:3-->5)
                boundary_coords[(m + (n) * 2 * k), 0] = xmax_wall + (dx / 2) + ((m - k) * dx)

            # y-coordinate
            boundary_coords[(m + (n) * 2 * k), 1] = ymin_wall + ((n - 1 / 2 + 1) * dx)

    for n in range(k):
        for m in range(no_cols):
            # x-coordinate
            boundary_coords[(2 * no_rows * k + m + (n) * no_cols), 0] = xmin_wall - (k * dx - dx / 2) + (m) * dx
            # y-coordinate
            boundary_coords[(2 * no_rows * k + m + (n) * no_cols), 1] = ymin_wall - (n - 1 / 2 + 1) * dx

    # print(np.array(boundary_coords).shape)
    # np.savetxt('data.csv', np.asarray(boundary_coords), delimiter=',')
    return boundary_coords


def fluid_initialization(particles, fluid_coords, flag, initial_fluid_velocity, rho_0_fluid, dx, d):
    number_of_fluid_particles = len(particles) + len(fluid_coords)
    fluid_particle_indices = [];
    count = 0
    for n in range(number_of_fluid_particles):
        temp = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        temp[0] = fluid_coords[n][0]  # x coordinate
        temp[1] = fluid_coords[n][1]  # y coordinate
        temp[2] = fluid_coords[n][2]  # z coordinate
        temp[3] = flag  # flag indicating fluid(0) and wall(1)
        temp[5] = rho_0_fluid  # initial density for ALL fluid particles. will be changed later
        temp[4] = temp[5] * (dx ** d)  # mass of the particle
        temp[6] = initial_fluid_velocity[0]  # initial x velocity of the particle
        temp[7] = initial_fluid_velocity[1]  # initial y velocity of the particle
        temp[8] = 0.0  # pressure acting on the particle(will be calculated from density)
        particles.append(temp)
        fluid_particle_indices.append(count);
        count = count + 1
    return particles, fluid_particle_indices


def boundary_initialization(particles, boundary_coords, flag, prescribed_wall_velocity, initial_wall_velocity):
    number_of_wall_particles = len(boundary_coords)
    count = len(particles)  # starts count from the last fluid particle
    wall_particle_indices = []
    for n in range(number_of_wall_particles):
        temp = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        temp[0] = boundary_coords[n][0]  # x coordinate
        temp[1] = boundary_coords[n][1]  # y coordinate
        temp[2] = boundary_coords[n][2]  # z coordinate
        temp[3] = flag  # flag indicating fluid(0) and wall(1)
        temp[4] = prescribed_wall_velocity[0]  # prescribed wall x velocity of the wall particle
        temp[5] = prescribed_wall_velocity[1]  # prescribed wall y velocity of the wall particle
        temp[6] = initial_wall_velocity[0]  # initial x velocity of the wall particle
        temp[7] = initial_wall_velocity[1]  # initial y velocity of the wall particle
        temp[8] = 0.0  # pressure acting on the particle(will be calculated from density)
        particles.append(temp)
        wall_particle_indices.append(count);
        count += 1;
    return particles, wall_particle_indices


if __name__ == "__main__":
    boundary_coords = create_boundary_coords(0, 1.614, 0, 0.8, 0, 0, 0.006)
    fluid_coords = create_fluid_coords(0, 0.6, 0, 0.3, 0, 0, 5000, 0.006)

    particles = []
    flag = 0;
    initial_fluid_velocity = [0, 0];
    rho_0_fluid = 1000;
    dx = 0.006;
    d = 2
    particles, fluid_particle_indices = fluid_initialization(particles, fluid_coords, flag, initial_fluid_velocity, rho_0_fluid, dx, d)

    flag = 1;
    prescribed_wall_velocity = [0, 0];
    initial_wall_velocity = [0, 0];
    particles, wall_particle_indices = boundary_initialization(particles, boundary_coords, flag, prescribed_wall_velocity, initial_wall_velocity)

    plt.scatter(boundary_coords[:, 0], boundary_coords[:, 1], s=5)

    plt.scatter(fluid_coords[:, 0], fluid_coords[:, 1], s=5)
    plt.show()
