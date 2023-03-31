from phi import math
from phi.math import Tensor, concat, rename_dims, instance, channel, stack, wrap, vec_length, where, divide_no_nan, zeros
from phi.field import PointCloud


def velocity_verlet(fluid: PointCloud,
                    wall: PointCloud,
                    prev_fluid_acc,
                    p_fluid,
                    p_wall,
                    d0_fluid,
                    d_fluid,
                    mass,
                    fluid_adiabatic_exp,
                    fluid_c_0,
                    fluid_p_0,
                    fluid_xi,
                    fluid_alpha,
                    max_dist,
                    gravity: Tensor,
                    kernel: str = 'wendland-c2'):
    """
    Compute one SPH time step using leapfrog Euler.

    The implementation is based on the paper [...]().

    Args:
        fluid: Pointcloud
        wall: Point cloud
        prev_fluid_acc: accleration of fluid particles from the previous time step
        p_fluid: Fluid pressure.
        p_wall: Obstacle pressure.
        d0_fluid: Fluid initial density (1000kg/m³ for water)
        d_fluid: Fluid density (here we are assuiming Weakly Compressible SPH, therefore density changes)
        mass: mass of fluid particles
        fluid_adiabatic_exp: 
        fluid_c_0: Artificial speed of sound . Approximately 10*sqrt(2*g*H)
        fluid_p_0:Initial pressure of the fluid = density*g*H
        fluid_xi: Back pressure (For free surface flows = 0)
        fluid_alpha: Artificial viscosity constant 
        max_dist: Cut off radius (3*dx)
        gravity: Gravity vector as `Tensor`

    Returns:
        fluid: Next fluid state
        wall: Obstacle particles with updated velocities
        fluid_acc:
        p_fluid: Fluid pressure.
        d_fluid: Fluid density.
        p_wall: Obstacle pressure.
    """
    dt = time_step_size(fluid_c_0, fluid, wall, fluid_alpha, gravity)
    #--neigbour distance calculation--
    x = concat([fluid.points, wall.points], 'particles')
    distances_vec = -math.pairwise_distances(x, max_dist)  #sending after performing cut-off to each function
    distances = vec_length(distances_vec)
    
    # --- Integrate velocity & position (using velocity verlet scheme)---
    if prev_fluid_acc is None:
        prev_fluid_acc = calculate_acceleration(fluid, wall, d_fluid, p_fluid, p_wall, mass, d0_fluid, fluid_xi, fluid_adiabatic_exp, fluid_p_0, max_dist, fluid_alpha, fluid_c_0, gravity, distances_vec, distances, kernel)
    fluid += (dt / 2) * prev_fluid_acc  # velocity update
    fluid = fluid.shifted((dt / 2) * fluid.values)  # position update
    # --- Density & Pressure ---
    p_wall, wall = boundary_update(fluid, wall, p_fluid, d_fluid, max_dist, gravity, distances_vec, distances, kernel)  # ToDo p_wall is not used, we can replace the assignment by _
    d_fluid += dt * calculate_density_derivative(fluid, wall, d_fluid, p_fluid, mass, d0_fluid, fluid_xi, fluid_adiabatic_exp, fluid_p_0, max_dist, distances_vec, distances, kernel)
    p_fluid = fluid_p_0 * ((d_fluid / d0_fluid) ** fluid_adiabatic_exp - 1) + fluid_xi
    # --- Integrate velocity & position ---
    fluid = fluid.shifted(dt / 2 * fluid.values)  # position update   ToDo This would require a new distance matrix since fluid positions were altered
    p_wall, wall = boundary_update(fluid, wall, p_fluid, d_fluid, max_dist, gravity, distances_vec, distances, kernel)
    fluid_acc = calculate_acceleration(fluid, wall, d_fluid, p_fluid, p_wall, mass, d0_fluid, fluid_xi, fluid_adiabatic_exp, fluid_p_0, max_dist, fluid_alpha, fluid_c_0, gravity, distances_vec, distances, kernel)
    fluid += (dt / 2) * fluid_acc  # velocity update
    return fluid, wall, fluid_acc, p_fluid, d_fluid, p_wall


def boundary_update(fluid: PointCloud, wall, p_fluid, d_fluid, max_dist, gravity: Tensor, distances_vec, distances, kernel: str):
    """Function calculates the wall pressure and updates the wall velocity"""
    #only for wall particle and their fluid neighbour
    distances_vec=distances_vec.particles[fluid.particles.size:].others[:fluid.particles.size]
    distances = distances.particles[fluid.particles.size:].others[:fluid.particles.size]

    h = 2 * fluid.elements.bounding_radius()
    ####CHANGE KERNAL HERE
    w = kernel_function(distances / h, fluid.spatial_rank, kernel) / h ** 2
    sum_p_w = w.others * p_fluid.particles
    sum_density_r_w = ((distances_vec.vector * gravity.vector) * w).others * d_fluid.particles
    sum_w = math.sum(w, 'others')
    p_wall = where(sum_w != 0, divide_no_nan((sum_density_r_w + sum_p_w), sum_w), sum_w)
    sum_v_w = w.others * fluid.values.particles
    v_wall = where(sum_w != 0, divide_no_nan(-sum_v_w, sum_w), wall.values)
    wall = wall.with_values(v_wall)
    return p_wall, wall


def calculate_density_derivative(fluid, wall, d_fluid, p_fluid, m_fluid, d0_fluid, fluid_xi, fluid_adiabatic_exp, fluid_p_0, r_c, distances_vec, distances, kernel: str):
    # wall_density=0, wall_mass=0 expected initially
    """This function implements the continuity equation by calculating and returning the rate of change of density"""
    x = concat([fluid.points, wall.points], 'particles')
    v = concat([fluid.values, wall.values], 'particles')
    d_wall = rename_dims(zeros(instance(wall)), 'particles', 'others')
    wall_mass = rename_dims(zeros(instance(wall)), 'particles', 'others')
    d_fluid = rename_dims(d_fluid, 'particles', 'others')  # fluid_particle_density and mass have initially particle dimension (i.e column vector)
    m_fluid = rename_dims(m_fluid, 'particles', 'others')
    density = concat([d_fluid, d_wall], 'others')  # 1D Scalar array of densities of all particles (now it is a row vector)
    
    mass = concat([m_fluid, wall_mass], 'others')
    
    particle_neighbour_density = where(distances == 0, 0, density)  # 0 for places which are not neighbours for particle under consideration. Stacks the particle density vertically. i.e. dupicates the densities
    particle_neighbour_mass = where(distances == 0, 0, mass)

    relative_v = v - rename_dims(v, 'particles', 'others')  # 2d matrix of ALL particle velocity
    dvx = relative_v['x'].particles[:fluid.particles.size].others[:]  # separating the x and y components
    dvy = relative_v['y'].particles[:fluid.particles.size].others[:]

    fluid_particle_relative_dist = distances.particles[:fluid.particles.size].others[:]
    dvx = where(fluid_particle_relative_dist == 0, 0, dvx)  # relative x-velocity between a fluid particle and its neighbour
    dvy = where(fluid_particle_relative_dist == 0, 0, dvy)

    h = 2 * fluid.elements.bounding_radius()

    #####CHANGE KERNEL DERIVATIVE HERE
    # 'q' = r/h = distances.particles[:fluid.particles.size] / h
    d_w_ab = kernel_derivative(distances.particles[:fluid.particles.size] / h, fluid.spatial_rank, kernel) / h ** (fluid.spatial_rank + 1)  # Consider all neighbours of fluid particles

    drx = distances_vec['x'].particles[:fluid.particles.size].others[:]
    dry = distances_vec['y'].particles[:fluid.particles.size].others[:]

    mod_dist = (distances.particles[:fluid.particles.size].others[:])
    fab_x = where(mod_dist != 0, divide_no_nan(drx, mod_dist) * d_w_ab, drx)
    fab_y = where(mod_dist != 0, divide_no_nan(dry, mod_dist) * d_w_ab, dry)
    
    fluid_fluid_neighbour_density = particle_neighbour_density.particles[:fluid.particles.size].others[:fluid.particles.size]    
    fluid_fluid_neighbour_mass= particle_neighbour_mass.particles[:fluid.particles.size].others[:fluid.particles.size]
    fluid_wall_neighbour_mass = particle_neighbour_mass.particles[:fluid.particles.size].others[fluid.particles.size:]

    # fluid_pressure is for each fluid particle
    fluid_particle_wall_neighbour_density_term = d0_fluid * ((p_fluid - fluid_xi) / fluid_p_0 + 1) ** (1 / fluid_adiabatic_exp)
    helper_matrix = distances.particles[:fluid.particles.size].others[fluid.particles.size:] #considering portion of distance-matrix corresponding to fluid_particles and wall neighbors
    helper_matrix = where(helper_matrix != 0, 1, helper_matrix)  # technically called adjacency matrix

    # ALL boundary particle neighbours for a given fluid particle will have same density
    fluid_wall_neighbour_density = helper_matrix * fluid_particle_wall_neighbour_density_term
    fluid_wall_neighbour_mass = where(helper_matrix != 0, (d0_fluid * fluid.elements.bounding_box().volume), fluid_wall_neighbour_mass)

    # joining the density and mass of the fluid and wall particle back together
    fluid_neighbour_density = concat([fluid_fluid_neighbour_density, fluid_wall_neighbour_density], 'others')
    fluid_neighbour_mass = concat([fluid_fluid_neighbour_mass, fluid_wall_neighbour_mass], 'others')

    neighbour_mass_by_density_ratio = divide_no_nan(fluid_neighbour_mass, fluid_neighbour_density)  # m_b/rho_b
    dot_product_result = (dvx * fab_x) + (dvy * fab_y)

    # Fluid_particle_density is a vector, each row of result is multiplied by the corresponding term of fluid_particle_density and then Sum along each row to get the result(drho_by_dt) for each fluid particle
    d_fluid = rename_dims(d_fluid, 'others', 'particles')
    d_rho_by_dt = math.sum((neighbour_mass_by_density_ratio * dot_product_result) * d_fluid, 'others')  # only for fluid particles
    return d_rho_by_dt


def calculate_acceleration(fluid: PointCloud, wall: PointCloud, d_fluid, p_fluid, p_wall, m_fluid, d0_fluid, fluid_xi, fluid_adiabatic_exp, fluid_p_0, r_c, fluid_alpha, fluid_c_0, gravity, distances_vec, distances, kernel: str):
    """This function implements the momentum equation and returns the  accleration on the fluid particles"""

    dx = h = 2 * fluid.elements.bounding_radius()
    epsilon = 0.01  # parameter to avoid zero denominator

    d_fluid = rename_dims(d_fluid, 'particles', 'others')
    d_wall = rename_dims(zeros(instance(wall)), 'particles', 'others')
    m_fluid = rename_dims(m_fluid, 'particles', 'others')
    m_wall = rename_dims(zeros(instance(wall)), 'particles', 'others')
    v_fluid = rename_dims(fluid.values, 'particles', 'others')
    v_wall = rename_dims(wall.values, 'particles', 'others')
    p_fluid = rename_dims(p_fluid, 'particles', 'others')
    p_wall = rename_dims(p_wall, 'particles', 'others')

    vel = concat([v_fluid, v_wall], 'others')
    density = concat([d_fluid, d_wall], 'others')
    mass = concat([m_fluid, m_wall], 'others')
    pressure = concat([p_fluid, p_wall], 'others')

    #alpha_ab = where(distances == 0, 0, fluid_alpha)  # Removing the particle itself from further calculation
    alpha_ab = where(distances == 0, 0, fluid_alpha)  # N X N matrix N-->all particles
    fluid_particle_neighbour_alpha_ab = alpha_ab.particles[:fluid.particles.size].others[:]

    #c_ab = where(distances == 0, 0, fluid_c_0)  # Removing the particle itself from further calculation
    c_ab = where(distances == 0, 0, fluid_c_0)  # N X N matrix N-->all particles
    fluid_particle_neighbour_c_ab = c_ab.particles[:fluid.particles.size].others[:]

    # Below we create matrices which have non-zero entries where the neighbour is inside cut-off radius 'r_c' rest all entries are 0
    vel = rename_dims(vel, 'others', 'particles')
    relative_v = vel - rename_dims(vel, 'particles', 'others')

    #neighbour_density = where(distances == 0, 0, density)  # Removing the particle itself from further calculation
    neighbour_density = where(distances == 0, 0, density)  # 0 for places which are not neighbours for particle under consideration. Stacks the particle density vertically. i.e. dupicates the densities

    #neighbour_mass = where(distances == 0, 0, mass)  # Removing the particle itself from further calculation
    neighbour_mass = where(distances == 0 , 0, mass)

    #neighbour_pressure = where(distances == 0, 0, pressure)  # Removing the particle itself from further calculation
    neighbour_pressure = where(distances == 0, 0, pressure)

    dvx = relative_v['x'].particles[:fluid.particles.size]  # separating the x and y components
    dvy = relative_v['y'].particles[:fluid.particles.size]

    fluid_all_neighbours_dist = distances.particles[:fluid.particles.size]
    dvx = where(fluid_all_neighbours_dist == 0, 0, dvx)  # relative x-velocity between a fluid particle and its neighbour
    dvy = where(fluid_all_neighbours_dist == 0, 0, dvy)

    # Slicing the distance matrix of fluid neighbour of fluid particles
    rad = distances.particles[:fluid.particles.size].others[:]

    #####CHANGE KERNEL DERIVATIVE HERE
    der_w = kernel_derivative(rad / h, fluid.spatial_rank, kernel) / h ** (fluid.spatial_rank + 1)

    drx = distances_vec['x'].particles[:fluid.particles.size].others[:]
    dry = distances_vec['y'].particles[:fluid.particles.size].others[:]

    # rho_b, m_b and p_b matrices rows are for each fluid particle and cols are fluid and wall particles
    fluid_neighbour_pressure = neighbour_pressure.particles[:fluid.particles.size]

    fluid_fluid_neighbour_density = neighbour_density.particles[:fluid.particles.size].others[:fluid.particles.size]
    fluid_fluid_neighbour_mass = neighbour_mass.particles[:fluid.particles.size].others[:fluid.particles.size]
    fluid_wall_neighbour_mass = neighbour_mass.particles[:fluid.particles.size].others[fluid.particles.size:]

    fluid_wall_neighbour_density_term = d0_fluid * ((p_fluid - fluid_xi) / fluid_p_0 + 1) ** (1 / fluid_adiabatic_exp)

    fluid_wall_neighbour_density_term = rename_dims(fluid_wall_neighbour_density_term, 'others', 'particles')
    helper_matrix = distances.particles[:fluid.particles.size].others[fluid.particles.size:]
    helper_matrix = where(helper_matrix != 0, 1, helper_matrix)  # technically called adjacency matrix

    # ALL boundary particle neighbours for a given fluid particle will have same density
    fluid_wall_neighbour_density = helper_matrix * fluid_wall_neighbour_density_term

    # fluid_particle_wall_neighbour_density=where(fluid_particle_wall_neighbour_density!=0,fluid_particle_wall_neighbour_density_term,fluid_particle_wall_neighbour_density)
    fluid_wall_neighbour_mass = where(helper_matrix != 0, (d0_fluid * dx * dx), fluid_wall_neighbour_mass)

    # joining the density and mass of the fluid and wall particle back together
    fluid_particle_neighbour_density = concat([fluid_fluid_neighbour_density, fluid_wall_neighbour_density], 'others')
    fluid_particle_neighbour_mass = concat([fluid_fluid_neighbour_mass, fluid_wall_neighbour_mass], 'others')

    # Momentum Equation for the pressure gradient part
    d_fluid = rename_dims(d_fluid, 'others', 'particles')
    p_fluid = rename_dims(p_fluid, 'others', 'particles')
    m_fluid = rename_dims(m_fluid, 'others', 'particles')

    # (rho_a + rho_b)
    particle_density_sum = where(fluid_particle_neighbour_density != 0, d_fluid + fluid_particle_neighbour_density, fluid_particle_neighbour_density)

    # rho_ab = 0.5 * (rho_a + rho_b)
    rho_ab = 0.5 * particle_density_sum

    # Setting minimum density as the initial density
    d_fluid = where(d_fluid == 0, d0_fluid, d_fluid)

    # p_ab = ((rho_b * p_a) + (rho_a * p_b)) / (rho_a + rho_b)
    term_1 = fluid_particle_neighbour_density * p_fluid
    term_2 = fluid_neighbour_pressure * d_fluid
    p_ab = divide_no_nan((term_1 + term_2), particle_density_sum)

    # pressure_fact = - (1/m_a) * ((m_a/rho_a)**2 + (m_b/rho_b)**2) * (p_ab) * der_W #equation 5
    fluid_neighbour_mass_by_density_ratio = divide_no_nan(fluid_particle_neighbour_mass, fluid_particle_neighbour_density) ** 2  # (m_b/rho_b)²
    fluid_mass_by_density_ratio = divide_no_nan(m_fluid, d_fluid) ** 2  # (m_a/rho_a)²

    # sum = (m_a/rho_a)**2 + (m_b/rho_b)**2
    mass_by_density_sum = where(fluid_neighbour_mass_by_density_ratio != 0, fluid_mass_by_density_ratio + fluid_neighbour_mass_by_density_ratio, fluid_neighbour_mass_by_density_ratio)

    # pressure_fact=-(1/m_a) * ((m_a/rho_a)**2 + (m_b/rho_b)**2) * (p_ab) * der_W (equation 5)
    pressure_fact = (-1 / m_fluid) * mass_by_density_sum * p_ab * der_w

    # a_x
    a_x = math.sum(pressure_fact * divide_no_nan(drx, rad), 'others')
    # a_y
    a_y = math.sum(pressure_fact * divide_no_nan(dry, rad), 'others')

    # --- ARTIFICIAL VISCOSITY ---
    visc_art_fact = zeros(instance(fluid_all_neighbours_dist))  # zeros matrix with size: (fluid_particles X all_particles)
    temp_matrix = (drx * dvx) + (dry * dvy)

    # visc_art_fact_term = m_b * alpha_ab * h * c_ab * (((dvx * drx) + (dvy * dry))/(rho_ab * ((rad**2) + epsilon * (h**2)))) * der_W
    visc_art_fact_term = fluid_particle_neighbour_mass * fluid_particle_neighbour_alpha_ab * h * fluid_particle_neighbour_c_ab * divide_no_nan(temp_matrix, (rho_ab * ((rad ** 2) + epsilon * (h ** 2)))) * der_w
    visc_art_fact = where(temp_matrix < 0, visc_art_fact_term, visc_art_fact)

    # a_x
    a_x = a_x + math.sum(visc_art_fact * divide_no_nan(drx, rad), 'others')
    # a_y
    a_y = a_y + math.sum(visc_art_fact * divide_no_nan(dry, rad), 'others')

    #Gravity addition
    a_y = a_y + gravity['y']
    fluid_acc = stack([a_x, a_y], channel(vector='x,y'))
    return fluid_acc


def time_step_size(fluid_c_0, fluid: PointCloud, wall: PointCloud, fluid_alpha, gravity: Tensor):
    """Implements adaptive time step ensuring CFL condition"""
    v_max_magnitude = math.maximum(math.max(vec_length(fluid.values)), math.max(vec_length(wall.values)))
    c_max = fluid_c_0  # wall_c_0 =0 so no point in taking the max out of them
    h = 2 * fluid.elements.bounding_radius()
    dt_1 = 0.25 * h / (c_max + v_max_magnitude)  # single value
    mu = 0.5 / (fluid.spatial_rank + 2) * fluid_alpha * h * c_max  # viscous condition
    dt_2 = 0.125 * (h ** 2) / mu
    dt_3 = 0.25 * math.sqrt(h / vec_length(gravity))
    return math.min(wrap([dt_1, dt_2, dt_3], instance('time_steps')))


def kernel_function(q: Tensor, spatial_rank: int, kernel='wendland-c2'):
    """
    Compute the SPH kernel value at a normalized scalar distance `q`.

    Args:
        q: Normalized distance `phi.math.Tensor`.
        spatial_rank: Dimensionality of the simulation.
        kernel: Which kernel to use, one of `'wendland-c2'`, `'quintic-spline'`.

    Returns:
        `phi.math.Tensor`
    """
    if kernel == 'quintic-spline':  # cutoff at q = 3 (d=3h)
        alpha_d = {1: 1/120, 2: 7/478/PI, 3: 1/120/PI}[spatial_rank]
        w1 = (3-q)**5 - 6 * (2-q)**5 + 15 * (1-q)**5
        w2 = (3-q)**5 - 6 * (2-q)**5
        w3 = (3-q)**5
        return alpha_d * where(q > 3, 0, where(q > 2, w3, where(q > 1, w2, w1)))
    elif kernel == 'wendland-c2':  # cutoff at q=2 (d=2h)
        alpha_d = {2: 7/4/PI, 3: 21/16/PI}[spatial_rank]
        w = (1 - 0.5*q)**4 * (2*q + 1)
        return alpha_d * where(q <= 2, w, 0)
    else:
        raise ValueError(kernel)


def kernel_derivative(q: Tensor, spatial_rank: int, kernel='wendland-c2') -> Tensor:
    """
    Compute the kernel derivative *dw/dq* of a scale-independent kernel, evaluated at normalized distance `q`.

    Args:
        q: Normalized distance: physical distance divided by particle diameter.
        spatial_rank: Number of spatial dimensions of the simulation, `int`.
        kernel: Which kernel to use, one of `'wendland-c2'`, `'quintic-spline'`.

    Returns:
        `phi.math.Tensor`
    """
    if kernel == 'quintic-spline':
        alpha_d = {1: 1/120, 2: 7/478/PI, 3: 1/120/PI}[spatial_rank]
        dw_dq_1 = (3-q)**4 - 6 * (2-q)**4 + 15 * (1-q)**4
        dw_dq_2 = (3-q)**4 - 6 * (2-q)**4
        dw_dq_3 = (3-q)**4
        return -5 * alpha_d * where(q > 3, 0, where(q > 2, dw_dq_3, where(q > 1, dw_dq_2, dw_dq_1)))
    elif kernel == 'wendland-c2':
        alpha_d = {2: 7/4/PI, 3: 21/16/PI}[spatial_rank]
        dw_dq = ((1 - 0.5 * q) ** 3) * q
        return -5 * alpha_d * where(q <= 2, dw_dq, 0)
