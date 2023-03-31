from phi import math
from phi.field import PointCloud
from phi.math import Tensor, concat, instance, wrap, vec_length, where, divide_no_nan, zeros, PI


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
                    kernel: str = 'quintic-spline'):
    """
    Compute one SPH time step using velocity verlet Euler.

    The implementation is based on the paper [juSPH](https://www.sciencedirect.com/science/article/pii/S2352711022000954).

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
        fluid_c_0: Artificial speed of sound. Approximately 10*sqrt(2*g*H)
        fluid_p_0:Initial pressure of the fluid = density*g*H
        fluid_xi: Back pressure (For free surface flows = 0)
        fluid_alpha: Artificial viscosity constant
        max_dist: Cut off radius (3*dx)
        gravity: Gravity vector as `Tensor`
        kernel: Which kernel to use, one of `'wendland-c2'`, `'quintic-spline'`.

    Returns:
        fluid: Next fluid state
        wall: Obstacle particles with updated velocities
        fluid_acc:
        p_fluid: Fluid pressure.
        d_fluid: Fluid density.
        p_wall: Obstacle pressure.
    """
    dt = time_step_size(fluid_c_0, concat([fluid.values, wall.values], 'particles'), 2 * fluid.elements.bounding_radius(), fluid_alpha, vec_length(gravity))
    # --- Neighbour distance calculation ---
    x = concat([fluid.points, wall.points], 'particles')
    distances_vec = -math.pairwise_distances(x, max_dist)  # sending after performing cut-off to each function
    distances = vec_length(distances_vec)
    # --- Integrate velocity & position ---
    if prev_fluid_acc is None:
        prev_fluid_acc = calculate_acceleration(fluid, wall, d_fluid, p_fluid, p_wall, mass, d0_fluid, fluid_xi, fluid_adiabatic_exp, fluid_p_0, fluid_alpha, fluid_c_0, gravity, distances_vec, distances, kernel)
    fluid += (dt / 2) * prev_fluid_acc  # velocity update
    fluid = fluid.shifted((dt / 2) * fluid.values)  # position update
    # --- Density & Pressure ---
    p_wall, wall = boundary_update(fluid, wall, p_fluid, d_fluid, gravity, distances_vec, distances, kernel)  # ToDo p_wall is not used, we can replace the assignment by _
    d_fluid += dt * calculate_density_derivative(fluid, wall, d_fluid, p_fluid, mass, d0_fluid, fluid_xi, fluid_adiabatic_exp, fluid_p_0, distances_vec, distances, kernel)
    p_fluid = fluid_p_0 * ((d_fluid / d0_fluid) ** fluid_adiabatic_exp - 1) + fluid_xi
    # --- Integrate velocity & position ---
    fluid = fluid.shifted(dt / 2 * fluid.values)  # position update   ToDo This would require a new distance matrix since fluid positions were altered
    p_wall, wall = boundary_update(fluid, wall, p_fluid, d_fluid, gravity, distances_vec, distances, kernel)
    fluid_acc = calculate_acceleration(fluid, wall, d_fluid, p_fluid, p_wall, mass, d0_fluid, fluid_xi, fluid_adiabatic_exp, fluid_p_0, fluid_alpha, fluid_c_0, gravity, distances_vec, distances, kernel)
    fluid += (dt / 2) * fluid_acc  # velocity update
    return fluid, wall, fluid_acc, p_fluid, d_fluid, p_wall


def boundary_update(fluid: PointCloud, wall, p_fluid, d_fluid, gravity: Tensor, distances_vec, distances, kernel: str):
    """Function calculates the wall pressure and updates the wall velocity"""
    # only for wall particle and their fluid neighbour
    distances_vec = distances_vec.particles[fluid.particles.size:].particles.dual[:fluid.particles.size]
    distances = distances.particles[fluid.particles.size:].particles.dual[:fluid.particles.size]
    h = 2 * fluid.elements.bounding_radius()
    w = kernel_function(distances / h, fluid.spatial_rank, kernel) / h ** 2
    sum_p_w = w.particles.dual * p_fluid.particles
    sum_density_r_w = ((distances_vec.vector * gravity.vector) * w).particles.dual * d_fluid.particles
    sum_w = math.sum(w, '~particles')
    p_wall = where(sum_w != 0, divide_no_nan((sum_density_r_w + sum_p_w), sum_w), sum_w)
    sum_v_w = w.particles.dual * fluid.values.particles
    v_wall = where(sum_w != 0, divide_no_nan(-sum_v_w, sum_w), wall.values)
    wall = wall.with_values(v_wall)
    return p_wall, wall


def calculate_density_derivative(fluid, wall, d_fluid, p_fluid, m_fluid, d0_fluid, fluid_xi, fluid_adiabatic_exp, fluid_p_0, distances_vec, distances, kernel: str):
    # wall_density=0, wall_mass=0 expected initially
    """This function implements the continuity equation by calculating and returning the rate of change of density"""
    # x = concat([fluid.points, wall.points], 'particles')
    v = concat([fluid.values, wall.values], 'particles')
    d_wall = wall_mass = zeros(instance(wall)).particles.as_dual()
    d_fluid = d_fluid.particles.as_dual()  # fluid_particle_density and mass have initially particle dimension (i.e column vector)
    m_fluid = m_fluid.particles.as_dual()
    density = concat([d_fluid, d_wall], '~particles')  # 1D Scalar array of densities of all particles (now it is a row vector)
    mass = concat([m_fluid, wall_mass], '~particles')

    particle_neighbour_density = where(distances == 0, 0, density)  # 0 for places which are not neighbours for particle under consideration. Stacks the particle density vertically. i.e. dupicates the densities
    particle_neighbour_mass = where(distances == 0, 0, mass)

    relative_v = v - v.particles.as_dual()  # 2d matrix of ALL particle velocity
    dv = relative_v.particles[:fluid.particles.size]

    fluid_particle_relative_dist = distances.particles[:fluid.particles.size]
    dv = where(fluid_particle_relative_dist == 0, 0, dv)  # relative velocity between a fluid particle and its neighbour

    h = 2 * fluid.elements.bounding_radius()
    d_w_ab = kernel_derivative(distances.particles[:fluid.particles.size] / h, fluid.spatial_rank, kernel) / h ** (fluid.spatial_rank + 1)  # Consider all neighbours of fluid particles

    dr = distances_vec.particles[:fluid.particles.size]

    mod_dist = (distances.particles[:fluid.particles.size])
    fab = where(mod_dist != 0, divide_no_nan(dr, mod_dist) * d_w_ab, dr)

    fluid_fluid_neighbour_density = particle_neighbour_density.particles[:fluid.particles.size].particles.dual[:fluid.particles.size]
    fluid_fluid_neighbour_mass = particle_neighbour_mass.particles[:fluid.particles.size].particles.dual[:fluid.particles.size]
    fluid_wall_neighbour_mass = particle_neighbour_mass.particles[:fluid.particles.size].particles.dual[fluid.particles.size:]

    # fluid_pressure is for each fluid particle
    fluid_particle_wall_neighbour_density_term = d0_fluid * ((p_fluid - fluid_xi) / fluid_p_0 + 1) ** (1 / fluid_adiabatic_exp)
    helper_matrix = distances.particles[:fluid.particles.size].particles.dual[fluid.particles.size:]  # considering portion of distance-matrix corresponding to fluid_particles and wall neighbors
    helper_matrix = where(helper_matrix != 0, 1, helper_matrix)  # technically called adjacency matrix

    # ALL boundary particle neighbours for a given fluid particle will have same density
    fluid_wall_neighbour_density = helper_matrix * fluid_particle_wall_neighbour_density_term
    fluid_wall_neighbour_mass = where(helper_matrix != 0, (d0_fluid * fluid.elements.bounding_box().volume), fluid_wall_neighbour_mass)

    # joining the density and mass of the fluid and wall particle back together
    fluid_neighbour_density = concat([fluid_fluid_neighbour_density, fluid_wall_neighbour_density], '~particles')
    fluid_neighbour_mass = concat([fluid_fluid_neighbour_mass, fluid_wall_neighbour_mass], '~particles')

    neighbour_mass_by_density_ratio = divide_no_nan(fluid_neighbour_mass, fluid_neighbour_density)  # m_b/rho_b
    dot_product_result = dv.vector * fab.vector

    # Fluid_particle_density is a vector, each row of result is multiplied by the corresponding term of fluid_particle_density and then Sum along each row to get the result(drho_by_dt) for each fluid particle
    d_fluid = d_fluid.particles.dual.as_instance()
    d_rho_by_dt = math.sum((neighbour_mass_by_density_ratio * dot_product_result) * d_fluid, '~particles')  # only for fluid particles
    return d_rho_by_dt


def calculate_acceleration(fluid: PointCloud, wall: PointCloud, d_fluid, p_fluid, p_wall, m_fluid, d0_fluid, fluid_xi, fluid_adiabatic_exp, fluid_p_0, fluid_alpha, fluid_c_0, gravity, distances_vec, distances, kernel: str):
    """
    This function implements the momentum equation and returns the acceleration on the fluid particles.
    """
    dx = h = 2 * fluid.elements.bounding_radius()
    epsilon = 0.01  # parameter to avoid zero denominator
    m_wall = d_wall = zeros(instance(wall))
    vel = concat([fluid.values, wall.values], 'particles')
    density = concat([d_fluid, d_wall], 'particles').particles.as_dual()
    mass = concat([m_fluid, m_wall], 'particles').particles.as_dual()
    pressure = concat([p_fluid, p_wall], 'particles').particles.as_dual()

    # alpha_ab = where(distances == 0, 0, fluid_alpha)  # Removing the particle itself from further calculation
    alpha_ab = where(distances == 0, 0, fluid_alpha)  # N X N matrix N-->all particles  ToDo this just puts fluid_alpha to all all neighbors
    fluid_particle_neighbour_alpha_ab = alpha_ab.particles[:fluid.particles.size]

    # c_ab = where(distances == 0, 0, fluid_c_0)  # Removing the particle itself from further calculation
    c_ab = where(distances == 0, 0, fluid_c_0)  # N X N matrix N-->all particles
    fluid_particle_neighbour_c_ab = c_ab.particles[:fluid.particles.size]

    # Below we create matrices which have non-zero entries where the neighbour is inside cut-off radius 'r_c' rest all entries are 0
    relative_v = vel - vel.particles.as_dual()  # ToDo only for the sparsity pattern of distances

    # neighbour_density = where(distances == 0, 0, density)  # Removing the particle itself from further calculation
    neighbour_density = where(distances == 0, 0, density)  # 0 for places which are not neighbours for particle under consideration. Stacks the particle density vertically. i.e. dupicates the densities

    # neighbour_mass = where(distances == 0, 0, mass)  # Removing the particle itself from further calculation
    neighbour_mass = where(distances == 0, 0, mass)

    # neighbour_pressure = where(distances == 0, 0, pressure)  # Removing the particle itself from further calculation
    neighbour_pressure = where(distances == 0, 0, pressure)

    rel_v_fluid = relative_v.particles[:fluid.particles.size]
    fluid_all_neighbours_dist = distances.particles[:fluid.particles.size]
    dv = where(fluid_all_neighbours_dist == 0, 0, rel_v_fluid)

    # Slicing the distance matrix of fluid neighbour of fluid particles
    rad = distances.particles[:fluid.particles.size]

    #####CHANGE KERNEL DERIVATIVE HERE
    der_w = kernel_derivative(rad / h, fluid.spatial_rank, kernel) / h ** (fluid.spatial_rank + 1)

    dr = distances_vec.particles[:fluid.particles.size]

    # rho_b, m_b and p_b matrices rows are for each fluid particle and cols are fluid and wall particles
    fluid_neighbour_pressure = neighbour_pressure.particles[:fluid.particles.size]

    fluid_fluid_neighbour_density = neighbour_density.particles[:fluid.particles.size].particles.dual[:fluid.particles.size]
    fluid_fluid_neighbour_mass = neighbour_mass.particles[:fluid.particles.size].particles.dual[:fluid.particles.size]
    fluid_wall_neighbour_mass = neighbour_mass.particles[:fluid.particles.size].particles.dual[fluid.particles.size:]

    fluid_wall_neighbour_density_term = d0_fluid * ((p_fluid - fluid_xi) / fluid_p_0 + 1) ** (1 / fluid_adiabatic_exp)

    helper_matrix = distances.particles[:fluid.particles.size].particles.dual[fluid.particles.size:]
    helper_matrix = where(helper_matrix != 0, 1, helper_matrix)  # technically called adjacency matrix  # ToDo this would be dense

    # ALL boundary particle neighbours for a given fluid particle will have same density
    fluid_wall_neighbour_density = helper_matrix * fluid_wall_neighbour_density_term

    # fluid_particle_wall_neighbour_density=where(fluid_particle_wall_neighbour_density!=0,fluid_particle_wall_neighbour_density_term,fluid_particle_wall_neighbour_density)
    fluid_wall_neighbour_mass = where(helper_matrix != 0, (d0_fluid * dx * dx), fluid_wall_neighbour_mass)

    # joining the density and mass of the fluid and wall particle back together
    fluid_particle_neighbour_density = concat([fluid_fluid_neighbour_density, fluid_wall_neighbour_density], '~particles')
    fluid_particle_neighbour_mass = concat([fluid_fluid_neighbour_mass, fluid_wall_neighbour_mass], '~particles')

    # --- Momentum Equation for the pressure gradient part ---
    # (rho_a + rho_b)
    particle_density_sum = where(fluid_particle_neighbour_density != 0, d_fluid + fluid_particle_neighbour_density, fluid_particle_neighbour_density)
    rho_ab = 0.5 * particle_density_sum
    d_fluid = where(d_fluid == 0, d0_fluid, d_fluid)  # Setting minimum density as the initial density

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

    a = math.sum(pressure_fact * divide_no_nan(dr, rad), '~particles')

    # --- ARTIFICIAL VISCOSITY ---
    visc_art_fact = zeros(instance(fluid_all_neighbours_dist))  # zeros matrix with size: (fluid_particles X all_particles)
    temp_matrix = dr.vector * dv.vector
    # visc_art_fact_term = m_b * alpha_ab * h * c_ab * (((dvx * drx) + (dvy * dry))/(rho_ab * ((rad**2) + epsilon * (h**2)))) * der_W
    visc_art_fact_term = fluid_particle_neighbour_mass * fluid_particle_neighbour_alpha_ab * h * fluid_particle_neighbour_c_ab * divide_no_nan(temp_matrix, (rho_ab * ((rad ** 2) + epsilon * (h ** 2)))) * der_w
    visc_art_fact = where(temp_matrix < 0, visc_art_fact_term, visc_art_fact)

    a += math.sum(visc_art_fact * divide_no_nan(dr, rad), '~particles')
    return a + gravity


def time_step_size(fluid_c_0, velocity: Tensor, particle_size, fluid_alpha, max_force: Tensor):
    """
    Determine maximum SPH step time from the CFL condition.

    Args:
        fluid_c_0: Artificial speed of sound.
        velocity: Fluid particle / obstacle velocities occurring in the simulation.
            The maximum velocity determines the time step.
        particle_size: Size / diameter of fluid particles.
        fluid_alpha: Artificial viscosity.
        max_force: Scalar `Tensor`. Maximum external force that is exerted on any particle, such as gravity.

    Returns:
        `dt`: Time increment.
    """
    v_max_magnitude = math.max(vec_length(velocity))
    c_max = fluid_c_0  # wall_c_0 =0 so no point in taking the max out of them
    h = particle_size
    dt_1 = 0.25 * h / (c_max + v_max_magnitude)  # single value
    mu = 0.5 / (velocity.vector.size + 2) * fluid_alpha * h * c_max  # viscous condition
    dt_2 = 0.125 * (h ** 2) / mu
    dt_3 = 0.25 * math.sqrt(h / max_force)
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
