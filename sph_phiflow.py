from phi import math
from phi.math import Tensor, concat, rename_dims, instance, channel, stack, wrap
from phi.field import PointCloud


def leapfrog_sph(fluid: PointCloud,
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
                 fluid_Xi,
                 fluid_alpha,
                 r_c,
                 h,
                 gravity: Tensor,
                 dx):
    """

    Args:
        fluid:
        wall:
        prev_fluid_acc:
        p_fluid:
        p_wall: Fluid pressure
        d0_fluid: Fluid initial density
        d_fluid: Fluid density
        mass: Fluid mass
        fluid_adiabatic_exp:
        fluid_c_0:
        fluid_p_0:
        fluid_Xi:
        fluid_alpha:
        r_c:
        h:
        gravity:
        dx:

    Returns:

    """
    fluid = fluid.with_values(math.expand(fluid.values, instance(fluid)))
    wall = wall.with_values(math.expand(wall.values, instance(wall)))
    dt = time_step_size(fluid_c_0, fluid, wall, h, fluid_alpha, gravity)
    if prev_fluid_acc is None:
        prev_fluid_acc = calculate_acceleration(fluid, wall, d_fluid, p_fluid, p_wall, mass, d0_fluid, fluid_Xi, fluid_adiabatic_exp, fluid_p_0, h, r_c, fluid_alpha, fluid_c_0, gravity)
    # --- Integrate velocity & position ---
    v_fluid = fluid.values + dt / 2 * prev_fluid_acc
    x_fluid = fluid.elements.shifted(dt / 2 * v_fluid)
    fluid = fluid.with_elements(x_fluid).with_values(v_fluid)
    # --- Pressure ---
    p_wall, wall = boundary_update(fluid, wall, p_fluid, d_fluid, r_c, h, gravity)
    drho_by_dt = calculate_density_derivative(fluid, wall, d_fluid, p_fluid, mass, d0_fluid, fluid_Xi, fluid_adiabatic_exp, fluid_p_0, h, r_c, dx)
    d_fluid = d_fluid + (dt * drho_by_dt)
    p_fluid = fluid_p_0 * ((d_fluid / d0_fluid) ** fluid_adiabatic_exp - 1) + fluid_Xi
    # --- Integrate position ---
    fluid = fluid.with_elements(fluid.elements.shifted(.5 * dt * v_fluid))
    # --- Compute new acceleration and integrate velocity ---
    p_wall, wall = boundary_update(fluid, wall, p_fluid, d_fluid, r_c, h, gravity)  # ToDo This would require a new distance matrix since fluid positions were altered
    fluid_acc = calculate_acceleration(fluid, wall, d_fluid, p_fluid, p_wall, mass, d0_fluid, fluid_Xi, fluid_adiabatic_exp, fluid_p_0, h, r_c, fluid_alpha, fluid_c_0, gravity)
    v_fluid = v_fluid + (dt / 2) * fluid_acc
    return fluid.with_values(v_fluid), wall, fluid_acc, p_fluid, d_fluid, p_wall


def boundary_update(fluid: PointCloud, wall, p_fluid, d_fluid, max_dist, h, gravity: Tensor):
    x = concat([fluid.points, wall.points], 'particles')
    distances_vec = -math.pairwise_distances(x, max_dist).particles[fluid.particles.size:].others[:fluid.particles.size]  # fluid neighbors of wall particles
    distances = math.vec_length(distances_vec)
    w = kernel_function(h, distances / h)
    sum_p_w = w.others * p_fluid.particles
    sum_density_r_w = (distances_vec['y'] * gravity['y'] * w).others * d_fluid.particles
    sum_w = math.sum(w, 'others')
    p_wall = math.where(sum_w != 0, math.divide_no_nan((sum_density_r_w + sum_p_w), sum_w), sum_w)
    sum_v_w = w.others * fluid.values.particles
    v_wall = math.where(sum_w != 0, math.divide_no_nan(-sum_v_w, sum_w), wall.values)
    wall = wall.with_values(v_wall)
    return p_wall, wall


def calculate_density_derivative(fluid, wall, d_fluid, p_fluid, m_fluid, d0_fluid, fluid_Xi, fluid_adiabatic_exp, fluid_p_0, h, r_c, dx):
    # wall_density=0, wall_mass=0 expected initially
    x = concat([fluid.points, wall.points], 'particles')
    v = concat([fluid.values, wall.values], 'particles')
    d_wall = rename_dims(math.zeros(instance(wall)), 'particles', 'others')
    wall_mass = rename_dims(math.zeros(instance(wall)), 'particles', 'others')
    d_fluid = rename_dims(d_fluid, 'particles', 'others')  # fluid_particle_density and mass have initially particle dimension (i.e column vector)
    m_fluid = rename_dims(m_fluid, 'particles', 'others')
    density = concat([d_fluid, d_wall], 'others')  # 1D Scalar array of densities of all particles (now it is a row vector)
    mass = concat([m_fluid, wall_mass], 'others')
    # Compute distance between all particles
    distances_vec = x - rename_dims(x, 'particles', 'others')  # contains both the x and y component of separation between particles
    distances = math.vec_length(distances_vec)  # contains magnitude of distance between ALL particles

    particle_neighbour_density = math.where(distances == 0, 0, density)
    particle_neighbour_density = math.where(distances > r_c, 0, particle_neighbour_density)  # 0 for places which are not neighbours for particle under consideration. Stacks the particle density vertically. i.e. dupicates the densities

    particle_neighbour_mass = math.where(distances == 0, 0, mass)
    particle_neighbour_mass = math.where(distances > r_c, 0, particle_neighbour_mass)

    relative_v = v - rename_dims(v, 'particles', 'others')  # 2d matrix of ALL particle velocity
    dvx = relative_v['x'].particles[:fluid.particles.size].others[:]  # separating the x and y components
    dvy = relative_v['y'].particles[:fluid.particles.size].others[:]

    fluid_particle_relative_dist = distances.particles[:fluid.particles.size].others[:]
    dvx = math.where(fluid_particle_relative_dist > r_c, 0, dvx)  # relative x-velocity between a fluid particle and its neighbour
    dvy = math.where(fluid_particle_relative_dist > r_c, 0, dvy)

    # Do all the cut off things before the final cut off for distance matrix
    distances_vec = math.where(distances > r_c, 0, distances_vec)
    distances = math.where(distances > r_c, 0, distances)  # Stores the distance between neighbours which are inside cutoff radius

    # Slicing the distance matrix of ALL neighbour of fluid particles
    q = distances.particles[:fluid.particles.size].others[:] / h

    DWab = kernel_derivative(h, q) / h

    drx = distances_vec['x'].particles[:fluid.particles.size].others[:]
    dry = distances_vec['y'].particles[:fluid.particles.size].others[:]

    mod_dist = (distances.particles[:fluid.particles.size].others[:])
    Fab_x = math.where(mod_dist != 0, math.divide_no_nan(drx, mod_dist) * DWab, drx)
    Fab_y = math.where(mod_dist != 0, math.divide_no_nan(dry, mod_dist) * DWab, dry)

    fluid_neighbour_density = particle_neighbour_density.particles[:fluid.particles.size].others[:]
    fluid_neighbour_mass = particle_neighbour_mass.particles[:fluid.particles.size].others[:]

    # Splitting the density and mass of fluid particle neighbours into two matrices....QQQQQQ CAN WE AVOID THIS? as we have to join it back anyways
    fluid_fluid_neighbour_density = fluid_neighbour_density.particles[:].others[:fluid.particles.size]
    # fluid_wall_neighbour_density = fluid_neighbour_density.particles[:].others[fluid.particles.size:]
    fluid_fluid_neighbour_mass = fluid_neighbour_mass.particles[:].others[:fluid.particles.size]
    fluid_wall_neighbour_mass = fluid_neighbour_mass.particles[:].others[fluid.particles.size:]

    ######'fluid_pressure' is for each fluid particle
    fluid_particle_wall_neighbour_density_term = d0_fluid * ((p_fluid - fluid_Xi) / fluid_p_0 + 1) ** (1 / fluid_adiabatic_exp)
    helper_matrix = distances.particles[:fluid.particles.size].others[fluid.particles.size:]
    helper_matrix = math.where(helper_matrix != 0, 1, helper_matrix)  # technically called adjacency matrix

    # ALL boundary particle neighbours for a given fluid particle will have same density
    fluid_wall_neighbour_density = helper_matrix * fluid_particle_wall_neighbour_density_term
    fluid_wall_neighbour_mass = math.where(helper_matrix != 0, (d0_fluid * dx * dx), fluid_wall_neighbour_mass)

    # joining the density and mass of the fluid and wall particle back together
    fluid_neighbour_density = concat([fluid_fluid_neighbour_density, fluid_wall_neighbour_density], 'others')
    fluid_neighbour_mass = concat([fluid_fluid_neighbour_mass, fluid_wall_neighbour_mass], 'others')

    neighbour_mass_by_density_ratio = math.divide_no_nan(fluid_neighbour_mass, fluid_neighbour_density)  # m_b/rho_b
    dot_product_result = (dvx * Fab_x) + (dvy * Fab_y)

    # Fluid_particle_density is a vector, each row of result is multiplied by the corresponding term of fluid_particle_density and then Sum along each row to get the result(drho_by_dt) for each fluid particle
    d_fluid = rename_dims(d_fluid, 'others', 'particles')
    d_rho_by_dt = math.sum((neighbour_mass_by_density_ratio * dot_product_result) * d_fluid, 'others')  # only for fluid particles
    return d_rho_by_dt


def calculate_acceleration(fluid: PointCloud, wall: PointCloud, d_fluid, p_fluid, p_wall, m_fluid, d0_fluid, fluid_Xi, fluid_adiabatic_exp, fluid_p_0, h, r_c, fluid_alpha, fluid_c_0, gravity):
    dx = h
    epsilon = 0.01  # parameter to avoid zero denominator

    x = concat([fluid.points, wall.points], 'particles')  # concatenating fluid coords and then boundary coords

    d_fluid = rename_dims(d_fluid, 'particles', 'others')
    d_wall = rename_dims(math.zeros(instance(wall)), 'particles', 'others')
    m_fluid = rename_dims(m_fluid, 'particles', 'others')
    m_wall = rename_dims(math.zeros(instance(wall)), 'particles', 'others')
    v_fluid = rename_dims(fluid.values, 'particles', 'others')
    v_wall = rename_dims(wall.values, 'particles', 'others')
    p_fluid = rename_dims(p_fluid, 'particles', 'others')
    p_wall = rename_dims(p_wall, 'particles', 'others')

    vel = concat([v_fluid, v_wall], 'others')
    density = concat([d_fluid, d_wall], 'others')
    mass = concat([m_fluid, m_wall], 'others')
    pressure = concat([p_fluid, p_wall], 'others')

    # Compute distance between all particles
    distances_vec = x - rename_dims(x, 'particles', 'others')  # contains both the x and y component of separation between particles
    distances = math.vec_length(distances_vec)  # contains magnitude of distance between ALL particles

    alpha_ab = math.where(distances == 0, 0, fluid_alpha)  # Removing the particle itself from further calculation
    alpha_ab = math.where(distances > r_c, 0, alpha_ab)  # N X N matrix N-->all particles
    fluid_particle_neighbour_alpha_ab = alpha_ab.particles[:fluid.particles.size].others[:]

    c_ab = math.where(distances == 0, 0, fluid_c_0)  # Removing the particle itself from further calculation
    c_ab = math.where(distances > r_c, 0, c_ab)  # N X N matrix N-->all particles
    fluid_particle_neighbour_c_ab = c_ab.particles[:fluid.particles.size].others[:]

    # Below we create matrices which have non-zero entries where the neighbour is inside cut-off radius 'r_c' rest all entries are 0
    vel = rename_dims(vel, 'others', 'particles')
    relative_v = vel - rename_dims(vel, 'particles', 'others')

    neighbour_density = math.where(distances == 0, 0, density)  # Removing the particle itself from further calculation
    neighbour_density = math.where(distances > r_c, 0, neighbour_density)  # 0 for places which are not neighbours for particle under consideration. Stacks the particle density vertically. i.e. dupicates the densities

    neighbour_mass = math.where(distances == 0, 0, mass)  # Removing the particle itself from further calculation
    neighbour_mass = math.where(distances > r_c, 0, neighbour_mass)

    neighbour_pressure = math.where(distances == 0, 0, pressure)  # Removing the particle itself from further calculation
    neighbour_pressure = math.where(distances > r_c, 0, neighbour_pressure)

    dvx = relative_v['x'].particles[:fluid.particles.size]  # separating the x and y components
    dvy = relative_v['y'].particles[:fluid.particles.size]

    fluid_all_neighbours_dist = distances.particles[:fluid.particles.size]
    dvx = math.where(fluid_all_neighbours_dist > r_c, 0, dvx)  # relative x-velocity between a fluid particle and its neighbour
    dvy = math.where(fluid_all_neighbours_dist > r_c, 0, dvy)

    ######Do all the cut off things before the final cut off for distance matrix
    distances_vec = math.where(distances > r_c, 0, distances_vec)
    distances = math.where(distances > r_c, 0, distances)  # Stores the distance between neighbours which are inside cutoff radius

    # Slicing the distance matrix of fluid neighbour of fluid particles
    rad = distances.particles[:fluid.particles.size].others[:]

    q = rad / h

    der_W = kernel_derivative(h, q) / h

    drx = distances_vec['x'].particles[:fluid.particles.size].others[:]
    dry = distances_vec['y'].particles[:fluid.particles.size].others[:]

    # rho_b, m_b and p_b matrices rows are for each fluid particle and cols are fluid and wall particles
    ########THE BELOW 3 MATRICES CAN BE REMOVED
    fluid_particle_neighbour_density = neighbour_density.particles[:fluid.particles.size].others[:]
    fluid_particle_neighbour_mass = neighbour_mass.particles[:fluid.particles.size].others[:]
    fluid_particle_neighbour_pressure = neighbour_pressure.particles[:fluid.particles.size].others[:]

    # Splitting the density and mass of fluid particle neighbours into two matrices....QQQQQQ CAN WE AVOID THIS? as we have to join it back anyways
    fluid_fluid_neighbour_density = fluid_particle_neighbour_density.others[:fluid.particles.size]
    fluid_wall_neighbour_density = fluid_particle_neighbour_density.others[fluid.particles.size:]
    fluid_fluid_neighbour_mass = fluid_particle_neighbour_mass.others[:fluid.particles.size]
    fluid_wall_neighbour_mass = fluid_particle_neighbour_mass.others[fluid.particles.size:]

    ######'fluid_particle_pressure' is calculated and stored for each fluid particle
    fluid_wall_neighbour_density_term = d0_fluid * ((p_fluid - fluid_Xi) / fluid_p_0 + 1) ** (1 / fluid_adiabatic_exp)

    fluid_wall_neighbour_density_term = rename_dims(fluid_wall_neighbour_density_term, 'others', 'particles')
    helper_matrix = distances.particles[:fluid.particles.size].others[fluid.particles.size:]
    helper_matrix = math.where(helper_matrix != 0, 1, helper_matrix)  # technically called adjacency matrix

    # ALL boundary particle neighbours for a given fluid particle will have same density
    fluid_wall_neighbour_density = helper_matrix * fluid_wall_neighbour_density_term

    # fluid_particle_wall_neighbour_density=math.where(fluid_particle_wall_neighbour_density!=0,fluid_particle_wall_neighbour_density_term,fluid_particle_wall_neighbour_density)
    fluid_wall_neighbour_mass = math.where(helper_matrix != 0, (d0_fluid * dx * dx), fluid_wall_neighbour_mass)

    # joining the density and mass of the fluid and wall particle back together
    fluid_particle_neighbour_density = concat([fluid_fluid_neighbour_density, fluid_wall_neighbour_density], 'others')
    fluid_particle_neighbour_mass = concat([fluid_fluid_neighbour_mass, fluid_wall_neighbour_mass], 'others')

    # Momentum Equation for the pressure gradient part
    d_fluid = rename_dims(d_fluid, 'others', 'particles')
    p_fluid = rename_dims(p_fluid, 'others', 'particles')
    m_fluid = rename_dims(m_fluid, 'others', 'particles')

    # (rho_a + rho_b)
    particle_density_sum = math.where(fluid_particle_neighbour_density != 0, d_fluid + fluid_particle_neighbour_density, fluid_particle_neighbour_density)

    # rho_ab = 0.5 * (rho_a + rho_b)
    rho_ab = 0.5 * particle_density_sum

    # Setting minimum density as the initial density
    d_fluid = math.where(d_fluid == 0, d0_fluid, d_fluid)

    # p_ab = ((rho_b * p_a) + (rho_a * p_b)) / (rho_a + rho_b)
    term_1 = fluid_particle_neighbour_density * p_fluid
    term_2 = fluid_particle_neighbour_pressure * d_fluid
    p_ab = math.divide_no_nan(((term_1) + (term_2)), (particle_density_sum))

    # pressure_fact = - (1/m_a) * ((m_a/rho_a)**2 + (m_b/rho_b)**2) * (p_ab) * der_W #equation 5
    fluid_neighbour_mass_by_density_ratio = math.divide_no_nan(fluid_particle_neighbour_mass, fluid_particle_neighbour_density) ** 2  # (m_b/rho_b)²
    fluid_mass_by_density_ratio = math.divide_no_nan(m_fluid, d_fluid) ** 2  # (m_a/rho_a)²

    # sum = (m_a/rho_a)**2 + (m_b/rho_b)**2
    mass_by_density_sum = math.where(fluid_neighbour_mass_by_density_ratio != 0, fluid_mass_by_density_ratio + fluid_neighbour_mass_by_density_ratio, fluid_neighbour_mass_by_density_ratio)

    # pressure_fact=-(1/m_a) * ((m_a/rho_a)**2 + (m_b/rho_b)**2) * (p_ab) * der_W (equation 5)
    pressure_fact = (-1 / m_fluid) * mass_by_density_sum * p_ab * der_W

    # a_x
    a_x = math.sum(pressure_fact * math.divide_no_nan(drx, rad), 'others')
    # a_y
    a_y = math.sum(pressure_fact * math.divide_no_nan(dry, rad), 'others')

    ### ARTIFICIAL VISCOSITY
    visc_art_fact = math.zeros(instance(fluid_all_neighbours_dist))  # zeros matrix with size: (fluid_particles X all_particles)
    temp_matrix = (drx * dvx) + (dry * dvy)

    # visc_art_fact_term = m_b * alpha_ab * h * c_ab * (((dvx * drx) + (dvy * dry))/(rho_ab * ((rad**2) + epsilon * (h**2)))) * der_W
    visc_art_fact_term = fluid_particle_neighbour_mass * fluid_particle_neighbour_alpha_ab * h * fluid_particle_neighbour_c_ab * math.divide_no_nan(temp_matrix, (rho_ab * ((rad ** 2) + epsilon * (h ** 2)))) * der_W
    visc_art_fact = math.where(temp_matrix < 0, visc_art_fact_term, visc_art_fact)

    # a_x
    a_x = a_x + math.sum(visc_art_fact * math.divide_no_nan(drx, rad), 'others')
    # a_y
    a_y = a_y + math.sum(visc_art_fact * math.divide_no_nan(dry, rad), 'others')
    a_y = a_y + gravity['y']
    fluid_acc = stack([a_x, a_y], channel(vector='x,y'))
    return fluid_acc


def time_step_size(fluid_c_0, fluid: PointCloud, wall: PointCloud, h, fluid_alpha, gravity: Tensor):
    v_max_magnitude = math.maximum(math.max(math.vec_length(fluid.values)), math.max(math.vec_length(wall.values)))
    c_max = fluid_c_0  # wall_c_0 =0 so no point in taking the max out of them
    dt_1 = 0.25 * h / (c_max + v_max_magnitude)  # single value
    mu = 0.5 / (fluid.spatial_rank + 2) * fluid_alpha * h * c_max  # viscous condition
    dt_2 = 0.125 * (h ** 2) / mu
    dt_3 = 0.25 * math.sqrt(h / math.vec_length(gravity))
    return math.min(wrap([dt_1, dt_2, dt_3], instance('time_steps')))


def kernel_function(h, q):
    # Quintic Spline used as Weighting function
    # cutoff radius r_c = 3 * h;
    alpha_d = 0.004661441847880 / (h * h)
    # Weighting function
    w = math.where((q < 3) & (q >= 2), alpha_d * ((3 - q) ** 5), q)
    w = math.where((q < 2) & (q >= 1), alpha_d * (((3 - q) ** 5) - 6 * ((2 - q) ** 5)), w)
    w = math.where((q < 1) & (q > 0), alpha_d * (((3 - q) ** 5) - 6 * ((2 - q) ** 5) + 15 * ((1 - q) ** 5)), w)
    w = math.where((q >= 3), 0, w)
    return w


def kernel_derivative(h, q):
    alpha_d = -0.0233072092393989 / (h * h)
    der_w = math.where((q < 3) & (q >= 2), alpha_d * ((3 - q) ** 4), q)
    der_w = math.where((q < 2) & (q >= 1), alpha_d * (((3 - q) ** 4) - 6 * ((2 - q) ** 4)), der_w)
    der_w = math.where((q < 1) & (q > 0), alpha_d * (((3 - q) ** 4) - 6 * ((2 - q) ** 4) + 15 * ((1 - q) ** 4)), der_w)
    der_w = math.where((q >= 3), 0, der_w)
    return der_w
