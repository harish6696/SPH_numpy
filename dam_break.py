from phi.flow import *
from sph_phiflow import step, time_step_size

d = 2  # dimension of the problem (indirectly dimension of the kernal function)
dx = 0.006  # distance between particles
h = dx  # cut off radius
r_c = 3 * h  # Quintic Spline
alpha = 0.02  # viscosity coefficient value of water

width = dx * np.ceil(1.61 / dx)  # width=1.614 for dx= 0.006
g = -9.81
###CHANGE LATER TO 0.3
height = 0.3
v_max = np.sqrt(2 * abs(g) * height)

###### Properties of Fluid Particles #######
fluid_initial_density = 1000.0
fluid_adiabatic_exp = 7.0  # adiabatic coefficient (pressure coefficient)
fluid_c_0 = 10.0 * v_max  # artificial speed of sound c_0 and v_max = 2*abs(g)*height
fluid_p_0 = (fluid_initial_density * ((fluid_c_0) ** 2)) / fluid_adiabatic_exp  # reference pressure
fluid_Xi = 0.0  # background pressure
fluid_mu = 0.01  # viscosity
fluid_alpha = alpha  # artificial visc factor

#####UNCOMMENT LATER
fluid_coords = pack_dims(math.meshgrid(x=100, y=50), 'x,y', instance('particles')) * (0.6 / 100, 0.3 / 50) + (0.003, 0.003)  # 5000 fluid particle coordinates created


fluid_particles = PointCloud(Sphere(fluid_coords, radius=0.01))  # """is this radius only for visualization?????????????"""

fluid_velocity = fluid_particles * (0.0, 0.0)  # can we remove this unnecessary point cloud creation ?
fluid_particles = fluid_particles.with_values(fluid_velocity.values)  # fluid particles is a point cloud with elements as points of fluid coordinates and values as velocity

single_fluid_particle_mass = fluid_initial_density * dx ** d
fluid_particle_mass = math.ones(instance(fluid_coords)) * single_fluid_particle_mass

fluid_pressure = math.zeros(instance(fluid_coords))

###### Properties of Wall particles #####
wall_initial_density = 0.0
wall_adiabatic_exp = 0.0  # adiabatic coefficient (pressure coefficient)
wall_c_0 = 0.0  # artificial speed of sound c_0
wall_p_0 = 0.0  # reference pressure
wall_Xi = 0.0  # background pressure
wall_mu = 0.0  # viscosity
wall_alpha = 0 - 0  # artificial visc factor

left_wall_coords = pack_dims(math.meshgrid(x=3, y=134), 'x,y', instance('particles')) * ((0.018 / 3), (0.804 / 134)) + (-0.015, 0.003)
# #print(f"{left_wall_coords:full:shape}")
right_wall_coords = pack_dims(math.meshgrid(x=3, y=134), 'x,y', instance('particles')) * ((0.018 / 3), (0.804 / 134)) + (1.617, 0.003)
center_wall_coords = (pack_dims(math.meshgrid(x=275, y=3), 'x,y', instance('particles')) * ((1.65 / 275), (0.018 / 3)) + (-0.015, -0.015))

# concatenating the wall coordinates
wall_coords = math.concat([left_wall_coords, right_wall_coords, center_wall_coords], 'particles')  # 1629 wall particles

wall_particles = PointCloud(Sphere(wall_coords, radius=0.01), color='#FFA500')

wall_initial_velocity = wall_particles * (0, 0)
wall_particles = wall_particles.with_values(wall_initial_velocity.values)
wall_pressure = math.zeros(instance(wall_coords))
wall_density = math.zeros(instance(wall_coords))

fluid_pressure = fluid_initial_density * abs(g) * (height - fluid_particles.points['y'])

fluid_density = fluid_initial_density * (((fluid_pressure - fluid_Xi) / fluid_p_0) + 1) ** (1.0 / fluid_adiabatic_exp)




t = 0
n_dt = 0

################################################
### To be corrected
################################################
H = math.max(fluid_particles.points['y']) - math.min(fluid_particles.points['y']) + dx
math.print(H.all)
print(height)
print('Actual height of water column is: ' + str(H))
if abs(H.all - height) >= 1.0e-6:
    print("wrong height specified")
    print(H.any - height)
    # exit()

# reference_value
v_ref = math.sqrt(2 * abs(g) * H)
t_ref = H / v_ref

time_nondim = 5  # nondimensional running time (t/t_ref)
t_end = numpy.asarray(time_nondim * t_ref)

print("Total simulation time is: " + str(t_end))
print("Total number of fluid particles: " + str(fluid_particles.elements.center.particles.size))
print("Total number of wall particles: " + str(wall_particles.elements.center.particles.size))

fluid_traj = []

# wall_particle_pressure, wall_particles = boundary_update(fluid_particles, wall_particles, fluid_particle_pressure, fluid_particle_density, d, r_c, h, g)

print("Simulation progress: " + str((100 * (t / t_end))) + " Time step " + str(n_dt))
dt = time_step_size(fluid_c_0, fluid_particles, wall_particles, h, fluid_alpha, d, g)
print("Time step size is: " + str(dt))

for n_dt in range(11600):
    print(n_dt)
    fluid_particles, wall_particles, fluid_particle_acceleration, fluid_particle_pressure, fluid_particle_density, wall_particle_pressure = \
        step(fluid_particles, wall_particles, None, fluid_pressure, wall_pressure, fluid_initial_density, fluid_density, fluid_particle_mass, fluid_adiabatic_exp, fluid_c_0, fluid_p_0, fluid_Xi, fluid_alpha, dt, n_dt, d, r_c, h, g, dx)
