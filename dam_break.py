from phi.jax.flow import *
from sph_phiflow import velocity_verlet


velocity_verlet = jit_compile(velocity_verlet)

# --- Setup ---
gravity = vec(x=0, y=-9.81)
dx = 0.006  # distance between particles
width = dx * np.ceil(1.61 / dx)  # width=1.614 for dx= 0.006
height = 0.3 ##TO BE MODIFIED ACCORDING TO THE SIMULATION
v_max = math.sqrt(2 * math.vec_length(gravity) * height)

# --- Properties of Fluid Particles ---
fluid_initial_density = 1000.0
fluid_adiabatic_exp = 7.0  # adiabatic coefficient (pressure coefficient)
fluid_c_0 = 10.0 * v_max  # artificial speed of sound c_0 and v_max = 2*abs(g)*height
fluid_p_0 = (fluid_initial_density * (fluid_c_0 ** 2)) / fluid_adiabatic_exp  # reference pressure
fluid_Xi = 0.0  # background pressure
fluid_alpha = 0.02  #  artificial viscosity factor (changed this from 0.02 to 0.1)

#x_fluid = pack_dims(math.meshgrid(x=100, y=50), 'x,y', instance('particles')) * (0.6 / 100, 0.3 / 50) + (0.003, 0.003)
x_fluid = pack_dims(math.meshgrid(x=25, y=25), 'x,y', instance('particles')) * (0.15/25, 0.15/25) + (0.825,0.005)
#x_fluid = pack_dims(math.meshgrid(x=3, y=2), 'x,y', instance('particles')) * (0.018/3.0, 0.012/2.0) + (0.825,0.10)  # 6 fluid particle coordinates created 

fluid = PointCloud(Sphere(x_fluid, radius=dx / 2)) * (0.0, 0.0)
single_fluid_particle_mass = fluid_initial_density * (dx ** fluid.spatial_rank)
fluid_particle_mass = math.ones(instance(x_fluid)) * single_fluid_particle_mass
# fluid_pressure = math.zeros(instance(x_fluid))
fluid_pressure = fluid_initial_density * math.vec_length(gravity) * (height - fluid.points['y'])
fluid_density = fluid_initial_density * (((fluid_pressure - fluid_Xi) / fluid_p_0) + 1) ** (1.0 / fluid_adiabatic_exp)
H = math.max(fluid.points['y']) - math.min(fluid.points['y']) + dx
print(f'height={height}, actual height of water column is: ' + str(H))
assert not abs(H - height >= 1.0e-6).all, f"wrong height specified: {H - height}"

# --- Properties of Wall particles ---
# wall_initial_density = 0.0
# wall_adiabatic_exp = 0.0  # adiabatic coefficient (pressure coefficient)
# wall_c_0 = 0.0  # artificial speed of sound c_0
# wall_p_0 = 0.0  # reference pressure
# wall_Xi = 0.0  # background pressure
# wall_mu = 0.0  # viscosity
# wall_alpha = 0.0  # artificial viscosity factor

left_wall_coords = pack_dims(math.meshgrid(x=3, y=134), 'x,y', instance('particles')) * ((0.018 / 3), (0.804 / 134)) + (-0.015, 0.003)
right_wall_coords = pack_dims(math.meshgrid(x=3, y=134), 'x,y', instance('particles')) * ((0.018 / 3), (0.804 / 134)) + (1.617, 0.003)
center_wall_coords = (pack_dims(math.meshgrid(x=275, y=3), 'x,y', instance('particles')) * ((1.65 / 275), (0.018 / 3)) + (-0.015, -0.015))
x_wall = concat([left_wall_coords, right_wall_coords, center_wall_coords], 'particles')  # 1629 wall particles

# center_wall_coords = (pack_dims(math.meshgrid(x=3, y=2), 'x,y', instance('particles')) * ( (0.6/100), (0.012/2) ) + (0.815,-0.015))
# x_wall = center_wall_coords

wall = PointCloud(Sphere(x_wall, radius=dx / 2)) * (0, 0)
wall_pressure = math.zeros(instance(x_wall))
wall_density = math.zeros(instance(x_wall))

# v_ref = math.sqrt(2 * math.vec_length(gravity) * H)
# t_ref = H / v_ref

# --- Run the simulation ---
fluid_trj = [fluid]
for i in range(25000):  # 11600
    print('timestep '+str(i))
    fluid, wall, acceleration, fluid_pressure, fluid_density, wall_pressure = velocity_verlet(
        fluid, wall, None, fluid_pressure, wall_pressure, fluid_initial_density, fluid_density, fluid_particle_mass, fluid_adiabatic_exp, fluid_c_0, fluid_p_0, fluid_Xi, fluid_alpha, gravity)
    if i % 100 == 0:
        fluid_trj.append(fluid)

# --- video ---
fluid_trj = stack(fluid_trj, batch('time'), expand_values=True)
print("Creating video...")
vis.plot(vis.overlay(wall.elements, fluid_trj.elements), animate='time')
vis.savefig('vid_dam_break.mp4')
print("Video saved")
