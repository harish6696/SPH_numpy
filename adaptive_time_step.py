#from phi.flow import *
from phi.jax.flow import *
#math.set_global_precision(32)

def time_step_size(fluid_c_0,fluid_particles, wall_particles,h,fluid_alpha,d,g):

    fluid_particle_velocity=math.expand(fluid_particles.values, instance(fluid_particles)) # adding particle dimension to fluid_particles.values
    wall_particle_velocity = math.expand(wall_particles.values, instance(wall_particles))
    
    particle_velocity= math.concat([fluid_particle_velocity,wall_particle_velocity], 'particles')
 
    vmax_magnitude= math.max(math.vec_length(particle_velocity))

    c_max= fluid_c_0  # wall_c_0 =0 so no point in taking the max out of them

    dt_1=0.25*h/(c_max+vmax_magnitude) # single value

    #viscous condition
    mu=0.5/(d+2) *fluid_alpha*h*c_max
    dt_2=0.125*(h**2)/mu

    dt_3= 0.25*math.sqrt(h/abs(g))

    dt= tensor([dt_1,dt_2,dt_3], instance('time_steps') )

    dt = math.min(dt)

    return dt








