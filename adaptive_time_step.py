from phi.flow import *
import numpy as np

def time_step_size(c_0,particles,h,alpha,d,g):

    dt=np.zeros(3)   # take the minimum out of these 3
    particles=np.array(particles)
    particle_velocity_magnitude=np.zeros(len(particles))
    particle_velocity_magnitude=math.sqrt((particles[:,6])**2 +(particles[:,7])**2)
    vmax_magnitude= np.max(particle_velocity_magnitude) # find the  maximum velocity

    c_max= np.max(c_0)

    dt[0]=0.25*h/(c_max+vmax_magnitude) # single value

    #viscous condition
    mu=0.5/(d+2) * np.max(alpha)*h*c_max
    dt[1]=0.125*(h**2)/mu

    dt[2]= 0.25*math.sqrt(h/abs(g))

    dt = np.min(dt)

    return dt








