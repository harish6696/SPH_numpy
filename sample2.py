from phi.flow import *

#fluid_velocity= tensor([(1, 0), (10, 2), (0, 12)], instance('particles') & channel(vector='x,y'))

a_x = tensor([1.5,1,1], instance('particles') )
a_y = tensor([2,3,2], instance('particles') )

a = stack([a_x, a_y], channel(vector='x,y'))

from phi.flow import *

fluid_coords = pack_dims(math.meshgrid(x=1, y=3), 'x,y', instance('particles')) * (0.2/1, 0.12/3) + (0.003,0.003)
wall_coords = pack_dims(math.meshgrid(x=1, y=3), 'x,y', instance('particles')) * ( (0.018/1), (0.204/3) ) + (-0.015, 0.003) 

fluid_particles = PointCloud(Sphere(fluid_coords, radius=0.002))
wall_particles = PointCloud(Sphere(wall_coords, radius=0.002))

fluid_particle_velocity = fluid_particles*(0, 0)
wall_particle_velocity = wall_particles*(0, 0)

#fluid_coords = fluid_particles.elements.center

#print('original position')
#math.print(fluid_coords)

#fluid_particle_velocity = fluid_particle_velocity + a *2

#print('updated velocity: ')
#math.print(fluid_particle_velocity['x'].values)

#print('ALL PARTICLE VELOCITY')
#particle_velocity = math.concat([fluid_particle_velocity, wall_particle_velocity], 'particles')
#math.print(particle_velocity.values)

#particle_velocity_matrix = particle_velocity.values - math.rename_dims(particle_velocity.values, 'particles', 'others')

coords=math.concat([fluid_coords,wall_coords],'particles')
distance_matrix_vec= coords - math.rename_dims(coords, 'particles', 'others') # contains both the x and y component of separation between particles
distance_matrix = math.vec_length(distance_matrix_vec) # contains magnitude of distance between ALL particles

print('original distance matrix')
math.print(distance_matrix)



distance_matrix = distance_matrix.particles[:fluid_coords.particles.size].others[fluid_coords.particles.size:]

print('cutt off')
math.print(distance_matrix)

fpwnd= math.zeros(instance(distance_matrix))
#print('fluid particle wall neighbour density')
#print(fpwnd)

term= tensor([2,3,4], instance('particles') )

wall_coords = math.rename_dims(wall_coords, 'particles','others')

term=expand(term, instance(wall_coords))

#
print('expanded term')
print(term)
math.print(term)


#fpwnd_new = math.where()




#print('vel_x')
#math.print(math.rename_dims(fluid_particle_velocity['x'].values,'particles','others'))
#print('final')
#print(fluid_particle_velocity)

fluid_particle_velocity=math.expand(fluid_particle_velocity.values, instance(fluid_particle_velocity))

#print('hh')
res = distance_matrix.others*fluid_particle_velocity['x'].particles

#math.print(res)



#math.print(particle_velocity_matrix['x'])

#print('max_velocity_magnitude')
#math.print(math.max(math.vec_length(fluid_particle_velocity.values)))

#fluid_coords = fluid_coords + fluid_particle_velocity*2

#print('update position_x')
#math.print(fluid_coords.values['x'])