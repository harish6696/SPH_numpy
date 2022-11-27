from kernal_function import *
from phi.flow import *

def calculate_acceleration(fluid_particles, wall_particles,wall_particle_velocity ,fluid_particle_density,fluid_particle_pressure,wall_particle_pressure,fluid_particle_mass,fluid_particle_velocity,fluid_initial_density,fluid_Xi,fluid_adiabatic_exp,fluid_p_0, h, d, r_c, dx, fluid_alpha,fluid_c_0,g):

    dx=h; epsilon= 0.01 # parameter to avoid zero denominator 

    fluid_coords=fluid_particles.points   #fluid_particles and boundary_particles are a point cloud objects
    wall_coords=wall_particles.points
    particle_coords=math.concat([fluid_coords,wall_coords],'particles')  # concatenating fluid coords and then boundary coords
    

    fluid_particle_density = math.rename_dims(fluid_particle_density,'particles','others')
    wall_particle_density=math.rename_dims(math.zeros(instance(wall_coords)),'particles','others')

    #print('fluid old density')
    #print(fluid_particle_density)
    
    fluid_particle_mass = math.rename_dims(fluid_particle_mass,'particles', 'others')
    wall_particle_mass=math.rename_dims(math.zeros(instance(wall_coords)),'particles','others')
    ###CHECK THIS IF REQUIRED WHILE RUNNING THE CODE
    
    fluid_particle_velocity = math.rename_dims(fluid_particle_velocity, 'particles', 'others')
    wall_particle_velocity=math.rename_dims(wall_particle_velocity,'particles','others')

    fluid_particle_pressure = math.rename_dims(fluid_particle_pressure, 'particles', 'others')
    wall_particle_pressure=math.rename_dims(wall_particle_pressure,'particles','others')
    
    particle_velocity = math.concat([fluid_particle_velocity.values,wall_particle_velocity.values], dim='others')
    particle_density= math.concat([fluid_particle_density,wall_particle_density], dim='others') #1D Scalar array of densities of all particles
    particle_mass= math.concat([fluid_particle_mass,wall_particle_mass], dim='others')
    particle_pressure=math.concat([fluid_particle_pressure,wall_particle_pressure], dim='others')

    #Compute distance between all particles 
    distance_matrix_vec= particle_coords - math.rename_dims(particle_coords, 'particles', 'others') # contains both the x and y component of separation between particles
    distance_matrix = math.vec_length(distance_matrix_vec) # contains magnitude of distance between ALL particles

    alpha_ab = math.where(distance_matrix==0,0,fluid_alpha) #Removing the particle itself from further calculation
    alpha_ab=math.where(distance_matrix> r_c, 0, alpha_ab)   #N X N matrix N-->all particles
    fluid_particle_neighbour_alpha_ab=alpha_ab.particles[:fluid_coords.particles.size].others[:]
    
    c_ab = math.where(distance_matrix==0,0,fluid_c_0) #Removing the particle itself from further calculation
    c_ab=math.where(distance_matrix> r_c, 0, c_ab)   #N X N matrix N-->all particles
    fluid_particle_neighbour_c_ab=c_ab.particles[:fluid_coords.particles.size].others[:]

    #Below we create matrices which have non-zero entries where the neighbour is inside cut-off radius 'r_c' rest all entries are 0
    particle_velocity=math.rename_dims(particle_velocity,'others','particles')
    particle_velocity_matrix =  particle_velocity- math.rename_dims(particle_velocity, 'particles', 'others') 

    particle_neighbour_density = math.where(distance_matrix==0,0,particle_density)  #Removing the particle itself from further calculation
    particle_neighbour_density = math.where(distance_matrix > r_c, 0, particle_neighbour_density) #0 for places which are not neighbours for particle under consideration. Stacks the particle density vertically. i.e. dupicates the densities
    
    particle_neighbour_mass = math.where(distance_matrix==0,0,particle_mass) #Removing the particle itself from further calculation
    particle_neighbour_mass = math.where(distance_matrix > r_c, 0, particle_neighbour_mass)

    particle_neighbour_pressure = math.where(distance_matrix==0,0,particle_pressure) #Removing the particle itself from further calculation
    particle_neighbour_pressure= math.where(distance_matrix > r_c,0, particle_neighbour_pressure)

    dvx=particle_velocity_matrix['x'].particles[:fluid_coords.particles.size].others[:]# separating the x and y components
    dvy=particle_velocity_matrix['y'].particles[:fluid_coords.particles.size].others[:]

    fluid_particle_all_neighbours_dist=distance_matrix.particles[:fluid_coords.particles.size].others[:]
    dvx=math.where(fluid_particle_all_neighbours_dist>r_c,0,dvx) # relative x-velocity between a fluid particle and its neighbour
    dvy=math.where(fluid_particle_all_neighbours_dist>r_c,0,dvy)

    ######Do all the cut off things before the final cut off for distance matrix
    distance_matrix_vec= math.where(distance_matrix > r_c, 0, distance_matrix_vec)
    distance_matrix = math.where(distance_matrix > r_c, 0, distance_matrix)   # Stores the distance between neighbours which are inside cutoff radius

 
    #Slicing the distance matrix of fluid neighbour of fluid particles
    rad=distance_matrix.particles[:fluid_coords.particles.size].others[:]

    q=rad/h 

    der_W = kernal_derivative(d, h, q) / h
    #print('hhh')
    #math.print(der_W)
    #breakpoint()

    drx=distance_matrix_vec['x'].particles[:fluid_coords.particles.size].others[:]
    dry=distance_matrix_vec['y'].particles[:fluid_coords.particles.size].others[:] 
    
    #rho_b, m_b and p_b matrices rows are for each fluid particle and cols are fluid and wall particles
    ########THE BELOW 3 MATRICES CAN BE REMOVED
    fluid_particle_neighbour_density=particle_neighbour_density.particles[:fluid_coords.particles.size].others[:]
    fluid_particle_neighbour_mass=particle_neighbour_mass.particles[:fluid_coords.particles.size].others[:]
    fluid_particle_neighbour_pressure = particle_neighbour_pressure.particles[:fluid_coords.particles.size].others[:]

    #Splitting the density and mass of fluid particle neighbours into two matrices....QQQQQQ CAN WE AVOID THIS? as we have to join it back anyways
    fluid_particle_fluid_neighbour_density=fluid_particle_neighbour_density.particles[:].others[:fluid_coords.particles.size]
    fluid_particle_wall_neighbour_density=fluid_particle_neighbour_density.particles[:].others[fluid_coords.particles.size:]
    fluid_particle_fluid_neighbour_mass=fluid_particle_neighbour_mass.particles[:].others[:fluid_coords.particles.size]
    fluid_particle_wall_neighbour_mass=fluid_particle_neighbour_mass.particles[:].others[fluid_coords.particles.size:]

    ######'fluid_pressure' is for each fluid particle 
    fluid_particle_wall_neighbour_density_term=fluid_initial_density * ((fluid_particle_pressure - fluid_Xi) / fluid_p_0 + 1)**(1/fluid_adiabatic_exp)
    #print('term:')
    #print(fluid_particle_wall_neighbour_density_term)
    fluid_particle_wall_neighbour_density_term= math.rename_dims(fluid_particle_wall_neighbour_density_term,'others','particles')
    helper_matrix = distance_matrix.particles[:fluid_coords.particles.size].others[fluid_coords.particles.size:] 
    helper_matrix = math.where(helper_matrix!=0, 1 , helper_matrix)  # technically called adjacency matrix
    #print('helper matrix:')
    #print(helper_matrix)

    #ALL boundary particle neighbours for a given fluid particle will have same density 
    fluid_particle_wall_neighbour_density = helper_matrix * fluid_particle_wall_neighbour_density_term
    
    #print('fpwnd')
    #print(fluid_particle_wall_neighbour_density)
    #breakpoint()

    #fluid_particle_wall_neighbour_density=math.where(fluid_particle_wall_neighbour_density!=0,fluid_particle_wall_neighbour_density_term,fluid_particle_wall_neighbour_density)
    fluid_particle_wall_neighbour_mass =math.where(helper_matrix!=0,(fluid_initial_density*dx*dx),fluid_particle_wall_neighbour_mass)



    #joining the density and mass of the fluid and wall particle back together
    fluid_particle_neighbour_density=math.concat([fluid_particle_fluid_neighbour_density, fluid_particle_wall_neighbour_density], 'others')
    fluid_particle_neighbour_mass =math.concat([fluid_particle_fluid_neighbour_mass, fluid_particle_wall_neighbour_mass], 'others')

    #print('fluid_particle_ALL_neighbour_density: ')
    #print(fluid_particle_neighbour_density)
    #print('fluid_particle_ALL_neighbour_mass: ')
    #print(fluid_particle_neighbour_mass)
    
    #Momentum Equation for the pressure gradient part
    #######CHECK THIS below line WHILE RUNNING THE FULL CODE
    fluid_particle_density= math.rename_dims(fluid_particle_density,'others','particles') 
    fluid_particle_pressure = math.rename_dims(fluid_particle_pressure, 'others', 'particles')
    fluid_particle_mass = math.rename_dims(fluid_particle_mass,'others','particles')
        
    #(rho_a + rho_b) 
    particle_density_sum=math.where(fluid_particle_neighbour_density!=0, fluid_particle_density+fluid_particle_neighbour_density,fluid_particle_neighbour_density)
    #print('particle density sum')
    #print(particle_density_sum)

    #rho_ab = 0.5 * (rho_a + rho_b)
    rho_ab=0.5*particle_density_sum


    #Setting minimum density as the initial density
    fluid_particle_density=math.where(fluid_particle_density==0,fluid_initial_density,fluid_particle_density)
    #print('fluid particle density')
    #print(fluid_particle_density)
    #print('fpnd')
    #print(fluid_particle_neighbour_density)
    #print('fluid_particle_pressure')
    #print(fluid_particle_pressure)

    #p_ab = ((rho_b * p_a) + (rho_a * p_b)) / (rho_a + rho_b) 
    term_1 = fluid_particle_neighbour_density*fluid_particle_pressure
    #print('term 1')
    #print(term_1)

    term_2 = fluid_particle_neighbour_pressure*fluid_particle_density
    #print('term 2')
    #print(term_2)
    
    p_ab= math.divide_no_nan(((term_1) + (term_2)),(particle_density_sum))
    #print('p_ab')
    #print(p_ab)

    
    #pressure_fact = - (1/m_a) * ((m_a/rho_a)**2 + (m_b/rho_b)**2) * (p_ab) * der_W #equation 5
    
    fluid_neighbour_mass_by_density_ratio=math.divide_no_nan(fluid_particle_neighbour_mass,fluid_particle_neighbour_density)**2 #(m_b/rho_b)² 
    fluid_mass_by_density_ratio=math.divide_no_nan(fluid_particle_mass,fluid_particle_density)**2  # (m_a/rho_a)²



    #print('m_b/rho_b')
    #print(fluid_neighbour_mass_by_density_ratio)

    #print('m_a/rho_a')
    #print(fluid_mass_by_density_ratio)

    #sum = (m_a/rho_a)**2 + (m_b/rho_b)**2
    mass_by_density_sum =math.where(fluid_neighbour_mass_by_density_ratio!=0,fluid_mass_by_density_ratio+fluid_neighbour_mass_by_density_ratio,fluid_neighbour_mass_by_density_ratio)

    # pressure_fact=-(1/m_a) * ((m_a/rho_a)**2 + (m_b/rho_b)**2) * (p_ab) * der_W (equation 5)
    pressure_fact=(-1/fluid_particle_mass)*mass_by_density_sum*p_ab*der_W

    #print('fluid_particle_mass')
    #print(fluid_particle_mass)

    #print('mass by density sum')
    #print(mass_by_density_sum)

    #print('p_ab')
    #print(p_ab)

    #print('der W')
    #print(der_W)

    #print('pressure_fact')
    #print(pressure_fact)

    #a_x
    a_x=math.sum(pressure_fact*math.divide_no_nan(drx,rad),'others')
    #a_y
    a_y=math.sum(pressure_fact*math.divide_no_nan(dry,rad),'others')
    
    ### ARTIFICIAL VISCOSITY
    visc_art_fact=math.zeros(instance(fluid_particle_all_neighbours_dist))  #zeros matrix with size: (fluid_particles X all_particles)
    temp_matrix= (drx*dvx) + (dry*dvy) 

    # visc_art_fact_term = m_b * alpha_ab * h * c_ab * (((dvx * drx) + (dvy * dry))/(rho_ab * ((rad**2) + epsilon * (h**2)))) * der_W
    visc_art_fact_term = fluid_particle_neighbour_mass * fluid_particle_neighbour_alpha_ab * h * fluid_particle_neighbour_c_ab* ((temp_matrix)/(rho_ab * ((rad**2) + epsilon * (h**2)))) * der_W
    visc_art_fact = math.where(temp_matrix<0, visc_art_fact_term, visc_art_fact)


    #a_x
    a_x=a_x+math.sum(visc_art_fact*math.divide_no_nan(drx,rad),'others')
    #a_y
    a_y=a_y+math.sum(visc_art_fact*math.divide_no_nan(dry,rad),'others')

    #### GRAVITY
    a_y=a_y+g

    fluid_particle_acceleration = stack([a_x, a_y], channel(vector='x,y'))
    #Acceleration of only fluid particles calculated

    #print('hhh')
    #math.print(a_x)
    #breakpoint()
    #print('REached end of calculate_acc()....hurray!')
    #breakpoint()
    return fluid_particle_acceleration





