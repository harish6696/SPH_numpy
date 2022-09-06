from kernal_function import *
from sklearn import neighbors

def calculate_acceleration(particles, fluid_particle_indices,h, d, r_c,alpha, c_0, g, rho_0, p_0, Xi,gamma):
    #acceleration ofonly fluid particles calculated
    a_x = np.zeros(len(fluid_particle_indices))
    a_y = np.zeros(len(fluid_particle_indices))

    dx=h; epsilon= 0.01 # parameter to avoid zero denominator 

    # Find the nearest neighbour of all particles
    positions = list(zip(particles[:,0], particles[:,1])) # stores x,y in a tuple
    neighbor_ids, distances = neighbors.KDTree(positions).query_radius(positions,r_c,return_distance=True,sort_results=True) # Find the neighbours within the query radius
    positions=np.array(positions)
    #Nearest neighour list of each fluid particles
    fluid_particle_neighbours = neighbor_ids[:len(fluid_particle_indices)]# this array has the neighbours of all the fluid particles including self particle index.
    #fluid_particle_neighbour_distances = distances[:len(fluid_particle_indices)] # array of arrays(distance between the source particle and its neighbour), first element is 0 (distance between source and source)

    for n in range(len(fluid_particle_indices)): #loop over 5000 fluid particles
        a=fluid_particle_indices[n] # w = 1,2,3..n:  where n is the number of fluid particles
        #print("For fluid index "+ str(a))
        for m in range(1,len(fluid_particle_neighbours[n])): ###range should start from 1 because neighbours has the particle itself with zero distance
            b=fluid_particle_neighbours[n][m]
            #print("For fluid neighbour index "+ str(b))
            # distance between particles
            drx = particles[a,0] - particles[b,0]
            dry = particles[a,1] - particles[b,1]
            rad = np.sqrt((drx**2) + (dry**2))
            #print("Distance between particles: "+str(a)+"and "+str(b)+" is: "+str(rad))     
            if(rad==0):
                print("a is: " + str(a) + " type is: "+ str(particles[a,3]))
                print("b is: " +str(b)+ " type is: "+ str(particles[b,3]))
                print("/n")

            q = rad / h
            # derivative of Kernel
            der_W = (kernal_derivative(d, h, q)) / h

            # velocity difference between particles
            dvx = particles[a,6] - particles[b,6]
            dvy = particles[a,7] - particles[b,7]
            
            # pressure of particles
            p_a = particles[a,8]
            p_b = particles[b,8]

            # particle type
            flag_a = int(particles[a,3])  #"3rd index" of particles is the flag, flag=1 is boundary and flag=0 is fluid
            flag_b = int(particles[b,3])
           
            # densities, mass
            rho_a = particles[a,5]
            m_a   = particles[a,4]
            if(flag_b == 1): # if b is a boundary particle
                # calculate density (Eq 15)
                #if(rho_0[flag_a]==0 or np.isnan(rho_0[flag_a])):
                #    print("rho_0[0] is "+ str(rho_0[flag_a]))
                #if(particles[a,8]==0 or np.isnan(particles[a,8])):
                #    print("Fluid Particle Density is "+ str(particles[a,8]))
                #     particles[a,8]=1000
                #    rho_b = rho_0[flag_a] * (((particles[a,8] - Xi[flag_a]) / p_0[flag_a]) + 1) **(1/gamma[flag_a])
                #else:     
                #print("For source particle: "+ str(a)+ " and wall neighbour "+ str(b)+ "|rho_0[0]= "+str(rho_0[flag_a])+ "| particles[a,8]= "+str(particles[a,8])+ "| p_0[0] "+ str(p_0[flag_a])+"| Xi is "+ str(Xi[flag_a])+"| Gamma= "+str(gamma[flag_a]))   
                rho_b = rho_0[flag_a] * (((particles[a,8] - Xi[flag_a]) / p_0[flag_a]) + 1) **(1/gamma[flag_a])
                m_b = rho_0[flag_a] * dx * dx    
                #if(rho_b==0):
                #    print("Zero rho_b density value by particle index "+ str(b)+" of type: "+ str(flag_b)) 
                    
            else: # if b is a fluid particle
                rho_b = particles[b,5]
                m_b   = particles[b,4]
                
            rho_ab = 0.5 * (rho_a + rho_b)
            
            if(rho_a==0):
                print("Zero rho_a density value by particle index "+ str(a)+" of type: "+ str(flag_a))
                rho_a=1000
            if(m_a==0):
                print("Zero m_a value by particle index "+ str(a)+" of type: "+ str(flag_a))

             # momentum equation Pressure Gradient part
            p_ab = ((rho_b * p_a) + (rho_a * p_b)) / (rho_a + rho_b) 


            pressure_fact = - (1/m_a) * ((m_a/rho_a)**2 + (m_b/rho_b)**2) * (p_ab) * der_W #equation 5
             
            
            # acceleration due to pressure gradient (eq7) 
            a_x[a] = a_x[a] + (pressure_fact * (drx / rad))
            a_y[a] = a_y[a] + (pressure_fact * (dry / rad))

            #print(a_x[a])
            #breakpoint()

            # if free slip condition applies only consider particles which are not fluid particles                       
            # artificial viscosity 
            if flag_b != 1: # so if particle b is a fluid (flag=0) for wall particle (flag=1)
                alpha_ab = 0.5 * (alpha[flag_a] + alpha[flag_b])
                c_ab = 0.5 * (c_0[flag_a] + c_0[flag_b])
            else:   #if particle b is a wall particle
                alpha_ab = alpha[flag_a]
                c_ab = c_0[flag_a]
                    
            if (((drx * dvx) + (dry * dvy)) < 0) :
                #print("Calculating visc art fact inside velocity derivatives")
                visc_art_fact = m_b * alpha_ab * h * c_ab * (((dvx * drx) + (dvy * dry))/(rho_ab * ((rad**2) + epsilon * (h**2)))) * der_W
            else:
                visc_art_fact = 0               
            
            a_x[a] = a_x[a] + ((visc_art_fact) * (drx / rad))
            a_y[a] = a_y[a] + ((visc_art_fact) * (dry / rad))

        # Gravity
        a_x[a] = a_x[a] + 0
        a_y[a] = a_y[a] + g

    #print("Reached end of vel derivative 3")
    #breakpoint()
    return a_x, a_y





