
import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as plt
from kernal_function import *

def boundary_update(particles, fluid_particle_indices,wall_particle_indices,d,r_c,h,g):

    #particles is a numpy array
    # velocity components for ALL(fluid and solid) particles
    v_wall=np.zeros((len(particles[:,0]),2))
    
    #assigning prescribed wall velocity (by default 0)
    v_wall[len(fluid_particle_indices):,0]=particles[len(fluid_particle_indices):,4] 
    v_wall[len(fluid_particle_indices):,1]=particles[len(fluid_particle_indices):,5]

    # velocity components for ALL(fluid and solid) particles
    vel_x=particles[:,6]
    vel_y=particles[:,7]

    pressure=np.zeros(len(wall_particle_indices))
    positions = list(zip(particles[:,0], particles[:,1])) # stores x,y in a tuple
    neighbor_ids, distances = neighbors.KDTree(positions).query_radius(positions,r_c,return_distance=True,sort_results=True) # Find the neighbours within the query radius
    positions=np.array(positions)

    neighbor_ids=np.array(neighbor_ids)
    wall_particle_neighbours = neighbor_ids[len(fluid_particle_indices):]# this array has ALL the neighbours of all the wall particles including self particle index.
    #wall_particle_neighbours =  # this array has ONLY FLUID neighbours of all the wall particles
    #[item for item in wall_particle_neighbours if item < len(fluid_particle_indices)]
    #print(wall_particle_neighbours[1])
    #breakpoint()
    wall_particle_fluid_neighbours=[]
    for i in wall_particle_neighbours:
        temp=[]
        for j in i:
            if(j<len(fluid_particle_indices) or j==i[0]):
                temp.append(j)
        wall_particle_fluid_neighbours.append(temp)

    #print(wall_particle_fluid_neighbours)
    # print("Done")
    #breakpoint()

    for n in range(len(wall_particle_indices)): #loop over 1629 wall-particles 
        w=wall_particle_indices[n] # w = n,n+1,....n+1628,, n is the number of fluid particles
        # print("For wall index "+ str(w))
        sum_pW=0.0;sum_rhorW=0.0;sum_W=0.0 # initializing wall pressure
        sum_vWx=0.0;sum_vWy=0.0 # initializing wall velocities

        for m in range(1,len(wall_particle_fluid_neighbours[n])): ###range should start from 1 because neighbours has the particle itself with zero distance
            f=wall_particle_fluid_neighbours[n][m]
            #print("For neighbour index "+ str(f))
            drx=particles[w,0]-particles[f,0] 
            dry=particles[w,1]-particles[f,1]
            rad = np.sqrt((drx**2) + (dry**2))
            #print("Distance between particles: "+str(w)+"and "+str(f)+" is: "+str(rad))     
            """ 
            print("First source index is: "+str(w))
            print("First neighbour index is: "+str(f))
            
            rad=np.sqrt(drx**2 + dry**2)
            print(rad)
            
            breakpoint()

            diff=rad-wall_particle_distances[n][m]
            print(str(diff) + "\t") 
            """
            
            #q= wall_particle_distances[n][m]/h # non-dimensional distance
            q = rad / h
            #rad = np.sqrt(drx^2 + dry^2)
            W = kernal_function(d,h,q)  #kernal function (Quintic Spline)
    
            sum_pW = sum_pW + (particles[f,8]*W)
            sum_rhorW = sum_rhorW + (particles[f,5] * W *((drx * 0) + (dry * g)))
            sum_W = sum_W + W

            # building up the SPH average for wall velocity(no slip boundary condition)
            sum_vWx = sum_vWx + (vel_x[f] * W)
            sum_vWy = sum_vWy + (vel_y[f] * W)
        
        #print("\n ")
        if (sum_W==0):
            pressure[n]=0

        else:
            # combining terms to get pressure acting on source wall particle 
            pressure[n] = ( sum_pW + sum_rhorW ) / sum_W 

            # calculate wall velocity(no slip boundary condition) 
            vel_x[w] = (2 * v_wall[w,0]) - (sum_vWx / sum_W)
            vel_y[w] = (2 * v_wall[w,1]) - (sum_vWy / sum_W)
    #breakpoint()
    return pressure,vel_x,vel_y


    #PLOTTING
    """     plt.scatter(*zip(*positions))
    labels = ["%i" % s for s in range(len(positions))]
    ax = plt.axes()
    for u in range(len(positions)):
        plt.text(positions[u, 0],positions[u, 1], '%s' % (labels[u]), size=10, zorder=1, color='k')

    plt.show() """



                
    

