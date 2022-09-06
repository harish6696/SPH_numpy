import random
from turtle import position
import numpy as np
import matplotlib.pyplot as plt

def distance(p, points, min_distance):
    """
    Determines if any points in the list are less than the minimum specified 
    distance apart.

    Parameters
    ----------
    p : tuple
        `(x,y,z)` point.
    points : ndarray
        Array of points to check against. `x, y, z` points are columnwise.
    min_distance : float
        Minimum allowable distance between any two points.

    Returns
    -------
    bool
        True if point `p` is at least `min_distance` from all points in `points`.

    """
    distances = np.sqrt(np.sum((p+points)**2, axis=1))
    distances = np.where(distances < min_distance)
    return distances[0].size < 1

def return_points(xmin,xmax,ymin,ymax,zmin,zmax,N):
    points = np.array([])       # x, y, z columnwise
    while points.shape[0] < N:
        x = random.choice(np.linspace(xmin, xmax, 100000))  #Set limits of x bounds
        y = random.choice(np.linspace(ymin, ymax, 100000))  #Set limits of y bounds
        z = random.choice(np.linspace(zmin, zmax, 100000))   #Set limits of z bounds
        p = (x,y,z)
        if len(points) == 0:                # add first point blindly
            points = np.array([p])
        elif distance(p, points, 0.03):     # ensure the minimum distance is met
            points = np.vstack((points, p))
    # print(points.shape)
    # points=np.delete(points, 1,0) # deletes entire row 1 (Remember indices are from 0 )
    # print(points.shape)
    assert np.shape(points)[0] == N, "Points generated are less than requested"
    return points

def plot_points(points):
    
    fig = plt.figure(figsize=(18,9))
    ax = plt.axes(projection='3d')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_zlim([-10, 10])
    ax.set_title('Particle Positions',fontsize=18)
    ax.set_xlabel('X',fontsize=14)
    ax.set_ylabel('Y',fontsize=14)
    ax.set_zlabel('Z',fontsize=14)
    ax.scatter(points[:,0], points[:,1], points[:,2])
    plt.show()



if __name__== "__main__":
    points=return_points(10,15,50,60,0,1,1000)
    print(points.shape)
    plot_points(points)