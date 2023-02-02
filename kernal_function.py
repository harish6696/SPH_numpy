
import numpy as np
#from phi.flow import *
from phi.flow import *
#math.set_global_precision(32)

def kernal_function(d,h,q):
    # Quintic Spline used as Weighting function
    # cutoff radius r_c = 3 * h;
    # normalisation parameter alpha_d
    #if d == 2:
    alpha_d = 0.004661441847880 / (h * h)
    # elif d == 3:
    #     alpha_d = 1 / (120 * math.pi * (h**3)) # for 3D
    # elif d == 1:
    #     alpha_d = 1 / (120 * h)
    #print('q inside kernal')
    #math.print(q)
    # Weighting function
    W= math.where((q<3) & (q>=2) ,  alpha_d * ((3-q)**5)                                     ,q)
    #print('W1')
    #math.print(W1)
    W= math.where((q<2) & (q>=1) ,  alpha_d * (((3-q)**5) - 6 * ((2-q)**5))                  ,W)
    #print('W2')
    #math.print(W2)
    W= math.where((q<1) & (q>0)  ,  alpha_d * (((3-q)**5) - 6 * ((2-q)**5) + 15 * ((1-q)**5)),W)
    #print('W3')
    #math.print(W3)
    W= math.where((q>=3)         ,  0                                                        ,W)
    #print('final W')
    #math.print(W)
    #breakpoint()
    return W


def kernal_derivative(d, h, q):
    # normalisation parameter
    #if d == 2:      # for 2D
    alpha_d = -0.0233072092393989 / (h * h)
        #print(alpha_d)
    # elif d == 3:
    #     alpha_d = -5 / (120 * math.pi * (h**3)) # for 3D
    # elif d == 1:
    #     alpha_d = -5 / (120 * h)
    
    # Derivative of Weighting function
    #der_W = math.zeros(instance(q))
    
    der_W= math.where((q<3) & (q>=2) ,  alpha_d * ((3-q)**4)                                     ,q)
    der_W= math.where((q<2) & (q>=1) ,  alpha_d * (((3-q)**4) - 6 * ((2-q)**4))                  ,der_W)
    der_W= math.where((q<1) & (q>0)  ,  alpha_d * (((3-q)**4) - 6 * ((2-q)**4) + 15 * ((1-q)**4)),der_W)
    der_W= math.where((q>=3)         ,  0                                                        ,der_W)
    
    # print('der W')
    # math.print(der_W)
    # breakpoint()
    return der_W

