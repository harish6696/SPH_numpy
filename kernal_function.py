
import numpy as np
from phi.flow import *

def kernal_function(d,h,q):
    # Quintic Spline used as Weighting function
    # cutoff radius r_c = 3 * h;
    # normalisation parameter alpha_d
    if d == 2:
        alpha_d = 0.004661441847880 / (h * h)
    elif d == 3:
        alpha_d = 1 / (120 * math.pi * (h**3)) # for 3D
    elif d == 1:
        alpha_d = 1 / (120 * h)

    # Weighting function
    W= math.where((q<3) & (q>=2) ,  alpha_d * ((3-q)**5)                                     ,q)
    W= math.where((q<2) & (q>=1) ,  alpha_d * (((3-q)**5) - 6 * ((2-q)**5))                  ,q)
    W= math.where((q<1)          ,  alpha_d * (((3-q)**5) - 6 * ((2-q)**5) + 15 * ((1-q)**5)),q)
    W= math.where((q>=3)         ,  0                                                        ,q)

    return W


def kernal_derivative(d, h, q):
    # normalisation parameter
    if d == 2:      # for 2D
        alpha_d = -0.0233072092393989 / (h * h)
    elif d == 3:
        alpha_d = -5 / (120 * math.pi * (h**3)) # for 3D
    elif d == 1:
        alpha_d = -5 / (120 * h)
    
    # Derivative of Weighting function
    der_W= math.where((q<3) & (q>=2) ,  alpha_d * ((3-q)**4)                                     ,q)
    der_W= math.where((q<2) & (q>=1) ,  alpha_d * (((3-q)**4) - 6 * ((2-q)**4))                 ,q)
    der_W= math.where((q<1)          ,  alpha_d * (((3-q)**4) - 6 * ((2-q)**4) + 15 * ((1-q)**4)),q)
    der_W= math.where((q>=3)         ,  0                                                        ,q)

    return der_W

