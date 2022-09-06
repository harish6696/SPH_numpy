
import numpy as np

def kernal_function(d,h,q):
    # Quintic Spline
    # cutoff radius r_c = 3 * h;

    # normalisation parameter alpha_d
    if d == 2:
        #alpha_d = 7 / (478 * pi * h^2); # for 2D
        alpha_d = 0.004661441847880 / (h * h)
    elif d == 3:
        alpha_d = 1 / (120 * np.pi * (h**3)) # for 3D
    elif d == 1:
        alpha_d = 1 / (120 * h)
    

    # Weighting function
    if q < 3 and q >= 2:
        W = alpha_d * ((3-q)**5)
    elif q < 2 and q >= 1:
        W = alpha_d * (((3-q)**5) - 6 * ((2-q)**5))
    elif q < 1:
        W = alpha_d * (((3-q)**5) - 6 * ((2-q)**5) + 15 * ((1-q)**5))
    elif q >= 3:
        W = 0
    
    return W


def kernal_derivative(d, h, q):
    # normalisation parameter
    if d == 2:
        #alpha_d = -5 * 7 / (478 * pi * h^2)  # for 2D
        alpha_d = -0.0233072092393989 / (h * h)
    elif d == 3:
        alpha_d = -5 / (120 * np.pi * (h**3)) # for 3D
    elif d == 1:
        alpha_d = -5 / (120 * h)
    

    # derivative of Weighting function
    if q < 3 and q >= 2:
        der_W = alpha_d * ((3-q)**4)
    elif q < 2 and q >= 1:
        der_W = alpha_d * (((3-q)**4) - 6 * ((2-q)**4))
    elif q < 1:
        der_W = alpha_d * (((3-q)**4) - 6 * ((2-q)**4) + 15 * ((1-q)**4))
    elif q >= 3:
        der_W = 0
    
    return der_W

