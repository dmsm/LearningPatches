import torch as th
from torch.nn import functional as F


def coons_mtds(s, t, params):
    # params [..., 12, 3]
    params = params[...,None]
    det = ((params[...,0,0,:]*(s - 1)**3 - params[...,9,0,:]*(s - 1)**3 - s*(3*params[...,3,0,:]*(t - 1)**2 - 3*params[...,4,0,:]*(t - 1)**2 + 3*params[...,5,0,:]*t**2 - 3*params[...,6,0,:]*t**2 + 6*params[...,5,0,:]*t*(t - 1) - 3*params[...,4,0,:]*t*(2*t - 2)) + params[...,3,0,:]*s - params[...,6,0,:]*s + (s - 1)*(3*params[...,0,0,:]*(t - 1)**2 - 3*params[...,11,0,:]*(t - 1)**2 - 3*params[...,9,0,:]*t**2 + 3*params[...,10,0,:]*t**2 + 6*params[...,10,0,:]*t*(t - 1) - 3*params[...,11,0,:]*t*(2*t - 2)) - params[...,0,0,:]*(s - 1) + params[...,9,0,:]*(s - 1) - params[...,3,0,:]*s**3 + params[...,6,0,:]*s**3 - 3*params[...,1,0,:]*s*(s - 1)**2 + 3*params[...,2,0,:]*s**2*(s - 1) - 3*params[...,7,0,:]*s**2*(s - 1) + 3*params[...,8,0,:]*s*(s - 1)**2)**2 + (params[...,0,1,:]*(s - 1)**3 - params[...,9,1,:]*(s - 1)**3 - s*(3*params[...,3,1,:]*(t - 1)**2 - 3*params[...,4,1,:]*(t - 1)**2 + 3*params[...,5,1,:]*t**2 - 3*params[...,6,1,:]*t**2 + 6*params[...,5,1,:]*t*(t - 1) - 3*params[...,4,1,:]*t*(2*t - 2)) + params[...,3,1,:]*s - params[...,6,1,:]*s + (s - 1)*(3*params[...,0,1,:]*(t - 1)**2 - 3*params[...,11,1,:]*(t - 1)**2 - 3*params[...,9,1,:]*t**2 + 3*params[...,10,1,:]*t**2 + 6*params[...,10,1,:]*t*(t - 1) - 3*params[...,11,1,:]*t*(2*t - 2)) - params[...,0,1,:]*(s - 1) + params[...,9,1,:]*(s - 1) - params[...,3,1,:]*s**3 + params[...,6,1,:]*s**3 - 3*params[...,1,1,:]*s*(s - 1)**2 + 3*params[...,2,1,:]*s**2*(s - 1) - 3*params[...,7,1,:]*s**2*(s - 1) + 3*params[...,8,1,:]*s*(s - 1)**2)**2 + (params[...,0,2,:]*(s - 1)**3 - params[...,9,2,:]*(s - 1)**3 - s*(3*params[...,3,2,:]*(t - 1)**2 - 3*params[...,4,2,:]*(t - 1)**2 + 3*params[...,5,2,:]*t**2 - 3*params[...,6,2,:]*t**2 + 6*params[...,5,2,:]*t*(t - 1) - 3*params[...,4,2,:]*t*(2*t - 2)) + params[...,3,2,:]*s - params[...,6,2,:]*s + (s - 1)*(3*params[...,0,2,:]*(t - 1)**2 - 3*params[...,11,2,:]*(t - 1)**2 - 3*params[...,9,2,:]*t**2 + 3*params[...,10,2,:]*t**2 + 6*params[...,10,2,:]*t*(t - 1) - 3*params[...,11,2,:]*t*(2*t - 2)) - params[...,0,2,:]*(s - 1) + params[...,9,2,:]*(s - 1) - params[...,3,2,:]*s**3 + params[...,6,2,:]*s**3 - 3*params[...,1,2,:]*s*(s - 1)**2 + 3*params[...,2,2,:]*s**2*(s - 1) - 3*params[...,7,2,:]*s**2*(s - 1) + 3*params[...,8,2,:]*s*(s - 1)**2)**2)*((params[...,0,0,:]*(t - 1)**3 - params[...,3,0,:]*(t - 1)**3 + t*(3*params[...,8,0,:]*(s - 1)**2 - 3*params[...,9,0,:]*(s - 1)**2 + 3*params[...,6,0,:]*s**2 - 3*params[...,7,0,:]*s**2 - 6*params[...,7,0,:]*s*(s - 1) + 3*params[...,8,0,:]*s*(2*s - 2)) - params[...,6,0,:]*t + params[...,9,0,:]*t + (t - 1)*(3*params[...,0,0,:]*(s - 1)**2 - 3*params[...,1,0,:]*(s - 1)**2 + 3*params[...,2,0,:]*s**2 - 3*params[...,3,0,:]*s**2 + 6*params[...,2,0,:]*s*(s - 1) - 3*params[...,1,0,:]*s*(2*s - 2)) - params[...,0,0,:]*(t - 1) + params[...,3,0,:]*(t - 1) + params[...,6,0,:]*t**3 - params[...,9,0,:]*t**3 + 3*params[...,4,0,:]*t*(t - 1)**2 - 3*params[...,5,0,:]*t**2*(t - 1) + 3*params[...,10,0,:]*t**2*(t - 1) - 3*params[...,11,0,:]*t*(t - 1)**2)**2 + (params[...,0,1,:]*(t - 1)**3 - params[...,3,1,:]*(t - 1)**3 + t*(3*params[...,8,1,:]*(s - 1)**2 - 3*params[...,9,1,:]*(s - 1)**2 + 3*params[...,6,1,:]*s**2 - 3*params[...,7,1,:]*s**2 - 6*params[...,7,1,:]*s*(s - 1) + 3*params[...,8,1,:]*s*(2*s - 2)) - params[...,6,1,:]*t + params[...,9,1,:]*t + (t - 1)*(3*params[...,0,1,:]*(s - 1)**2 - 3*params[...,1,1,:]*(s - 1)**2 + 3*params[...,2,1,:]*s**2 - 3*params[...,3,1,:]*s**2 + 6*params[...,2,1,:]*s*(s - 1) - 3*params[...,1,1,:]*s*(2*s - 2)) - params[...,0,1,:]*(t - 1) + params[...,3,1,:]*(t - 1) + params[...,6,1,:]*t**3 - params[...,9,1,:]*t**3 + 3*params[...,4,1,:]*t*(t - 1)**2 - 3*params[...,5,1,:]*t**2*(t - 1) + 3*params[...,10,1,:]*t**2*(t - 1) - 3*params[...,11,1,:]*t*(t - 1)**2)**2 + (params[...,0,2,:]*(t - 1)**3 - params[...,3,2,:]*(t - 1)**3 + t*(3*params[...,8,2,:]*(s - 1)**2 - 3*params[...,9,2,:]*(s - 1)**2 + 3*params[...,6,2,:]*s**2 - 3*params[...,7,2,:]*s**2 - 6*params[...,7,2,:]*s*(s - 1) + 3*params[...,8,2,:]*s*(2*s - 2)) - params[...,6,2,:]*t + params[...,9,2,:]*t + (t - 1)*(3*params[...,0,2,:]*(s - 1)**2 - 3*params[...,1,2,:]*(s - 1)**2 + 3*params[...,2,2,:]*s**2 - 3*params[...,3,2,:]*s**2 + 6*params[...,2,2,:]*s*(s - 1) - 3*params[...,1,2,:]*s*(2*s - 2)) - params[...,0,2,:]*(t - 1) + params[...,3,2,:]*(t - 1) + params[...,6,2,:]*t**3 - params[...,9,2,:]*t**3 + 3*params[...,4,2,:]*t*(t - 1)**2 - 3*params[...,5,2,:]*t**2*(t - 1) + 3*params[...,10,2,:]*t**2*(t - 1) - 3*params[...,11,2,:]*t*(t - 1)**2)**2) - ((params[...,0,0,:]*(s - 1)**3 - params[...,9,0,:]*(s - 1)**3 - s*(3*params[...,3,0,:]*(t - 1)**2 - 3*params[...,4,0,:]*(t - 1)**2 + 3*params[...,5,0,:]*t**2 - 3*params[...,6,0,:]*t**2 + 6*params[...,5,0,:]*t*(t - 1) - 3*params[...,4,0,:]*t*(2*t - 2)) + params[...,3,0,:]*s - params[...,6,0,:]*s + (s - 1)*(3*params[...,0,0,:]*(t - 1)**2 - 3*params[...,11,0,:]*(t - 1)**2 - 3*params[...,9,0,:]*t**2 + 3*params[...,10,0,:]*t**2 + 6*params[...,10,0,:]*t*(t - 1) - 3*params[...,11,0,:]*t*(2*t - 2)) - params[...,0,0,:]*(s - 1) + params[...,9,0,:]*(s - 1) - params[...,3,0,:]*s**3 + params[...,6,0,:]*s**3 - 3*params[...,1,0,:]*s*(s - 1)**2 + 3*params[...,2,0,:]*s**2*(s - 1) - 3*params[...,7,0,:]*s**2*(s - 1) + 3*params[...,8,0,:]*s*(s - 1)**2)*(params[...,0,0,:]*(t - 1)**3 - params[...,3,0,:]*(t - 1)**3 + t*(3*params[...,8,0,:]*(s - 1)**2 - 3*params[...,9,0,:]*(s - 1)**2 + 3*params[...,6,0,:]*s**2 - 3*params[...,7,0,:]*s**2 - 6*params[...,7,0,:]*s*(s - 1) + 3*params[...,8,0,:]*s*(2*s - 2)) - params[...,6,0,:]*t + params[...,9,0,:]*t + (t - 1)*(3*params[...,0,0,:]*(s - 1)**2 - 3*params[...,1,0,:]*(s - 1)**2 + 3*params[...,2,0,:]*s**2 - 3*params[...,3,0,:]*s**2 + 6*params[...,2,0,:]*s*(s - 1) - 3*params[...,1,0,:]*s*(2*s - 2)) - params[...,0,0,:]*(t - 1) + params[...,3,0,:]*(t - 1) + params[...,6,0,:]*t**3 - params[...,9,0,:]*t**3 + 3*params[...,4,0,:]*t*(t - 1)**2 - 3*params[...,5,0,:]*t**2*(t - 1) + 3*params[...,10,0,:]*t**2*(t - 1) - 3*params[...,11,0,:]*t*(t - 1)**2) + (params[...,0,1,:]*(s - 1)**3 - params[...,9,1,:]*(s - 1)**3 - s*(3*params[...,3,1,:]*(t - 1)**2 - 3*params[...,4,1,:]*(t - 1)**2 + 3*params[...,5,1,:]*t**2 - 3*params[...,6,1,:]*t**2 + 6*params[...,5,1,:]*t*(t - 1) - 3*params[...,4,1,:]*t*(2*t - 2)) + params[...,3,1,:]*s - params[...,6,1,:]*s + (s - 1)*(3*params[...,0,1,:]*(t - 1)**2 - 3*params[...,11,1,:]*(t - 1)**2 - 3*params[...,9,1,:]*t**2 + 3*params[...,10,1,:]*t**2 + 6*params[...,10,1,:]*t*(t - 1) - 3*params[...,11,1,:]*t*(2*t - 2)) - params[...,0,1,:]*(s - 1) + params[...,9,1,:]*(s - 1) - params[...,3,1,:]*s**3 + params[...,6,1,:]*s**3 - 3*params[...,1,1,:]*s*(s - 1)**2 + 3*params[...,2,1,:]*s**2*(s - 1) - 3*params[...,7,1,:]*s**2*(s - 1) + 3*params[...,8,1,:]*s*(s - 1)**2)*(params[...,0,1,:]*(t - 1)**3 - params[...,3,1,:]*(t - 1)**3 + t*(3*params[...,8,1,:]*(s - 1)**2 - 3*params[...,9,1,:]*(s - 1)**2 + 3*params[...,6,1,:]*s**2 - 3*params[...,7,1,:]*s**2 - 6*params[...,7,1,:]*s*(s - 1) + 3*params[...,8,1,:]*s*(2*s - 2)) - params[...,6,1,:]*t + params[...,9,1,:]*t + (t - 1)*(3*params[...,0,1,:]*(s - 1)**2 - 3*params[...,1,1,:]*(s - 1)**2 + 3*params[...,2,1,:]*s**2 - 3*params[...,3,1,:]*s**2 + 6*params[...,2,1,:]*s*(s - 1) - 3*params[...,1,1,:]*s*(2*s - 2)) - params[...,0,1,:]*(t - 1) + params[...,3,1,:]*(t - 1) + params[...,6,1,:]*t**3 - params[...,9,1,:]*t**3 + 3*params[...,4,1,:]*t*(t - 1)**2 - 3*params[...,5,1,:]*t**2*(t - 1) + 3*params[...,10,1,:]*t**2*(t - 1) - 3*params[...,11,1,:]*t*(t - 1)**2) + (params[...,0,2,:]*(s - 1)**3 - params[...,9,2,:]*(s - 1)**3 - s*(3*params[...,3,2,:]*(t - 1)**2 - 3*params[...,4,2,:]*(t - 1)**2 + 3*params[...,5,2,:]*t**2 - 3*params[...,6,2,:]*t**2 + 6*params[...,5,2,:]*t*(t - 1) - 3*params[...,4,2,:]*t*(2*t - 2)) + params[...,3,2,:]*s - params[...,6,2,:]*s + (s - 1)*(3*params[...,0,2,:]*(t - 1)**2 - 3*params[...,11,2,:]*(t - 1)**2 - 3*params[...,9,2,:]*t**2 + 3*params[...,10,2,:]*t**2 + 6*params[...,10,2,:]*t*(t - 1) - 3*params[...,11,2,:]*t*(2*t - 2)) - params[...,0,2,:]*(s - 1) + params[...,9,2,:]*(s - 1) - params[...,3,2,:]*s**3 + params[...,6,2,:]*s**3 - 3*params[...,1,2,:]*s*(s - 1)**2 + 3*params[...,2,2,:]*s**2*(s - 1) - 3*params[...,7,2,:]*s**2*(s - 1) + 3*params[...,8,2,:]*s*(s - 1)**2)*(params[...,0,2,:]*(t - 1)**3 - params[...,3,2,:]*(t - 1)**3 + t*(3*params[...,8,2,:]*(s - 1)**2 - 3*params[...,9,2,:]*(s - 1)**2 + 3*params[...,6,2,:]*s**2 - 3*params[...,7,2,:]*s**2 - 6*params[...,7,2,:]*s*(s - 1) + 3*params[...,8,2,:]*s*(2*s - 2)) - params[...,6,2,:]*t + params[...,9,2,:]*t + (t - 1)*(3*params[...,0,2,:]*(s - 1)**2 - 3*params[...,1,2,:]*(s - 1)**2 + 3*params[...,2,2,:]*s**2 - 3*params[...,3,2,:]*s**2 + 6*params[...,2,2,:]*s*(s - 1) - 3*params[...,1,2,:]*s*(2*s - 2)) - params[...,0,2,:]*(t - 1) + params[...,3,2,:]*(t - 1) + params[...,6,2,:]*t**3 - params[...,9,2,:]*t**3 + 3*params[...,4,2,:]*t*(t - 1)**2 - 3*params[...,5,2,:]*t**2*(t - 1) + 3*params[...,10,2,:]*t**2*(t - 1) - 3*params[...,11,2,:]*t*(t - 1)**2))**2
    det = det.clamp_min_(1e-10).sqrt_()
    return det

def coons_normals(s, t, params):
    params = params[...,None]
    normals = th.stack([(params[...,0,2,:]*(s - 1)**3 - params[...,9,2,:]*(s - 1)**3 - s*(3*params[...,3,2,:]*(t - 1)**2 - 3*params[...,4,2,:]*(t - 1)**2 + 3*params[...,5,2,:]*t**2 - 3*params[...,6,2,:]*t**2 + 6*params[...,5,2,:]*t*(t - 1) - 3*params[...,4,2,:]*t*(2*t - 2)) + params[...,3,2,:]*s - params[...,6,2,:]*s + (s - 1)*(3*params[...,0,2,:]*(t - 1)**2 - 3*params[...,11,2,:]*(t - 1)**2 - 3*params[...,9,2,:]*t**2 + 3*params[...,10,2,:]*t**2 + 6*params[...,10,2,:]*t*(t - 1) - 3*params[...,11,2,:]*t*(2*t - 2)) - params[...,0,2,:]*(s - 1) + params[...,9,2,:]*(s - 1) - params[...,3,2,:]*s**3 + params[...,6,2,:]*s**3 - 3*params[...,1,2,:]*s*(s - 1)**2 + 3*params[...,2,2,:]*s**2*(s - 1) - 3*params[...,7,2,:]*s**2*(s - 1) + 3*params[...,8,2,:]*s*(s - 1)**2)*(params[...,0,1,:]*(t - 1)**3 - params[...,3,1,:]*(t - 1)**3 + t*(3*params[...,8,1,:]*(s - 1)**2 - 3*params[...,9,1,:]*(s - 1)**2 + 3*params[...,6,1,:]*s**2 - 3*params[...,7,1,:]*s**2 - 6*params[...,7,1,:]*s*(s - 1) + 3*params[...,8,1,:]*s*(2*s - 2)) - params[...,6,1,:]*t + params[...,9,1,:]*t + (t - 1)*(3*params[...,0,1,:]*(s - 1)**2 - 3*params[...,1,1,:]*(s - 1)**2 + 3*params[...,2,1,:]*s**2 - 3*params[...,3,1,:]*s**2 + 6*params[...,2,1,:]*s*(s - 1) - 3*params[...,1,1,:]*s*(2*s - 2)) - params[...,0,1,:]*(t - 1) + params[...,3,1,:]*(t - 1) + params[...,6,1,:]*t**3 - params[...,9,1,:]*t**3 + 3*params[...,4,1,:]*t*(t - 1)**2 - 3*params[...,5,1,:]*t**2*(t - 1) + 3*params[...,10,1,:]*t**2*(t - 1) - 3*params[...,11,1,:]*t*(t - 1)**2) - (params[...,0,1,:]*(s - 1)**3 - params[...,9,1,:]*(s - 1)**3 - s*(3*params[...,3,1,:]*(t - 1)**2 - 3*params[...,4,1,:]*(t - 1)**2 + 3*params[...,5,1,:]*t**2 - 3*params[...,6,1,:]*t**2 + 6*params[...,5,1,:]*t*(t - 1) - 3*params[...,4,1,:]*t*(2*t - 2)) + params[...,3,1,:]*s - params[...,6,1,:]*s + (s - 1)*(3*params[...,0,1,:]*(t - 1)**2 - 3*params[...,11,1,:]*(t - 1)**2 - 3*params[...,9,1,:]*t**2 + 3*params[...,10,1,:]*t**2 + 6*params[...,10,1,:]*t*(t - 1) - 3*params[...,11,1,:]*t*(2*t - 2)) - params[...,0,1,:]*(s - 1) + params[...,9,1,:]*(s - 1) - params[...,3,1,:]*s**3 + params[...,6,1,:]*s**3 - 3*params[...,1,1,:]*s*(s - 1)**2 + 3*params[...,2,1,:]*s**2*(s - 1) - 3*params[...,7,1,:]*s**2*(s - 1) + 3*params[...,8,1,:]*s*(s - 1)**2)*(params[...,0,2,:]*(t - 1)**3 - params[...,3,2,:]*(t - 1)**3 + t*(3*params[...,8,2,:]*(s - 1)**2 - 3*params[...,9,2,:]*(s - 1)**2 + 3*params[...,6,2,:]*s**2 - 3*params[...,7,2,:]*s**2 - 6*params[...,7,2,:]*s*(s - 1) + 3*params[...,8,2,:]*s*(2*s - 2)) - params[...,6,2,:]*t + params[...,9,2,:]*t + (t - 1)*(3*params[...,0,2,:]*(s - 1)**2 - 3*params[...,1,2,:]*(s - 1)**2 + 3*params[...,2,2,:]*s**2 - 3*params[...,3,2,:]*s**2 + 6*params[...,2,2,:]*s*(s - 1) - 3*params[...,1,2,:]*s*(2*s - 2)) - params[...,0,2,:]*(t - 1) + params[...,3,2,:]*(t - 1) + params[...,6,2,:]*t**3 - params[...,9,2,:]*t**3 + 3*params[...,4,2,:]*t*(t - 1)**2 - 3*params[...,5,2,:]*t**2*(t - 1) + 3*params[...,10,2,:]*t**2*(t - 1) - 3*params[...,11,2,:]*t*(t - 1)**2), (params[...,0,0,:]*(s - 1)**3 - params[...,9,0,:]*(s - 1)**3 - s*(3*params[...,3,0,:]*(t - 1)**2 - 3*params[...,4,0,:]*(t - 1)**2 + 3*params[...,5,0,:]*t**2 - 3*params[...,6,0,:]*t**2 + 6*params[...,5,0,:]*t*(t - 1) - 3*params[...,4,0,:]*t*(2*t - 2)) + params[...,3,0,:]*s - params[...,6,0,:]*s + (s - 1)*(3*params[...,0,0,:]*(t - 1)**2 - 3*params[...,11,0,:]*(t - 1)**2 - 3*params[...,9,0,:]*t**2 + 3*params[...,10,0,:]*t**2 + 6*params[...,10,0,:]*t*(t - 1) - 3*params[...,11,0,:]*t*(2*t - 2)) - params[...,0,0,:]*(s - 1) + params[...,9,0,:]*(s - 1) - params[...,3,0,:]*s**3 + params[...,6,0,:]*s**3 - 3*params[...,1,0,:]*s*(s - 1)**2 + 3*params[...,2,0,:]*s**2*(s - 1) - 3*params[...,7,0,:]*s**2*(s - 1) + 3*params[...,8,0,:]*s*(s - 1)**2)*(params[...,0,2,:]*(t - 1)**3 - params[...,3,2,:]*(t - 1)**3 + t*(3*params[...,8,2,:]*(s - 1)**2 - 3*params[...,9,2,:]*(s - 1)**2 + 3*params[...,6,2,:]*s**2 - 3*params[...,7,2,:]*s**2 - 6*params[...,7,2,:]*s*(s - 1) + 3*params[...,8,2,:]*s*(2*s - 2)) - params[...,6,2,:]*t + params[...,9,2,:]*t + (t - 1)*(3*params[...,0,2,:]*(s - 1)**2 - 3*params[...,1,2,:]*(s - 1)**2 + 3*params[...,2,2,:]*s**2 - 3*params[...,3,2,:]*s**2 + 6*params[...,2,2,:]*s*(s - 1) - 3*params[...,1,2,:]*s*(2*s - 2)) - params[...,0,2,:]*(t - 1) + params[...,3,2,:]*(t - 1) + params[...,6,2,:]*t**3 - params[...,9,2,:]*t**3 + 3*params[...,4,2,:]*t*(t - 1)**2 - 3*params[...,5,2,:]*t**2*(t - 1) + 3*params[...,10,2,:]*t**2*(t - 1) - 3*params[...,11,2,:]*t*(t - 1)**2) - (params[...,0,2,:]*(s - 1)**3 - params[...,9,2,:]*(s - 1)**3 - s*(3*params[...,3,2,:]*(t - 1)**2 - 3*params[...,4,2,:]*(t - 1)**2 + 3*params[...,5,2,:]*t**2 - 3*params[...,6,2,:]*t**2 + 6*params[...,5,2,:]*t*(t - 1) - 3*params[...,4,2,:]*t*(2*t - 2)) + params[...,3,2,:]*s - params[...,6,2,:]*s + (s - 1)*(3*params[...,0,2,:]*(t - 1)**2 - 3*params[...,11,2,:]*(t - 1)**2 - 3*params[...,9,2,:]*t**2 + 3*params[...,10,2,:]*t**2 + 6*params[...,10,2,:]*t*(t - 1) - 3*params[...,11,2,:]*t*(2*t - 2)) - params[...,0,2,:]*(s - 1) + params[...,9,2,:]*(s - 1) - params[...,3,2,:]*s**3 + params[...,6,2,:]*s**3 - 3*params[...,1,2,:]*s*(s - 1)**2 + 3*params[...,2,2,:]*s**2*(s - 1) - 3*params[...,7,2,:]*s**2*(s - 1) + 3*params[...,8,2,:]*s*(s - 1)**2)*(params[...,0,0,:]*(t - 1)**3 - params[...,3,0,:]*(t - 1)**3 + t*(3*params[...,8,0,:]*(s - 1)**2 - 3*params[...,9,0,:]*(s - 1)**2 + 3*params[...,6,0,:]*s**2 - 3*params[...,7,0,:]*s**2 - 6*params[...,7,0,:]*s*(s - 1) + 3*params[...,8,0,:]*s*(2*s - 2)) - params[...,6,0,:]*t + params[...,9,0,:]*t + (t - 1)*(3*params[...,0,0,:]*(s - 1)**2 - 3*params[...,1,0,:]*(s - 1)**2 + 3*params[...,2,0,:]*s**2 - 3*params[...,3,0,:]*s**2 + 6*params[...,2,0,:]*s*(s - 1) - 3*params[...,1,0,:]*s*(2*s - 2)) - params[...,0,0,:]*(t - 1) + params[...,3,0,:]*(t - 1) + params[...,6,0,:]*t**3 - params[...,9,0,:]*t**3 + 3*params[...,4,0,:]*t*(t - 1)**2 - 3*params[...,5,0,:]*t**2*(t - 1) + 3*params[...,10,0,:]*t**2*(t - 1) - 3*params[...,11,0,:]*t*(t - 1)**2), (params[...,0,1,:]*(s - 1)**3 - params[...,9,1,:]*(s - 1)**3 - s*(3*params[...,3,1,:]*(t - 1)**2 - 3*params[...,4,1,:]*(t - 1)**2 + 3*params[...,5,1,:]*t**2 - 3*params[...,6,1,:]*t**2 + 6*params[...,5,1,:]*t*(t - 1) - 3*params[...,4,1,:]*t*(2*t - 2)) + params[...,3,1,:]*s - params[...,6,1,:]*s + (s - 1)*(3*params[...,0,1,:]*(t - 1)**2 - 3*params[...,11,1,:]*(t - 1)**2 - 3*params[...,9,1,:]*t**2 + 3*params[...,10,1,:]*t**2 + 6*params[...,10,1,:]*t*(t - 1) - 3*params[...,11,1,:]*t*(2*t - 2)) - params[...,0,1,:]*(s - 1) + params[...,9,1,:]*(s - 1) - params[...,3,1,:]*s**3 + params[...,6,1,:]*s**3 - 3*params[...,1,1,:]*s*(s - 1)**2 + 3*params[...,2,1,:]*s**2*(s - 1) - 3*params[...,7,1,:]*s**2*(s - 1) + 3*params[...,8,1,:]*s*(s - 1)**2)*(params[...,0,0,:]*(t - 1)**3 - params[...,3,0,:]*(t - 1)**3 + t*(3*params[...,8,0,:]*(s - 1)**2 - 3*params[...,9,0,:]*(s - 1)**2 + 3*params[...,6,0,:]*s**2 - 3*params[...,7,0,:]*s**2 - 6*params[...,7,0,:]*s*(s - 1) + 3*params[...,8,0,:]*s*(2*s - 2)) - params[...,6,0,:]*t + params[...,9,0,:]*t + (t - 1)*(3*params[...,0,0,:]*(s - 1)**2 - 3*params[...,1,0,:]*(s - 1)**2 + 3*params[...,2,0,:]*s**2 - 3*params[...,3,0,:]*s**2 + 6*params[...,2,0,:]*s*(s - 1) - 3*params[...,1,0,:]*s*(2*s - 2)) - params[...,0,0,:]*(t - 1) + params[...,3,0,:]*(t - 1) + params[...,6,0,:]*t**3 - params[...,9,0,:]*t**3 + 3*params[...,4,0,:]*t*(t - 1)**2 - 3*params[...,5,0,:]*t**2*(t - 1) + 3*params[...,10,0,:]*t**2*(t - 1) - 3*params[...,11,0,:]*t*(t - 1)**2) - (params[...,0,0,:]*(s - 1)**3 - params[...,9,0,:]*(s - 1)**3 - s*(3*params[...,3,0,:]*(t - 1)**2 - 3*params[...,4,0,:]*(t - 1)**2 + 3*params[...,5,0,:]*t**2 - 3*params[...,6,0,:]*t**2 + 6*params[...,5,0,:]*t*(t - 1) - 3*params[...,4,0,:]*t*(2*t - 2)) + params[...,3,0,:]*s - params[...,6,0,:]*s + (s - 1)*(3*params[...,0,0,:]*(t - 1)**2 - 3*params[...,11,0,:]*(t - 1)**2 - 3*params[...,9,0,:]*t**2 + 3*params[...,10,0,:]*t**2 + 6*params[...,10,0,:]*t*(t - 1) - 3*params[...,11,0,:]*t*(2*t - 2)) - params[...,0,0,:]*(s - 1) + params[...,9,0,:]*(s - 1) - params[...,3,0,:]*s**3 + params[...,6,0,:]*s**3 - 3*params[...,1,0,:]*s*(s - 1)**2 + 3*params[...,2,0,:]*s**2*(s - 1) - 3*params[...,7,0,:]*s**2*(s - 1) + 3*params[...,8,0,:]*s*(s - 1)**2)*(params[...,0,1,:]*(t - 1)**3 - params[...,3,1,:]*(t - 1)**3 + t*(3*params[...,8,1,:]*(s - 1)**2 - 3*params[...,9,1,:]*(s - 1)**2 + 3*params[...,6,1,:]*s**2 - 3*params[...,7,1,:]*s**2 - 6*params[...,7,1,:]*s*(s - 1) + 3*params[...,8,1,:]*s*(2*s - 2)) - params[...,6,1,:]*t + params[...,9,1,:]*t + (t - 1)*(3*params[...,0,1,:]*(s - 1)**2 - 3*params[...,1,1,:]*(s - 1)**2 + 3*params[...,2,1,:]*s**2 - 3*params[...,3,1,:]*s**2 + 6*params[...,2,1,:]*s*(s - 1) - 3*params[...,1,1,:]*s*(2*s - 2)) - params[...,0,1,:]*(t - 1) + params[...,3,1,:]*(t - 1) + params[...,6,1,:]*t**3 - params[...,9,1,:]*t**3 + 3*params[...,4,1,:]*t*(t - 1)**2 - 3*params[...,5,1,:]*t**2*(t - 1) + 3*params[...,10,1,:]*t**2*(t - 1) - 3*params[...,11,1,:]*t*(t - 1)**2)], dim=-1)
    return F.normalize(normals, p=2, dim=-1)