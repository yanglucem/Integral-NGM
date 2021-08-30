# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:32:35 2021

@author: yl
"""

import numpy as np
from scipy.optimize import fsolve
from numpy.linalg import solve
import math
from functools import partial


def verhulst_ggrey(t,y,p):
    dy = p[0]*y + p[1]*y**2 + p[2]
    
    return np.array(dy)


def verhulst_grey(y,t,p):
    dy = p[0]*y + p[1]*y**2
    
    return np.array(dy)

# if c = 0, the model has a closed form solution
def verhulst_close(y0,p,t):
    y = np.array([(-p[1]/p[0] + math.exp(-p[0]*t[i]) * (1/y0 + p[1]/p[0]))**(-1) for i in range(len(t))])
    
    return y


def verhulst_im(y,t,p):
    dy1 = p[0]*y[0] + p[1]*y[0]*y[1]
    dy2 = y[0]  
    dy = np.array([dy1,dy2])
    
    return dy


def initial_y(x0,p):
    y01 = -(p[0] + (p[0]**2 + 4*p[1]*x0)**0.5)/(2*p[1])
    y02 = -(p[0] - (p[0]**2 + 4*p[1]*x0)**0.5)/(2*p[1])
    
    y0 = np.array([y01, y02])
    return y0
    

def initial_x(y0, p):
    x0 = p[0]*y0 + p[1]*y0**2 + p[2]
    
    return x0


# unknown y0
def par_hat_im(x, t, n):
    h = np.diff(t)
    z = (np.cumsum(h * x[1:n]) + np.cumsum(h * x[0:n-1])) / 2
    B = np.array([z, z**2, np.ones(n-1)]).T
    a1, a2, x0 = np.linalg.pinv(np.dot(B.T,B)).dot(B.T).dot(x[1:n])
    
    y0 = x0
    mu1 = a1 - 2 * a2 * y0
    mu2 = 2 * a2
    # def solve_y0(b):
    #     f1 = b[0] + a2 * b[1] - a
    #     f2 = b[0] * b[1] + a2 * b[1] ** 2 - x0
    #     return [f1, f2]    
    # a1, y0 = fsolve(solve_y0,np.array([0, 0]))  
   
    p = np.array([mu1, mu2])
    
    return p, x0, y0

# let y0 = x(t0), useless
def par_hat_im2(x, t, n):
    h = np.diff(t)
    z = (np.cumsum(h * x[1:n]) + np.cumsum(h * x[0:n-1]))/2
    B = np.array([z, 
                  (np.multiply(z + x[0], z + x[0]) - x[0] ** 2)/2, 
                  np.ones(n-1)]).T
    a1, a2, x0 = np.linalg.pinv(np.dot(B.T, B)).dot(B.T).dot(x[1:n])
    p = np.array([a1, a2])
    
    return p, x0
  

# grey discretization  
def par_hat_grey(x, t, n):
    h = np.diff(t)
    y = np.append(x[0], x[0] + np.cumsum(h * x[1:n]))
    z = 1/2 * ( y[0:n-1] + y[1:n] )
    B = np.array( [z, z**2] ).T
    a1, a2 = np.linalg.pinv(np.dot(B.T, B)).dot(B.T).dot(x[1:n])
    p = np.array([a1, a2])
    # x0 = (1 - a1) / a2
    
    return p



''' functions for comparison'''






