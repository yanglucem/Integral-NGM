# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:18:52 2021

@author: yl
"""
import numpy as np


# the integro-differential equation model
def lotka_int(t, y, p):  
    dy1 = p[0] * y[0] - p[1] * (y[0] * y[3] + y[1] * y[2])
    dy2 = p[2] * y[1] - p[3] * (y[0] * y[3] + y[1] * y[2])
    dy3 = y[0]
    dy4 = y[1]
    
    # notice y[0] is x1, y[1] is x2, y[2] is y1, y[3] is y2
    dx= [dy1,dy2,dy3,dy4]
    return dx


# nonlinear grey Lotka-Volterra model
def lotka_grey(t, y, p):
    dy1 = p[0] * y[0] - p[1] * y[0] * y[1]
    dy2 = p[2] * y[1] - p[3] * y[0] * y[1]
    
    dy = np.array([dy1, dy2])
    
    return dy
    

# assume x0 = y0, solving inital value
def initial(p):
    ini1 = (p[2] - 1) / p[3]
    ini2 = (p[0] - 1) / p[1]
    ini = np.array([ini1, ini2])
    
    return ini


# discretization by trapezoid rule     
def par_matching1(x, t, n):
    h = np.diff(t)
    z1 = (np.cumsum(h * x[1:n,0]) + np.cumsum(h * x[0:n-1,0])) / 2
    z2 = (np.cumsum(h * x[1:n,1]) + np.cumsum(h * x[0:n-1,1])) / 2
    
    B = np.array([z1,   # a1
                  #-2*(np.multiply(z1+x[0,0],z1+x[0,0])-np.square(x[0,0])),
                  -(np.multiply(z1 + x[0,0], z2 + x[0,1]) - x[0,0] * x[0,1]),    #c1
                  np.ones(n-1)]).T       # b1 : 
    D = np.array([z2,
                  #-2*(np.multiply(z2+x[0,1],z2+x[0,1])-np.square(x[0,1])),
                  -(np.multiply(z1 + x[0,0], z2 + x[0,1]) - x[0,0] * x[0,1]),
                  np.ones(n-1)]).T     #  b2 : 
    
    a1,c1,x01 = np.linalg.pinv(np.dot(B.T,B)).dot(B.T).dot(x[1:n,0])
    a2,c2,x02 = np.linalg.pinv(np.dot(D.T,D)).dot(D.T).dot(x[1:n,1])
    b1,b2 = [0,0]
    p = np.array([a1,b1,c1,a2,b2,c2])
    x0 = np.array([x01,x02])
    
    return p, x0


# assume y0 = x0
def par_hat_im(x, t, n):
    h = np.diff(t)
    
    z1 = (np.cumsum(h * x[1:n, 0]) + np.cumsum(h * x[0:n-1, 0])) / 2
    z2 = (np.cumsum(h * x[1:n, 1]) + np.cumsum(h * x[0:n-1, 1])) / 2
    
    B = np.array([z1, -z2,  - z1 * z2, np.ones(n-1)])  # - 0.5 * z1 ** 2,
    D = np.array([z2, -z1,  - z1 * z2, np.ones(n-1)])  # - 0.5 * z1 ** 2,
    
    A1, A2, a2,  x01 = np.linalg.pinv(np.dot(B, B.T)).dot(B).dot(x[1:n, 0])    
    B1, B2, b2,  x02 = np.linalg.pinv(np.dot(D, D.T)).dot(D).dot(x[1:n, 1])
    
    y01 = x01
    y02 = x02
    a1 = A1 + a2 * x02
    b1 = B1 + b2 * x01
    
    p = np.array([a1, a2, b1, b2])
    x0 = np.array([x01, x02])
    y0 = np.array([y01, y02])
    
    return p, x0, y0

   
# discretization by grey modelling     
def par_hat_grey(x, t, n):
    h = np.append(1,np.diff(t))
    y1 = np.cumsum(h * x[0:n,0])
    y2 = np.cumsum(h*x[0:n,1])
    
    z1 = (y1[0:n-1] + y1[1:n]) / 2
    z2 = (y2[0:n-1] + y2[1:n]) / 2
    
    B = np.array([z1,
                  -np.multiply(z1,z2)]).T
    D = np.array([z2, 
                  -np.multiply(z1,z2)]).T

    a1,a2 = np.linalg.pinv(np.dot(B.T,B)).dot(B.T).dot(x[1:n,0])              
    b1,b2 = np.linalg.pinv(np.dot(D.T,D)).dot(D.T).dot(x[1:n,1])
    p = np.array([a1, a2, b1, b2])
    
    # x01 = (b1 - 1) / b2
    # x02 = (a1 - 1) / a2
    # x0 = np.array([x01, x02])
     
    return p
    
    
    
    
    
    
    
    