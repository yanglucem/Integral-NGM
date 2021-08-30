# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 09:49:25 2021

@author: Lu Yang

For paper "On unified framework for nonlinear grey system models: an integro-differential equation perspective"

Simulation 1: grey Verhulst model under fixed noise level
"""

import numpy as np
from scipy.integrate import odeint
from utils_verhulst import verhulst_im, verhulst_grey, par_hat_im, par_hat_grey

# main process
# set true parameters and initial condition, note, let x0 = y0
y0 = 0.4

par_y = np.array([1.2, -0.5])  
x0 = y0 = (1 - par_y[0]) / par_y[1]
par_x = np.array([par_y[0], 2 * par_y[1]])   


# number of replications
nrep = 500;                        

#fixed noise level
T = 4
noiselevel = 10
hset = np.array([0.4, 0.20, 0.08, 0.04])

# table for parameters
mape_pars = np.zeros((7, len(hset)))
mean_pars = np.zeros((7, len(hset)))
stds_pars = np.zeros((7, len(hset)))


# save data for boxplot
table_xim_train = np.zeros((500, 0))
table_xim_test = np.zeros((500, 0))
table_xgrey_train = np.zeros((500, 0))
table_xgrey_test = np.zeros((500, 0))

table_pars_im = np.zeros((4, 0))
table_pars_grey = np.zeros((3, 0))

'''main loop'''
for iter_h in range(len(hset)):
    
    h = hset[iter_h]
    nobs = int(T / h) + 1
    
    # generate noise free data
    t_train = np.linspace(0, T, nobs)
    datalen = len(t_train)
    num_test = 3
    t_test = np.linspace(T + h, T + h * num_test, num_test)
    tspan = np.append(t_train, t_test)
    
    x_y = odeint(verhulst_im, np.append(x0, y0), tspan, args = (par_x,), 
                 rtol = 1e-12, atol = 1e-12)
    x = x_y[0:datalen, 0]
    y = x_y[0:datalen, 1]
    y_grey = odeint(verhulst_grey, y0, t_train, args=(par_y,), rtol=1e-12, 
                    atol=1e-12)
    
    # save parameters, errors and fitting series
    pars_im = np.zeros((4, nrep))
    ape_pars_im = np.zeros((4, nrep))
    rmse_xim = np.zeros((2, nrep))
    
    pars_grey = np.zeros((3, nrep))
    ape_pars_grey = np.zeros((3, nrep))
    rmse_xgrey = np.zeros((2, nrep))
    
    # 500 replication
    for iter_nrep in range(nrep):
        
        # add noise to true data
        np.random.seed(iter_nrep)
        noisemap = [np.std(x) * noiselevel * 0.01]
        noise = np.hstack( [noisemap * np.random.randn(datalen, 1)] )
        xn = x + noise[:,0]
        
        # integral matching-based modelling, y0 = x0, needed to be estimated
        par_im, x0_im, y0_im = par_hat_im(xn, t_train, datalen)
        x_im = odeint(verhulst_im, np.append(x0_im, y0_im), tspan, 
                      args=(par_im,), rtol=1e-12, atol=1e-12)
        
        ape_par_im = abs( (par_x - par_im) / par_x ) * 100
        ape_x0_im = abs( (x0 - x0_im) / x0 ) * 100  
        mse_xim = (x_im[:,0] - x_y[:,0]) ** 2
        
        pars_im[:, iter_nrep] = np.insert(np.append(par_im, x0_im), 
                                          [0], nobs, axis = 0)
        ape_pars_im[:, iter_nrep] = np.insert(np.append(ape_par_im, ape_x0_im),
                                              [0], nobs, axis = 0)
        rmse_xim[:, iter_nrep] = np.array([np.sqrt(np.mean(mse_xim[0 : datalen])),
                                            np.sqrt(np.mean(mse_xim[datalen + 1 :
                                                                    datalen + num_test]))])
        
        # nonlinear grey modelling
        par_grey = par_hat_grey(xn, t_train, datalen)
        y_grey = odeint(verhulst_grey, xn[0], tspan, args=(par_grey,), 
                        rtol=1e-12, atol=1e-12)
        tlen = len(y_grey)
        x_grey = np.append(y_grey[0], (y_grey[1:tlen,0] - 
                                       y_grey[0:tlen-1,0]) / h)
        ape_par_grey = abs( (par_y - par_grey) / par_y ) * 100 
        mse_xgrey = (x_grey - x_y[:,0]) ** 2 
        
        pars_grey[:,iter_nrep] = np.insert(par_grey, [0], datalen, axis = 0)
        ape_pars_grey[:, iter_nrep] = np.insert(ape_par_grey,
                                                [0], datalen, axis = 0)
        rmse_xgrey[:, iter_nrep] = np.array([np.sqrt(np.mean(mse_xgrey[0 : datalen])),
                                              np.sqrt(np.mean(mse_xgrey[datalen + 1 : 
                                                                        datalen + num_test]))])
       
        
    # sample mean and standard derivation of estimated parameters   
    mape_pars[:, iter_h] = np.append(ape_pars_grey.mean(axis = 1),
                                                  ape_pars_im.mean(axis = 1)) 
    mean_pars[:, iter_h] = np.append(pars_grey.mean(axis = 1), 
                                                  pars_im.mean(axis = 1))   
    stds_pars[:, iter_h] = np.append(pars_grey.std(axis = 1), 
                                                  pars_im.std(axis = 1))
    
    
    '''save data for boxplot'''
    table_xim_train = np.concatenate((table_xim_train, 
                                      rmse_xim[0,:].T.reshape(nrep,1)), axis=1)
    table_xim_test = np.concatenate((table_xim_test, 
                                     rmse_xim[1,:].T.reshape(nrep,1)),axis=1)
    table_xgrey_train = np.concatenate((table_xgrey_train, 
                                        rmse_xgrey[0,:].T.reshape(nrep,1)),axis=1)
    table_xgrey_test = np.concatenate((table_xgrey_test, 
                                       rmse_xgrey[1,:].T.reshape(nrep,1)),axis=1)
    
    np.savetxt("Simu1_output/noise_xim_train.csv", table_xim_train, delimiter=',')    
    np.savetxt("Simu1_output/noise_xim_test.csv", table_xim_test, delimiter=',')
    np.savetxt("Simu1_output/noise_xgrey_train.csv", table_xgrey_train, delimiter=',')
    np.savetxt("Simu1_output/noise_xgrey_test.csv", table_xgrey_test, delimiter=',')
    
    
    '''save data for violin plot'''
    table_pars_im = np.concatenate((table_pars_im, pars_im), axis = 1)
    table_pars_grey = np.concatenate((table_pars_grey, pars_grey), axis = 1)
    
    np.savetxt("Simu1_output/noise_pars_im.csv", table_pars_im, delimiter=',')    
    np.savetxt("Simu1_output/noise_pars_test.csv", table_pars_grey, delimiter=',')