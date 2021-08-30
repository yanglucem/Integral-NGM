# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 13:40:09 2021

@author: yang

For paper "On unified framework for nonlinear grey system models: an integro-differential equation perspective"

Simulation 2: grey Lokta-Velterra model under fixed noise level
"""

import numpy as np
from scipy.integrate import solve_ivp
from utils_lotka import initial, lotka_int, lotka_grey, par_hat_im, par_hat_grey


'''set experimental conditions'''

# set true parameters and inital consition, note, x0 = y0
par_y = np.array([ 1.2, 0.3, -1, -0.4])
par_x = np.array([ par_y[0], par_y[1], par_y[2], par_y[3] ])
x0 = y0 = initial(par_y)

# set exerimental conditions, fixed data length
nrep = 500
T = 5
noiselevel = 4
num_test = 3
hset = np.array([0.25, 0.10, 0.05, 0.01])


# saving parameters for table 1
mape_pars = np.zeros((12, len(hset)))
mean_pars = np.zeros((12, len(hset)))
stds_pars = np.zeros((12, len(hset)))

# saving fitting and forecasting error
table_x1im_train = np.zeros((500, 0))
table_x1im_test = np.zeros((500, 0))
table_x1grey_train = np.zeros((500, 0))
table_x1grey_test = np.zeros((500, 0))

table_x2im_train = np.zeros((500, 0))
table_x2im_test = np.zeros((500, 0))
table_x2grey_train = np.zeros((500, 0))
table_x2grey_test = np.zeros((500, 0))

# saving parameters for violin plot
table_pars_im = np.zeros((7, 0))
table_pars_grey = np.zeros((5, 0))


'''main loop'''

for iter_h in range(len(hset)):
    
    # generate noise free data
    h = hset[iter_h]
    nobs = int(T / h) + 1
    Tspan = T + num_test * h
    
    t_train = np.linspace(0, T, nobs)
    datalen = len(t_train)
    t_test = np.linspace(T + h, T + h * num_test, num_test)
    tspan = np.append(t_train, t_test)
        
    x_y_sol = solve_ivp(lotka_int, [0, Tspan], np.append(x0, y0), args=(par_x,), 
                        dense_output=True, rtol=1e-12, atol=1e-12)
    x_y = x_y_sol.sol(tspan).T
    x = x_y[0:datalen,0:2]
    
    # save parameters, errors and fitting series
    pars_im = np.zeros((7, nrep))
    ape_pars_im = np.zeros((7, nrep))
    rmse_x1im = np.zeros((2, nrep))
    rmse_x2im = np.zeros((2, nrep))
    
    pars_grey = np.zeros((5, nrep))
    ape_pars_grey = np.zeros((5, nrep))
    rmse_x1grey = np.zeros((2, nrep))
    rmse_x2grey = np.zeros((2, nrep))
    
    '''500 replication'''
    for iter_nrep in range(nrep):
        
        # add noise to true data
        np.random.seed(iter_nrep)
        noisemap = [np.std(x) * noiselevel *  0.01] 
        noise = (np.stack([noisemap * 
                           np.random.randn(datalen, 2)])).squeeze(axis = 0)
        xn = x[0:datalen,:] + noise        
               
        # integral matching-based method
        par_im, x0_im, y0_im = par_hat_im(xn, t_train, datalen)
        x_im_sol = solve_ivp(lotka_int, [0, Tspan], np.append(x0_im, y0_im), 
                         args=(par_im,), dense_output=True)
        x_im = x_im_sol.sol(tspan).T
        ape_par_im = abs( (par_x -par_im ) / par_x ) * 100
        ape_x0_im = abs( (x0 - x0_im) / x0 ) * 100  
        mse_im = (x_y[:,0:2]  - x_im[:,0:2]) ** 2
        
        pars_im[:,iter_nrep] = np.insert(np.append(par_im, x0_im), 
                                         [0], nobs, axis = 0)
        ape_pars_im[:,iter_nrep] = np.insert(np.append(ape_par_im, ape_x0_im),
                                              [0], nobs, axis = 0)
        rmse_x1im[:,iter_nrep] =  np.array([np.sqrt(np.mean(mse_im[0:datalen,0])),
                                           np.sqrt(np.mean(mse_im[datalen + 1:
                                                                  datalen + 
                                                                  num_test,0]))])
        rmse_x2im[:,iter_nrep] =  np.array([np.sqrt(np.mean(mse_im[0:datalen,1])),
                                           np.sqrt(np.mean(mse_im[datalen + 1:
                                                                  datalen + 
                                                                  num_test,1]))])
            
        # grey modelling
        par_grey = par_hat_grey(xn, t_train, datalen)
        y_grey_sol = solve_ivp(lotka_grey, [0, Tspan],  x0, 
                         args=(par_grey,), dense_output=True)
        y_grey = y_grey_sol.sol(tspan).T
        tlen = len(y_grey)
        x_grey = np.concatenate([xn[0,:].reshape(1,2),
                                 (y_grey[1:tlen] - y_grey[0: tlen - 1]) / h],
                                axis = 0)
        ape_par_grey = abs( (par_y - par_grey) / par_y ) * 100 
        mse_xgrey = (x_y[:,0:2] - x_grey[:,0:2]) ** 2
        
        pars_grey[:,iter_nrep] = np.insert(par_grey, [0], nobs, axis = 0)
        ape_pars_grey[:,iter_nrep] = np.insert(ape_par_grey,
                                               [0], nobs, axis = 0)
        rmse_x1grey[:,iter_nrep] = np.array([np.sqrt(np.mean(mse_xgrey[0:datalen,0])),
                                           np.sqrt(np.mean(mse_xgrey[datalen + 1:
                                                                  datalen + 
                                                                  num_test,0]))])
        rmse_x2grey[:,iter_nrep] = np.array([np.sqrt(np.mean(mse_xgrey[0:datalen,1])),
                                           np.sqrt(np.mean(mse_xgrey[datalen + 1:
                                                                  datalen + 
                                                                  num_test,1]))])
    
    mape_pars[:,iter_h] = np.around(np.append(ape_pars_grey.mean(axis = 1),
                                             ape_pars_im.mean(axis = 1)), 4)
                                       
    mean_pars[:,iter_h] = np.around(np.append(pars_grey.mean(axis = 1),
                                               pars_im.mean(axis = 1)), 4)  
    stds_pars[:,iter_h] = np.around(np.append(pars_grey.std(axis = 1), 
                                               pars_im.std(axis = 1)), 4)  
    
    '''fitting and forecasting error for boxplot'''
    table_x1im_train = np.concatenate((table_x1im_train, 
                                      rmse_x1im[0, :].T.reshape(nrep,1)),  
                                      axis = 1)
    table_x1im_test = np.concatenate((table_x1im_test, 
                                      rmse_x1im[1, :].T.reshape(nrep, 1)),  
                                    axis = 1)
    
    table_x2im_train = np.concatenate((table_x2im_train, 
                                      rmse_x2im[0, :].T.reshape(nrep,1)),  
                                      axis = 1)
    table_x2im_test = np.concatenate((table_x2im_test, 
                                      rmse_x2im[1, :].T.reshape(nrep, 1)),  
                                    axis = 1)
    
    table_x1grey_train = np.concatenate((table_x1grey_train, 
                                        rmse_x1grey[0, :].T.reshape(nrep, 1)),  
                                        axis = 1)
    table_x1grey_test = np.concatenate((table_x1grey_test, 
                                        rmse_x1grey[1, :].T.reshape(nrep, 1)),  
                                      axis = 1)
    
    table_x2grey_train = np.concatenate((table_x2grey_train, 
                                        rmse_x1grey[0, :].T.reshape(nrep, 1)),  
                                        axis = 1)
    table_x2grey_test = np.concatenate((table_x2grey_test, 
                                        rmse_x1grey[1, :].T.reshape(nrep, 1)),  
                                      axis = 1)
    
    
    '''saving data for boxplot'''
    np.savetxt("Simu2_output/noise_x1im_train.csv", table_x1im_train, delimiter=',')    
    np.savetxt("Simu2_output/noise_x1im_test.csv", table_x1im_test, delimiter=',')
    np.savetxt("Simu2_output/noise_x1grey_train.csv", table_x1grey_train, delimiter=',')
    np.savetxt("Simu2_output/noise_x1grey_test.csv", table_x1grey_test, delimiter=',')
    
    np.savetxt("Simu2_output/noise_x2im_train.csv", table_x2im_train, delimiter=',')    
    np.savetxt("Simu2_output/noise_x2im_test.csv", table_x2im_test, delimiter=',')
    np.savetxt("Simu2_output/noise_x2grey_train.csv", table_x2grey_train, delimiter=',')
    np.savetxt("Simu2_output/noise_x2grey_test.csv", table_x2grey_test, delimiter=',')
    
    
    '''data for violin plot'''
    table_pars_im = np.concatenate((table_pars_im, pars_im), axis = 1)
    table_pars_grey = np.concatenate((table_pars_grey, pars_grey), axis = 1)
    
    np.savetxt("Simu2_output/datalen_pars_im.csv", table_x1im_train, delimiter=',')    
    np.savetxt("Simu2_output/datalen_pars_grey.csv", table_x1im_test, delimiter=',')
    