#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 15:42:35 2018

@author: fpasqua
"""

import numpy as np
from numpy import tan,sqrt,reciprocal
import matplotlib.pyplot as plt

import warnings
#warnings.simplefilter('error')

gamma = 1.2336    # Ratio of Specific Heats Cp/Cv (Gamma)

def cot(x):
    return np.reciprocal(np.tan(x))

def prandtl_meyer(M):
    """Prandtl-Meyer function (output in radians)"""
    return (np.sqrt((gamma+1)/(gamma-1))*np.arctan(np.sqrt((gamma-1)*(M**2-1)/(gamma+1)))-np.arctan(np.sqrt(M**2-1)))

def inv_prandtl_meyer(nu,M_0):
    """Metodo di Newton su funzione di Prandtl-Meyer per calcolo inversa (ritorna M(nu))"""
    #TODO: Vedere se provare a utilizzare Newton-Secante su prandtl-meyer da scipy al posto di questo fatto a mano
    dM = .1 #Lasciare a circa .1
    M = M_0
    res = 1
    while res > .01:
        M2 = M + dM
        #TODO: Vedere di utilizzare prandtl_meyer(M)
        funv1 = (-nu+(np.sqrt((gamma+1)/(gamma-1))*np.arctan((np.sqrt((gamma-1)*(M**2-1)/(gamma+1))))-np.arctan(np.sqrt(M**2-1))))
        funv2 = (-nu+(np.sqrt((gamma+1)/(gamma-1))*np.arctan((np.sqrt((gamma-1)*(M2**2-1)/(gamma+1))))-np.arctan(np.sqrt(M2**2-1))))
        dv_dm = (funv2-funv1)/dM
        M = M - funv1/dv_dm
        res = np.abs(funv1)
    return M

N_PUNTI = 11

x = np.zeros((N_PUNTI,N_PUNTI))  #['x','r','theta','nu','M','mu'])
r = np.zeros((N_PUNTI,N_PUNTI))
theta = np.zeros((N_PUNTI,N_PUNTI))
nu = np.zeros((N_PUNTI,N_PUNTI))
M = np.zeros((N_PUNTI,N_PUNTI))
mu = np.zeros((N_PUNTI,N_PUNTI))

x[0,:] = [0]*N_PUNTI
r[0,:] = np.linspace(2,1,num=N_PUNTI) #non testiamo ancora l'asse
theta[0,:] = np.linspace(np.deg2rad(20),np.deg2rad(5),num=N_PUNTI)
M[0,:] = [1.01]*N_PUNTI#np.linspace(2,3,num=N_PUNTI)#[2]*N_PUNTI
nu[0,:] = prandtl_meyer(M[0,:])
mu[0,:] = np.arcsin(np.reciprocal(M[0,:]))

for gen in range(1,N_PUNTI):
    for point in range(N_PUNTI-gen):
        plus = np.s_[gen-1,point+1]
        minus = np.s_[gen-1,point]
        x_p = x[plus]
        x_m = x[minus]
        r_p = r[plus]
        r_m = r[minus]
        theta_p = theta[plus]
        theta_m = theta[minus]
        nu_p = nu[plus]
        nu_m = nu[minus]
        M_p = M[plus]
        M_m = M[minus]
        mu_p = mu[plus]
        mu_m = mu[minus]
        
        x_i = (r_p - r_m + tan(theta_m - mu_m)*x_m - tan(theta_p + mu_p)*x_p)/(tan(theta_m - mu_m) - tan(theta_p + mu_p))
        r_i = 0.5*( r_p + r_m + tan(theta_m - mu_m)*(x_i - x_m) + tan(theta_p + mu_p)*(x_i - x_p) )
        psi_p = theta_m + nu_m + 1/(sqrt(M_m**2 - 1) - cot(theta_m)) * (r_i - r_m)/r_m
        psi_m = theta_p - nu_p - 1/(sqrt(M_p**2 - 1) + cot(theta_p)) * (r_i - r_p)/r_p
        theta_i = 0.5 * (psi_p + psi_m)
        nu_i = 0.5 * (psi_p - psi_m)
        M_i = inv_prandtl_meyer(nu_i, 0.5*(M_p+M_m))
        mu_i = np.arcsin(1/M_i)
        
        counter = 0
        res_x = res_r = res_theta = res_nu = 1
        while not (res_x < 0.05 and res_r < 0.1 and res_theta < 0.02 and res_nu < 0.02):
            r_cp = (r_p + r_i)/2
            r_cm = (r_m + r_i)/2
            theta_cp = (theta_p + theta_i)/2
            theta_cm = (theta_m + theta_i)/2
            nu_cp = (nu_p + nu_i)/2
            nu_cm = (nu_m + nu_i)/2
            M_cp = inv_prandtl_meyer(nu_cp, (M_p + M_i)/2 )
            M_cm = inv_prandtl_meyer(nu_cm, (M_m + M_i)/2 )
            mu_cp = (mu_p + mu_i)/2
            mu_cm = (mu_m + mu_i)/2
            
            x_o,r_o,theta_o,nu_o = x_i,r_i,theta_i,nu_i
            
            x_i = (r_p - r_m + tan(theta_cm - mu_cm)*x_m - tan(theta_cp + mu_cp)*x_p)/(tan(theta_cm - mu_cm) - tan(theta_cp + mu_cp))
            r_i = 0.5*( r_p + r_m + tan(theta_cm - mu_cm)*(x_i - x_m) + tan(theta_cp + mu_cp)*(x_i - x_p) )
            psi_p = theta_m + nu_m + 1/(sqrt(M_cm**2 - 1) - cot(theta_cm)) * (r_i - r_m)/r_cm
            psi_m = theta_p - nu_p - 1/(sqrt(M_cp**2 - 1) + cot(theta_cp)) * (r_i - r_p)/r_cp
            theta_i = 0.5 * (psi_p + psi_m)
            nu_i = 0.5 * (psi_p - psi_m)
            M_i = inv_prandtl_meyer(nu_i, 0.5*(M_cp+M_cm))
            mu_i = np.arcsin(1/M_i)
            
            res_x = np.nan_to_num(np.abs((x_o - x_i)/x_i))
            res_r = np.nan_to_num(np.abs((r_o - r_i)/r_i))
            res_theta = np.nan_to_num(np.abs((theta_o - theta_i)/theta_i))
            res_nu = np.nan_to_num(np.abs((nu_o - nu_i)/nu_i))
            
            counter += 1
            print(f"m = {counter}\nres_x = {res_x} \tres_r = {res_r} \tres_theta = {res_theta} \tres_nu = {res_nu}")
            if counter > 10000:
                print("Convergence failed!")
                import sys
                sys.exit(1)
            
        now = np.s_[gen,point]
        x[now] = x_i
        r[now] = r_i
        theta[now] = theta_i
        nu[now] = nu_i
        M[now] = M_i
        mu[now] = mu_i
        
        plt.plot([x_p,x_i],[r_p,r_i],'g-')
        plt.plot([x_m,x_i],[r_m,r_i],'r-')
plt.show()