#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 18:35:48 2018

@author: fpasqua
"""
import numpy as np
from scipy.optimize import fsolve,root
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from numpy import pi,sqrt

#   Problem parameters
T_c = 2300      # Temperature in the combustion chamber (K)
P_c = 2e6     # Pressure in the combustion chamber (Pa)
P_amb = 101e3   # Ambient pressure (Pa)
T_amb = 293     # Ambient temperature (K)
gamma = 1.2336    # Ratio of Specific Heats Cp/Cv (Gamma)
W = 22580        # Molecular weight of gas (kg/kmol)
#width = .01      # Nozzle width (meters)
r_th = .0075     # Throat  height (meters) (or radius)

#   Method of Characteristics
num = 5      # Number of Characteristic lines
theta_i = 3.   # Initial step in theta (deg)
axis_radius = 0

# Parametri extra calcolati
dr = r_th/50
max_iter = 10000
R = 8314/W

#   Part A

#find where P becomes u
r = np.zeros(max_iter)
r[0] = r_th
A_star = np.pi*r_th**2  # Area alla strozzatura (star denota la strozzatura)
M = 1
dM1 = .1

Ae = np.zeros(max_iter)
A_ratio = np.zeros(max_iter)
Ma = np.zeros(max_iter)
P = np.zeros(max_iter)
Te = np.zeros(max_iter)
Tt = np.zeros(max_iter)
Ve = np.zeros(max_iter)
Vt = np.zeros(max_iter)
rhot = np.zeros(max_iter)
mdot = np.zeros(max_iter)
TT = np.zeros(max_iter)

for i in range(max_iter):
    r[i] = r[0] + i*dr
    Ae[i] = np.pi * r[i]**2
    A_Asq = np.square(Ae[i]/A_star)
    A_ratio[i] = np.sqrt(A_Asq)

    #Newton Rhapson on Eq. 5.20 - Anderson text
    #https://pastebin.com/ZCrR1w2A
    res = 1
    if i > 0 :
        M = Ma[i-1]

    while res > .001 :
        M2 = M + dM1
        funa1 = -A_Asq + (1/M**2)*((2/(gamma+1))*(1+(gamma-1)*M**2/2))**((gamma+1)/(gamma-1))
        funa2 = -A_Asq + (1/M2**2)*((2/(gamma+1))*(1+(gamma-1)*M2**2/2))**((gamma+1)/(gamma-1))
        dv_dm = (funa2-funa1)/dM1

        M = M - funa1/dv_dm
        res = np.abs(funa1)

    Ma[i] = M

    # Find Pressure
    P[i] = P_c*(1+(gamma-1)*Ma[i]**2/2)**(-gamma/(gamma-1))

    # Find thrust for each point
    Te[i] = T_c/(1+(gamma-1)*Ma[i]**2/2)
    Tt[i] = T_c/(1+(gamma-1)/2)
    Ve[i] = Ma[i]*np.sqrt(Te[i]*gamma*R)
    Vt[i] = np.sqrt(Tt[i]*gamma*R)
    rhot[i] = P[i]/(R*Te[i])
    mdot[i] = rhot[i]*Ve[i]*Ae[i]
    TT[i] = mdot[i]*Ve[i] + (P[i] - P_amb)*Ae[i]

    if P[i] < P_amb:
        #break
        #Calculate the pressure if shock wave exists at the exit planes
        P_exit = P[i]*(1+(gamma*2/(gamma+1))*(Ma[i]**2-1))

        if P_exit <= P_amb:
            P[i] = P_exit
            last_iter = i+1
            break
else:
    print("Thrust algorithm did not converge in max_iter={0}".format(max_iter))
    import sys
    sys.exit(1)
        
# Reshape
r = r[:last_iter]
Ae = Ae[:last_iter]
A_ratio = A_ratio[:last_iter]
Ma = Ma[:last_iter]
P = P[:last_iter]
Te = Te[:last_iter]
Tt = Tt[:last_iter]
Ve = Ve[:last_iter]
Vt = Vt[:last_iter]
rhot = rhot[:last_iter]
mdot = mdot[:last_iter]
TT = TT[:last_iter]

plt.figure(2)
plt.plot(r,TT)
plt.title('Thrust curve for throat radius {0}m'.format(r_th))
plt.xlabel('Exit Radius (m^2)')
plt.ylabel('Thrust (N)')

#   Part B  
#   Determine the nominal exit area of the nozzle 
#   to maximize thrust

b = np.argmax(TT)
#a = TT[b]
#   Over or Underexpand the nozzle
b = b #WTF?
A_max = Ae[b] 
Max_thrust = TT[b]
plt.plot(r[b],Max_thrust,'r*')
plt.legend(['Thrust Curve','Max Thrust'],loc='best')
plt.show()

#   Part C
#   Method of Characteristics

M_e = Ma[b]       #Mach number at ideal exit 

def prandtl_meyer(M):
    """Prandtl-Meyer function (output in radians)"""
    print("M: {0}".format(M))
    return (np.sqrt((gamma+1)/(gamma-1))*np.arctan(np.sqrt((gamma-1)*(M**2-1)/(gamma+1)))-np.arctan(np.sqrt(M**2-1)))

#Find theta_max by using equation 11.33
theta_max = np.rad2deg(prandtl_meyer(M_e))/2

#  D_theta for each char line
del_theta = (theta_max - theta_i)/(num-1)

def inv_prandtl_meyer(nu,M_0):
    """Metodo di Newton su funzione di Prandtl-Meyer per calcolo inversa (ritorna M(nu))"""
    #TODO: Vedere se provare a utilizzare Newton-Secante su prandtl-meyer da scipy al posto di questo fatto a mano
    dM = .1 #Lasciare a circa .1
    M = M_0
    res = 1
    while res > .01:
        M2 = M + dM
        #TODO: Vedere di utilizzare prandtl_meyer(M)
        funv1 = (-np.deg2rad(nu)+(np.sqrt((gamma+1)/(gamma-1))*np.arctan((np.sqrt((gamma-1)*(M**2-1)/(gamma+1))))-np.arctan(np.sqrt(M**2-1))))
        funv2 = (-np.deg2rad(nu)+(np.sqrt((gamma+1)/(gamma-1))*np.arctan((np.sqrt((gamma-1)*(M2**2-1)/(gamma+1))))-np.arctan(np.sqrt(M2**2-1))))
        dv_dm = (funv2-funv1)/dM
        M = M - funv1/dv_dm
        res = np.abs(funv1)
    return M
    

# Calcolo di points_a, matrice dei primi punti di ogni linea caratteristica alla strozzatura
# Per ogni caratteristica ho num (ci sono ripetizioni tra gli unici) punti totali
# points[numero_linea_carat]['variabile presa in considerazione'][numero del punto nella linea]
char_lines = [pd.DataFrame(np.zeros((i,6)), columns=['x','r','theta','nu','M','mu']) for i in reversed(range(2,num+2))]

# Calcolo dei punti iniziali (Punto A per ogni caratteristica)
for i,chln in enumerate(char_lines): #for i in range(num):
    chln['x'][0] = 0
    chln['r'][0] = r_th
    chln['theta'][0] = theta_i + i*del_theta
    chln['nu'][0] = chln['theta'][0]
    chln['M'][0] = inv_prandtl_meyer(np.deg2rad(chln['nu'][0]), 1) #PRALDT-MEYER INVERSO DA NU
    chln['mu'][0] = np.rad2deg(np.arcsin(1/chln['M'][0]))
    
def tand(x):
    return np.tan(np.deg2rad(x))

def cot(x):
    return np.reciprocal(np.tan(x))

def cotd(x):
    return np.reciprocal(tand(x))

def intersect_algorithm(high_point_data, low_point_data):
    """Argomenti: high_point_data= C+ data, low_point_data= C- data"""
    # Estrazione dati da tabelle
    x_1,r_1,theta_1,nu_1,M_1,mu_1 = high_point_data
    x_2,r_2,theta_2,nu_2,M_2,mu_2 = low_point_data
    # Calcolo di x_3 e r_3 con metodo esplicito (eq. 5-6)
    x_3 = (x_1*tand(theta_1-mu_1) - x_2*tand(theta_2+mu_2) - r_1 + r_2)/(tand(theta_1-mu_1) - tand(theta_2+mu_2))
    r_3 = 0.5*( (x_3-x_1)*tand(theta_1-mu_1) + (x_3-x_2)*tand(theta_2+mu_2) + r_1 + r_2 )
    # Stimo esplicitamente theta_3 e nu_3 (tilde entrambi) con la media dei rispettivi 1 e 2
    theta_3 = 0.5*(theta_1+theta_2)
    nu_3 = 0.5*(nu_1+nu_2)
    # Stimo M_3 con inv_prandtl_meyer
    M_3 = inv_prandtl_meyer(np.deg2rad(nu_3),0.5*(M_1+M_2))
    # Iterazione implicita su theta_3 e M_3 con fissato r_3
    def num_imp_integration(data,r_1,r_2,theta_1,theta_2,nu_1,nu_2,r_3):
        """data = (theta_3, M_3)"""
        theta_3, M_3 = data
        print("DEBUG: {0} {1}".format(theta_3, M_3))
        nu_3 = np.rad2deg(prandtl_meyer(M_3))
        # Eq. 7
        first = (r_3-r_1)/(np.sqrt(M_3**2-1)-cotd(theta_3))*1/r_3 - np.deg2rad(theta_3) - np.deg2rad(nu_3) + np.deg2rad(theta_1) + np.deg2rad(nu_1)
        # Eq. 8
        second = (r_2-r_3)/(np.sqrt(M_3**2-1)+cotd(theta_3))*1/r_3 - np.deg2rad(theta_3) + np.deg2rad(nu_3) + np.deg2rad(theta_2) - np.deg2rad(nu_2)
        return (first, second)
    def num_imp_int_prime(data,r_1,r_2,theta_1,theta_2,nu_1,nu_2,r_3):
        theta_3, M_3 = data
        j11 =  -1/180*pi + 1/180*pi*(cot(1/180*pi*theta_3)**2 + 1)*(r_1 - r_3)/(r_3*(sqrt(M_3**2 - 1) - cot(1/180*pi*theta_3))**2) 
        j12 = -2*M_3*(gamma - 1)*sqrt((gamma + 1)/(gamma - 1))/(((M_3**2 - 1)**2*(gamma - 1)**2/(gamma + 1)**2 + 1)*(gamma + 1)) +  1/(sqrt(M_3**2 - 1)*M_3) + M_3*(r_1 - r_3)/(sqrt(M_3**2 - 1)*r_3*(sqrt(M_3**2 - 1) - cot(1/180*pi*theta_3))**2)
        j21 =  -1/180*pi + 1/180*pi*(cot(1/180*pi*theta_3)**2 + 1)*(r_2 - r_3)/(r_3*(sqrt(M_3**2 - 1) + cot(1/180*pi*theta_3))**2)  
        j22 = 2*M_3*(gamma - 1)*sqrt((gamma + 1)/(gamma - 1))/(((M_3**2 - 1)**2*(gamma - 1)**2/(gamma + 1)**2 + 1)*(gamma + 1)) - 1/(sqrt(M_3**2 - 1)*M_3) - M_3*(r_2 - r_3)/(sqrt(M_3**2 - 1)*r_3*(sqrt(M_3**2 - 1) + cot(1/180*pi*theta_3))**2)
        jacobian = np.array([[j11,j12],[j21,j22]])
        return jacobian
# =============================================================================
#         [                                                                       
# -1/180*pi + 1/180*pi*(cot(1/180*pi*th3)^2 + 1)*(r1 - r3)/(r3*(sqrt(M3^2
# - 1) - cot(1/180*pi*th3))^2) -2*M3*(gam - 1)*sqrt((gam + 1)/(gam -
# 1))/(((M3^2 - 1)^2*(gam - 1)^2/(gam + 1)^2 + 1)*(gam + 1)) +
# 1/(sqrt(M3^2 - 1)*M3) + M3*(r1 - r3)/(sqrt(M3^2 - 1)*r3*(sqrt(M3^2 - 1)
# - cot(1/180*pi*th3))^2)]
# [                                                                       
# -1/180*pi + 1/180*pi*(cot(1/180*pi*th3)^2 + 1)*(r2 - r3)/(r3*(sqrt(M3^2
# - 1) + cot(1/180*pi*th3))^2)  2*M3*(gam - 1)*sqrt((gam + 1)/(gam -
# 1))/(((M3^2 - 1)^2*(gam - 1)^2/(gam + 1)^2 + 1)*(gam + 1)) -
# 1/(sqrt(M3^2 - 1)*M3) - M3*(r2 - r3)/(sqrt(M3^2 - 1)*r3*(sqrt(M3^2 - 1)
# + cot(1/180*pi*th3))^2)]
# =============================================================================
    (theta_3, M_3), infodict, ier, msg = fsolve(num_imp_integration,(theta_3, M_3), fprime=num_imp_int_prime, args=(r_1,r_2,theta_1,theta_2,nu_1,nu_2,r_3), maxfev=1000, factor=1, epsfcn=1e-10, full_output=True)
    if ier != 1: # Test di riuscita dell'algoritmo di convergenza
        print("Convergenza fallita sulla prima integrazione: %s"%msg)
        print(infodict)
    mu_3 = np.rad2deg(np.arcsin(1/M_3))
    def pos_imp_integration(data,x_1,x_2,r_1,r_2,theta_3,mu_3):
        """data = (x_3, r_3)"""
        x_3, r_3 = data
        first = (x_3-x_1)*tand(theta_3-mu_3)-r_3+r_1
        second = (x_3-x_2)*tand(theta_3+mu_3)-r_3+r_2
        return (first, second)
    (x_3, r_3), _, ier, msg = fsolve(pos_imp_integration,(theta_3, M_3), args=(x_1,x_2,r_1,r_2,theta_3,mu_3), maxfev=1000, factor=1, epsfcn=1e-10, full_output=True)
    if ier != 1: # Test di riuscita dell'algoritmo di convergenza
        print("Convergenza fallita sulla seconda integrazione: %s"%msg)
    # Di nuovo 2 e 3
    # Stimo esplicitamente theta_3 e nu_3 (tilde entrambi) con la media dei rispettivi 1 e 2
    theta_3 = 0.5*(theta_1+theta_2)
    nu_3 = 0.5*(nu_1+nu_2)
    # Stimo M_3 con inv_prandtl_meyer
    M_3 = inv_prandtl_meyer(np.deg2rad(nu_3),0.5*(M_1+M_2))
    (theta_3, M_3), _, ier, msg = fsolve(num_imp_integration,(theta_3, M_3), args=(r_1,r_2,theta_1,theta_2,nu_1,nu_2,r_3), maxfev=1000, factor=1, epsfcn=1e-10, full_output=True)
    if ier != 1: # Test di riuscita dell'algoritmo di convergenza
        print("Convergenza fallita sulla terza integrazione: %s"%msg)DIOCANE
    mu_3 = np.rad2deg(np.arcsin(1/M_3))
    # Calcolo i valori restanti
    nu_3 = np.rad2deg(prandtl_meyer(M_3))
    return (x_3,r_3,theta_3,nu_3,M_3,mu_3)

def axis_algorithm(pre_axis_point_data):
    # Estrazione dati da tabelle
    x_1,r_1,theta_1,nu_1,M_1,mu_1 = pre_axis_point_data
    # Calcolo di x_3 con metodo esplicito (eq. 9)
    x_3 = x_1 - r_1/tand(theta_1-mu_1)
    # Per definizione, raggio_3 e theta_3 sull'asse nullo
    r_3 = 0.
    theta_3 = 0.
    # Calcolo nu_3
    nu_3 = theta_1 + nu_1
    # Calcolo i valori restanti
    M_3 = inv_prandtl_meyer(np.deg2rad(nu_3),M_1)
    mu_3 = np.rad2deg(np.arcsin(1/M_3))
    return (x_3,r_3,theta_3,nu_3,M_3,mu_3)

def wall_algorithm(prec_wall_point_data, normal_point_data):
    # Estrazione dati da tabelle
    x_1,r_1,theta_1,nu_1,M_1,mu_1 = prec_wall_point_data
    x_2,r_2,theta_2,nu_2,M_2,mu_2 = normal_point_data
    # Per ipotesi di lavoro, theta_3 = theta_2
    theta_3 = deepcopy(theta_2)
    # Calcolo x_3 con metodo esplicito (eq. 10)
    x_3 = (x_2*tand(theta_2+mu_2) - x_1*tand(0.5*(theta_1+theta_2)) + r_1 - r_2)/(tand(theta_2+mu_2) + tand(0.5*(theta_1+theta_2)))
    # Calcolo r_3 con metodo esplicito (eq. 11)
    r_3 = r_2 + tand(theta_2+mu_2)*(x_3-x_2)
    return (x_3,r_3,theta_3)

# Inizio algoritmo di prova (ovvero senza linea iniziale)
def get_point_data(n_line, n_point):
    return tuple(char_lines[n_line].loc[n_point,:])
def set_point_data(n_line, n_point, data_tuple):
    char_lines[n_line].loc[n_point,:] = data_tuple

for n_line in range(num):
    n_punti_linea = (num+1)-n_line
    for n_point in range(1,n_punti_linea+1):
        #try:
        if n_point == 1:
            if n_line == 0:
                #AXIS speciale con p_1 = (0,0)
                set_point_data(n_line, n_point, axis_algorithm(get_point_data(0,0)))
            else:
                #AXIS normale con (i-1,2)
                set_point_data(n_line, n_point, axis_algorithm(get_point_data(n_line-1,2)))
        else:
            if n_line == 0:
                #INTERSECT C+ = (0,0) C- = (0,n_point-1)
                set_point_data(n_line, n_point, intersect_algorithm(get_point_data(n_line+1,0),get_point_data(0,n_point-1)))
            else:
                #INTERSECT C+ = (n_line-1,n_point+1) C- = (n_line,n_point-1)
                set_point_data(n_line, n_point, intersect_algorithm(get_point_data(n_line-1,n_point+1), get_point_data(n_line,n_point-1)))
        #except:
        #    raise Exception("Convergenza fallita sul punto ({0},{1})".format(n_line,n_point))
        
# Creazione Wall ugello

# Plot profilo delle rette caratteristiche
x = []
r = []
for i,chln in enumerate(char_lines):
    x.append(list(chln.loc[:,'x']))
    r.append(list(chln.loc[:,'r']))
for i, (x_list,r_list) in enumerate(zip(x,r)):
    plt.plot(x_list,r_list,label='%i'%i)
plt.legend(loc='best')
plt.show()
        
