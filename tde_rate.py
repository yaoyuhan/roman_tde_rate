#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:02:53 2025

@author: yuhanyao
"""

import os
import time
import numpy as np
from copy import deepcopy
from scipy.stats import poisson

from scipy.integrate import simpson
from astropy.io import fits
import astropy.constants as const
from astropy.table import Table
from astropy import units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from astropy.time import Time

import matplotlib
import matplotlib.pyplot as plt
fs= 10
matplotlib.rcParams['font.size']=fs
ms = 6
matplotlib.rcParams['lines.markersize']=ms

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)


def compute_universe_size():
    zs = np.array([0.1, 0.5, 1,2,3,6,10])
    Vs_Gpc3 = np.zeros(len(zs))
    for i in range(len(zs)):
        z = zs[i]
        D_Mpc = cosmo.comoving_distance([z])[0].value 
        V_Mpc3 = 4*np.pi / 3 * D_Mpc**3
        Vs_Gpc3[i] = V_Mpc3 / 1e+9
    
    plt.plot(zs, Vs_Gpc3, "o")
    plt.semilogy()
    plt.semilogx()
    plt.xlabel("z")
    plt.ylabel("Comoving volume (Gpc^3)")
    
    
def get_gmf_z(z = 4, lgxs = np.linspace(7, 11.5)):
    
    # get galaxy stellar mass function 
    xs = 10**lgxs
    
    if z==0:
        # Wright+2017
        lgM0 = 10.84
        lgphi1 = -4.30
        alpha1 = -0.
        lgphi2 = -3.94
        alpha2 = -1.79
        M0 = 10**lgM0
        phi1 = 10**lgphi1
        phi2 = 10**lgphi2
        phi = np.log(10) * np.exp(-(xs/M0)) * (xs/M0) * (phi1 * (xs/M0)**alpha1 + phi2 * (xs/M0)**alpha2)
        
    elif z>=4:
        # Song+2016
        if z==4:
            lgM0 = 10.50
            alpha = -1.55
            phi0 = 25.68e-5
        elif z==5:
            lgM0 = 10.97
            alpha = -1.70
            phi0 = 5.16e-5
        elif z==6:
            lgM0 = 10.72
            alpha = -1.91
            phi0 = 1.35e-5
        elif z==7:
            lgM0 = 10.78
            alpha = -1.95
            phi0 = 0.53e-5
        elif z==8:
            lgM0 = 10.72
            alpha = -2.25
            phi0 = 0.035e-5
        M0 = 10**lgM0
            
        phi = phi0 * (xs/M0) * (xs/M0)**alpha * np.exp(-(xs/M0))
    elif z<4:
        # Mcleod+2021
        if z==0.5:
            lgM0 = 10.80
            lgphi1 = -2.77
            alpha1 = -0.61
            lgphi2 = -3.26
            alpha2 = -1.52
        elif z==1.0:
            lgM0 = 10.72
            lgphi1 = -2.80
            alpha1 = -0.46
            lgphi2 = -3.26
            alpha2 = -1.53
        elif z==1.5:
            lgM0 = 10.72
            lgphi1 = -2.94
            alpha1 = -0.55
            lgphi2 = -3.54
            alpha2 = -1.65
        elif z==2.0:
            lgM0 = 10.77
            lgphi1 = -3.18
            alpha1 = -0.68
            lgphi2 = -3.84
            alpha2 = -1.73
        elif z==2.5:
            lgM0 = 10.77
            lgphi1 = -3.39
            alpha1 = -0.62
            lgphi2 = -3.78
            alpha2 = -1.74
        elif z==3.25:
            lgM0 = 10.84
            lgphi1 = -4.30
            alpha1 = -0.
            lgphi2 = -3.94
            alpha2 = -1.79
        M0 = 10**lgM0
        phi1 = 10**lgphi1
        phi2 = 10**lgphi2
        phi = np.log(10) * np.exp(-(xs/M0)) * (xs/M0) * (phi1 * (xs/M0)**alpha1 + phi2 * (xs/M0)**alpha2)
        
    return phi


def planck_nu(T, Rbb, nu):
    '''
    T in the unit of K
    Rbb in the unit of Rsun
    lamb in the unit of Hz
    '''
    x = const.h.cgs.value * nu / (const.k_B.cgs.value * T)
    x = np.array(x)
    # erg/cm2/Ang/sr/s/Hz
    Bnu = (2. * const.h.cgs.value * nu**3 ) /  ( const.c.cgs.value**2 ) / (np.exp(x) - 1. ) 
    spec = Bnu
    # convert back to ANGSTROM   
    Rbb *= const.R_sun.cgs.value
    spec1 = spec * (4. * np.pi * Rbb**2) * np.pi # erg/Hz/s
    # spec1 *= 1./ (4*np.pi*D**2) to correct for distance
    return spec1


def plot_gmf_z():
    lgxs = np.linspace(7, 12)
    zs = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.25, 4, 5,6,7,8])
    zs_bounds = np.array([0.3, 0.75, 1.25, 1.75, 2.25, 2.8, 3.7, 4.5, 5.5, 6.5, 7.5, 8.5])
    zs_left = zs_bounds[:-1]
    zs_right = zs_bounds[1:]
    
    
    ns = np.zeros(len(zs))
    plt.figure()
    ax = plt.subplot(111)
    for i in range(len(zs)):
        z = zs[i]
        phi = get_gmf_z(z, lgxs)
        
        ax.plot(lgxs, phi, label = "z=%.2f"%z)
        
        mask = (lgxs >= 9)
        x_selected = lgxs[mask]
        y_selected = phi[mask]
        
        # Perform numerical integration using Simpson's rule
        n = simpson(y_selected, x=x_selected)
        print (z, n)
        ns[i] = n
    ax.semilogy()
    ax.legend(ncol = 2)
    ax.set_xlim(8, 11.5)
    ax.set_ylim(1e-7, 1e-1)
    ax.set_xlabel("log(M/Msun)")
    ax.set_ylabel("# / dex / Mpc^3")
    return ns
    
    
def plot_tde_sed():
    
    ns = plot_gmf_z()
    zs = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.25, 4, 5,6,7,8])
    zs_bounds = np.array([0.3, 0.75, 1.25, 1.75, 2.25, 2.8, 3.7, 4.5, 5.5, 6.5, 7.5, 8.5])
    zs_left = zs_bounds[:-1]
    zs_right = zs_bounds[1:]
    
    Ns = np.zeros(len(zs))
    
    frac_area = 24 / (4*np.pi/np.pi**2*180**2)
    
    nu_rest = np.logspace(13.2, 15.8, 100)
    lam_rest_um = const.c.cgs.value / nu_rest * 10 * 1e+3
    
    # typical optical pars
    T1 = 2.5e+4
    Rbb1 = 7.2e+3 #Rsun, 10**14.7 / const.R_sun.cgs.value
    Lbb1 = const.sigma_sb.cgs.value * T1**4 * 4 * np.pi * (Rbb1 * const.R_sun.cgs.value)**2
    print ("Optical TDE. Lbb = %.2f e+43 erg/s"%(Lbb1 / 1e+43))
    
    plt.figure()
    ax = plt.subplot(111)
    for i in range(len(zs)):
        myz = zs[i]
        zleft = zs_left[i]
        zright = zs_right[i]
        # comoving volume 
        D_L_Mpc = cosmo.luminosity_distance(myz).value
        D_L_cm = D_L_Mpc * const.pc.cgs.value * 1e+6
        
        D_cd_Mpc_left = cosmo.comoving_distance(zleft).value
        D_cd_Mpc_right = cosmo.comoving_distance(zright).value
        V_c_Mpc3 = 4*np.pi/3 * (D_cd_Mpc_right**3 - D_cd_Mpc_left**3)
        
        
        nu_obs = nu_rest / (1+myz)
        lam_obs_um = lam_rest_um * (1+myz)
        spec1 = planck_nu(T1, Rbb1, nu_rest)
        flux1 = spec1 / (4 * np.pi * D_L_cm**2) # erg/Hz/s/cm^2
        mag1 = -2.5 * np.log10(flux1 / (3631e-23))
        
        ax.plot(lam_obs_um, mag1, label = "z=%.2f"%myz)
        print (V_c_Mpc3, "Mpc^3")
        
        Ns[i] = frac_area * V_c_Mpc3 * ns[i] * 3e-5
    ax.set_ylim(32, 22)
        
    ax.legend()
    ax.semilogx()
    ax.set_xlabel(r"$\mu$"+"m")
    ax.set_ylabel("AB mag")
    ax.plot([0.1, 10], [27, 27], "k:")
    ax.plot([1, 1], [30, 24], "k--")
    
        
    
    
    
    
    
    
    
    
    