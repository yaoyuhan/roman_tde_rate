#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:26:49 2025

@author: yuhanyao
"""
import pickle
import numpy as np
from scipy.interpolate import interp1d

import astropy.io.ascii as asci
import astropy.constants as const

import matplotlib
import matplotlib.pyplot as plt
fs= 10
matplotlib.rcParams['font.size']=fs
ms = 6
matplotlib.rcParams['lines.markersize']=ms

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)


def Planck(nu=None, T=None):
    """
    >> I = Planck(nu=1e14, T=1e4)

    return black body intensity (power per unit area per solid angle per frequency)
    """
    h = const.h.cgs.value
    c = const.c.cgs.value
    k_B = const.k_B.cgs.value
    x = np.exp(h*nu/(k_B*T))
    x = np.array(x)
    return 2*h/c**2 * nu**3 / (x-1)


def cc_bol(T, nu):
    """
    Color correction for blackbody spectrum
    (factor to multiply) from bolometric luminosity to nuLnu 
    
    note that nu is the frequency in TDE's rest-frame
    
    nuLnu / L_bol = nu pi B_nu(T) / (sigmaT^4)
    """
    return Planck(nu, T)*nu * np.pi /  (const.sigma_sb.cgs.value * T**4)


def generate_roman_lc():
    lamb_obs_um = 1
    peakmag = 26
    do_plot = False
    
    lamb_obs_cm = lamb_obs_um/1e+4
    
    zs = np.hstack([np.linspace(0.01, 4.99, 499),
                    np.linspace(5, 20, 151)])
    D_pcs = cosmo.luminosity_distance([zs])[0].value * 1e+6 
    D_cms = D_pcs * const.pc.cgs.value
    
    D_10pc = 10 * const.pc.cgs.value
    nu_obs = const.c.cgs.value / lamb_obs_cm
    tb = asci.read("./data/representative_tde.dat")
    zmaxs = np.zeros(len(tb))
    for i in range(len(tb)):
        ztfname = tb["ztfname"][i]
        #z = tb["z"][i]
        lgTbb = tb["lgTbb"][i]
        lgRbb = tb["lgRbb"][i]
        lgLbb = tb["lgLbb"][i]
        
        mpeaks_obs = np.zeros(len(zs))
        for j in range(len(zs)):
            z = zs[j]
            nu_rest = nu_obs * (1+z)
            nuLnu_rest = 10**lgLbb * cc_bol(10**lgTbb, nu=nu_rest)
            Lnu_obs = nuLnu_rest / nu_obs
            
            mpeaks_obs[j] = -2.5 * np.log10(Lnu_obs / (4 * np.pi* D_cms[j]**2)/3631e-23)
        
        if do_plot:
            plt.figure()
            ax = plt.subplot(111)
            ax.plot(zs, mpeaks_obs)
            ax.semilogy()
            ax.semilogx()
            ax.plot([zs[0], zs[-1]], [peakmag, peakmag])
            ax.set_xlabel("z")
            ax.set_ylabel("apparant mag")
            ax.set_title(ztfname)
            
        # get the highest redshift out to which this transient can be detected by Roman
        myfunc = interp1d(mpeaks_obs, zs)
        zmaxs[i] = myfunc(peakmag)
    
    plt.figure()
    ax = plt.subplot(111)
    for i in range(len(tb)):
        ztfname = tb["ztfname"][i]
        atname = tb["atname"][i]
        z = zmaxs[i]
        D_cm = cosmo.luminosity_distance([z])[0].value * 1e+6  * const.pc.cgs.value
        
        tpeak_mjd_guess = tb['tmax_visual'][i]
        fit_name = tb["fit_name"][i]
        fname = './data/mcmc_result/%s_%s.pickle'%(ztfname, fit_name)
        #print (fname)
        
        mcmc_dt = pickle.load(open(fname,'rb'),encoding='latin1')[0]
        tpeak_mjd =  tpeak_mjd_guess + mcmc_dt["tpeak"][0]*(1+z)
        
        lgT = mcmc_dt["T"][0]
        T1 = 10**lgT
        
        lcs = mcmc_dt["lc"]
        tmod = lcs["time"]-mcmc_dt["tpeak"][0]
        Lmod_ = lcs["L_kc"]
        Lmod = np.percentile(Lmod_, [50], axis=0)[0] # This is nuLnu rest-frame g-band
        
        nug = 6.3e+14
        Lbbs = Lmod / cc_bol(T1, nu=nug) # bolometric blackbody luminosity
        
        nu_rest = nu_obs * (1+z)
        nuLnu_rest = Lbbs * cc_bol(T1, nu=nu_rest)
        
        fratio = nuLnu_rest / nu_obs / (4 * np.pi * D_cm**2) / (3631e-23)
        ymod =  -2.5 * np.log10(fratio)
        
        
        ix = ymod < 30
        tmod = tmod[ix]
        ymod = ymod[ix]
        
        # time dilation
        t_obs = tmod * (1+z)
        
        if atname == "2020vdq":
            label = "subluminous"
        elif atname == "2020vwl":
            label = "typical"
        elif atname == "2020acka":
            label = "overluminous"
        label += ", "+r"$z = %.1f$"%(z)
        
        ax.plot(t_obs, ymod, label = label)
        
    ax.set_ylim(30, peakmag - 0.2)
    ax.set_xlim(-520, 2000)
    ax.legend()
    ax.tick_params(which = 'major', length = 4, top=True, direction = "in", 
                         right = True)
    ax.tick_params(which = 'minor', length = 2, top=True, direction = "in", 
                         right = True)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_major_locator(plt.MultipleLocator(500))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(100))
    ax.set_xlabel("Days since maximum")
    ax.set_ylabel("Observed AB mag")
    ax.set_label("Roman TDE light curve")
    plt.savefig("./figs/tde_panel_1.pdf")
    
    
if __name__=="__main__":
    generate_roman_lc()
        
        
        
    