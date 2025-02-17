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


def get_jetted_tde_lum(t=4, nu_rest = np.logspace(14, 15.5), verbose = False):
    """
    t: rest-frame days 
    nu_rest: the observing band frequency in the TDE's rest frame
    """
    nu0 = 1e+15 # a reference frequency in the TDE's rest frame
    
    ### power-law component
    tpeak_pl = 0.03 # in days
    beta_rise = 3.62 
    beta_decay = -0.46
    Lnu0_peak_pl = 10**30.56 # erg/s/Hz
    
    L1_nu_ref = 0
    if t <= tpeak_pl:
        L1_nu_ref = Lnu0_peak_pl * 10**(beta_rise * (t - tpeak_pl))
    else:
        L1_nu_ref = Lnu0_peak_pl * 10**(beta_decay * (t - tpeak_pl))
        
    alpha = -1.11
    L1_nu = L1_nu_ref * (nu_rest/nu0)**alpha
    nuLnu_comp1 = nu_rest * L1_nu
        
    ### blackbody component
    L0_peak_bb = 10**44.41 
    tpeak_bb = 5.84 # in days
    sigma_bb = 10**1.06
    tau_bb = 10**1.53
    T0 = 10**4.57
    
    L2_ref = 0 # nuLnu at nu0
    if t<= tpeak_bb:
        L2_ref = L0_peak_bb * np.exp(-1*(t - tpeak_bb)**2 / (2*sigma_bb**2))
    else:
        L2_ref = L0_peak_bb * np.exp(-1*(t - tpeak_bb) / tau_bb)
    
    _nuLnu = nu_rest * Planck(nu=nu_rest, T=T0)
    _nuLnu_0 = nu0*Planck(nu=nu0, T=T0)
    
    nuLnu_comp2 = L2_ref * _nuLnu / _nuLnu_0
        
    
    nuLnu = nuLnu_comp1 + nuLnu_comp2
    
    if verbose:
        plt.figure()
        plt.plot(nu_rest, nuLnu_comp1)
        plt.plot(nu_rest, nuLnu_comp2)
        plt.plot(nu_rest, nuLnu)
        plt.semilogx()
        plt.semilogy()
        plt.xlabel("nu_rest (Hz)")
        plt.xlabel("nuLnu (erg/s)")
    return nuLnu
    


def generate_roman_lc():
    lamb_obs_um = 1 
    peakmag = 26
    peakmag_jetted = 25.5
    verbose = False
    
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
        
        if verbose:
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
    
    # calculate the maximum distance of a jetted TDE
    mpeaks_obs = np.zeros(len(zs))
    ts_rest = np.hstack([np.linspace(0.01, 0.09, 9), np.linspace(1, 6, 51)])
    for j in range(len(zs)):
        z = zs[j]
        nu_rest = nu_obs * (1+z)
        nuLnus = np.zeros(len(ts_rest))
        for k in range(len(ts_rest)):
            nuLnus[k] = get_jetted_tde_lum(ts_rest[k], nu_rest)
        
        mags_obs =  -2.5 * np.log10(nuLnus / nu_obs / (4 * np.pi * D_cms[j]**2) / (3631e-23))
        #plt.plot(ts_rest, mags_obs)
        mpeaks_obs[j] = min(mags_obs)
    if verbose:
        plt.figure()
        ax = plt.subplot(111)
        ax.plot(zs, mpeaks_obs)
        ax.semilogy()
        ax.semilogx()
        ax.plot([zs[0], zs[-1]], [peakmag, peakmag])
        ax.set_xlabel("z")
        ax.set_ylabel("apparant mag")
        ax.set_title("AT2022cmc")
    myfunc = interp1d(mpeaks_obs, zs)
    zmax_jetted = myfunc(peakmag_jetted)
    
    
    plt.figure(figsize = (5, 4))
    ax = plt.subplot(111)
    
    nu_rest_jetted = nu_obs * (1+zmax_jetted)
    D_cm_jetted = cosmo.luminosity_distance([zmax_jetted])[0].value * 1e+6  * const.pc.cgs.value
    ts_rest_jetted = np.hstack([np.linspace(0.001, 0.040, 41), np.linspace(0.05, 0.1, 6), 
                                np.linspace(0.1, 5, 50), np.linspace(5, 50, 46)])
    nuLnus_jetted = np.zeros(len(ts_rest_jetted))
    for k in range(len(ts_rest_jetted)):
        nuLnus_jetted[k] = get_jetted_tde_lum(ts_rest_jetted[k], nu_rest_jetted)
    mags_obs_jetted =  -2.5 * np.log10(nuLnus_jetted / nu_obs / (4 * np.pi * D_cm_jetted**2) / (3631e-23))
    ind_peak = np.argsort(mags_obs_jetted)[0]
    ts_obs_jetted = (ts_rest_jetted - ts_rest_jetted[ind_peak])*(1+zmax_jetted)
    ax.plot(ts_obs_jetted, mags_obs_jetted, label = "Jetted TDE, "+r"$z = %.1f$"%zmax_jetted,
            linestyle = "--")
    
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
            linestyle = "-."
        elif atname == "2020vwl":
            label = "typical"
            linestyle = "-"
        elif atname == "2020acka":
            label = "overluminous"
            linestyle = ":"
        label += ", "+r"$z = %.1f$"%(z)
        
        ax.plot(t_obs, ymod, label = label, linestyle = linestyle)
        
    ax.set_ylim(30, peakmag_jetted - 0.2)
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
    plt.tight_layout()
    plt.savefig("./figs/tde_panel_1.pdf")
    
    
if __name__=="__main__":
    generate_roman_lc()
        
        
        
    