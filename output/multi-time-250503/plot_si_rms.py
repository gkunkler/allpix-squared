# Run this script from the directory with the root files

import ROOT
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Fitting Functions

def rms_fit(t,f, min_t=0.2, max_t = 50):

    # Fit configuration
    min_i = np.where(t<=min_t)
    max_i = np.where(t>=max_t)

    try:
        min_i = max(min_i[0][0], 0)
    except:
        min_i = 0

    try:
        max_i = min(max_i[0][0], len(t)-1)
    except:
        max_i = len(t)-1

    y_fit = f[min_i:max_i] 
    x_fit = t[min_i:max_i]

    def fit_exponential(x, a, n):
        return a * ((x)**n)
    
    params, covariance = curve_fit(fit_exponential, x_fit, y_fit)

    a, n = params
    a_v, n_v = covariance

    # print(f'params: a={a}, n={n}')
    # print(f'covariance: a={a_v}, n={n_v}')

    return a,n

def rms_initial_slope(t,f,i_range = [1,3]):

    x_slope = t[i_range[0]:i_range[1]]
    y_slope = f[i_range[0]:i_range[1]] 
    
    slope, intercept, r, p, std_err = linregress(x_slope, y_slope)

    return slope,intercept

ROOT.gSystem.Load("../../lib/libAllpixObjects.so") # Load the dictionaries for Allpix objects

plot_fit = True

# Plot in two adjacent subplots
rms_fig, rms_axs = plt.subplots(1,2,constrained_layout=True, figsize=(10,6), dpi=150, sharex=True, sharey=True)
if plot_fit: 
    fit_fig, fit_axs = plt.subplots(1,2,constrained_layout=True, figsize=(10,6), dpi=150,sharex=True, sharey=True)

configs = ['d1_cr1_e_-100V_25ns','d1_cr1_em_-100V_25ns','d1_cr1_eh_-100V_25ns','d1_cr1_ehm_-100V_25ns']
names = ['Electrons only','Electrons & Mirror Charges', 'Electrons & Holes','Electrons, Holes, Mirror Charges']
# configs = ['d1_cr0_e_-100V_25ns','d1_cr1_e_-100V_25ns','d1_cr1_eh_-100V_25ns','d1_cr1_em_-100V_25ns','d1_cr1_ehm_-100V_25ns']
# names = ['No CR','Electrons only','Electrons & Holes','Electrons & Mirror Charges','Electrons, Holes, Mirror Charges']
# configs = ['d1_cr0_e_0V_25ns','d0_cr1_e_0V_25ns','d1_cr1_e_0V_25ns']
# names = ['Diffusion only','CR only','Diffusion & CR']

# names = ['0V','-200V', '-300V', '-400V']
linestyles = ['-', '--', '-.', ':', '--']
title_addition = "100V Bias Voltage"

# File Input
for i in range(len(configs)):
    filePath = f"raw/{configs[i]}/modules.root"

    print(f'Reading data from {filePath}')
    file = ROOT.TFile(filePath) # read the file from the specified file path

    detectorName = "DetectorModel"
    try:
        dir = file.Get("InteractivePropagation").Get(detectorName) # Access the plots from InteractivePropagation
        total_multigraph = dir.Get("rms_total_graph")
        e_multigraph = dir.Get("rms_e_graph")
    except:
        print("Failed to read the plots. Exiting.")
        exit()

    # Extract the info into arrays
    total_graph_list = total_multigraph.GetListOfGraphs()
    e_graph_list = e_multigraph.GetListOfGraphs()

    t = np.array(total_graph_list[0].GetX())
    e_rms = np.array(total_graph_list[0].GetY())
    h_rms = np.array(total_graph_list[1].GetY())
    e_x_rms = np.array(e_graph_list[0].GetY())
    e_y_rms = np.array(e_graph_list[1].GetY())
    e_z_rms = np.array(e_graph_list[2].GetY())

    rms_axs[0].plot(t,e_x_rms, linestyle=linestyles[i], label=names[i])
    rms_axs[1].plot(t,e_z_rms, linestyle=linestyles[i], label=names[i])

    # Fitting
    if plot_fit:

        f = e_x_rms
        a, n = rms_fit(t,f, min_t=0.2, max_t=16)
        fit_axs[0].plot(t, f, color='k', linestyle='-', label=f'{names[i]}')
        fit_axs[0].plot(t, a*(t**n), linestyle='--', label=f'{names[i]}: {np.round(a,5)}*x^{np.round(n,4)}')

        f = e_z_rms
        a, n = rms_fit(t,f, min_t=0.2, max_t=16)
        fit_axs[1].plot(t, f, color='k', linestyle='-', label=f'{names[i]}')
        fit_axs[1].plot(t, a*(t**n), linestyle='--', label=f'{names[i]}: {np.round(a,5)}*x^{np.round(n,4)}')

        fit_axs[0].set_xlabel('time, t [ns]')
        fit_axs[0].set_ylabel('rms [mm]')
        fit_axs[1].set_xlabel('time, t [ns]')
        # fit_axs[1].set_ylabel('rms [mm]')

        fit_fig.suptitle(f'Electron Spread Fitting ({title_addition})')
        fit_axs[0].set_title('x-direction')
        fit_axs[1].set_title('z-direction')

        # Compare to diffusion const.
        if (False):
            mu = 1000 * 10**2 * 10**-9 # cm^/V/s * (10mm/cm)^2 * (1s/10^9ns) = mm^2/V/ns (this is for electrons)
            # mu = 100 * 10**2 * 10**-9 # cm^/V/s * (10mm/cm)^2 * (1s/10^9ns) = mm^2/V/ns (this is for holes)
            kB = 8.62*10**-5 # eV/K
            T = 273 # K
            q = 351 # e (average per charge group?)

            dim = 1
            D = kB*T*mu #/q # mm^2/ns
            print(D)
            a_exp = np.sqrt(2*D*dim)
            print(a_exp)
            fit_axs[1].plot(x_fit, fit_exponential(x_fit,a_exp,0.5), color='r', linestyle=':', label=f'expected diffusion: sqrt({dim}*2Dt) with a=sqrt({dim}*2D)={np.round(a_exp,5)}')

        fit_axs[0].legend()
        fit_axs[1].legend()

        fit_axs[0].set_xlim(-0.5,20)
        fit_axs[1].set_xlim(-0.5,20)
    
rms_axs[0].set_ylabel('rms [mm]')

for ax in rms_axs:
    ax.set_xlabel('t [ns]')
    ax.legend()
    ax.set_xlim(-0.5,20)

rms_fig.suptitle(f'Electron Spread During Propagation ({title_addition})')
rms_axs[0].set_title('x-direction')
rms_axs[1].set_title('z-direction')



plt.show()