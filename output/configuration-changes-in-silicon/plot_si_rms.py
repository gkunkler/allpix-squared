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

# Plot in two adjacent subplots
rms_fig, rms_axs = plt.subplots(1,2,constrained_layout=True, figsize=(10,5), sharey=True)


filePaths = ['modules_cr1_d1_e_100.root', 'modules_cr1_d1_eh_100.root', 'modules_cr1_d1_ehm_100.root']
names = ['Electrons only','Electrons & Holes','Electrons, Holes, Mirror Charges']
linestyles = ['-', '--', ':', '-.']

# File Input
for i in range(len(filePaths)):
    filePath = filePaths[i]

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
    
rms_axs[0].set_ylabel('rms [mm]')

for ax in rms_axs:
    ax.set_xlabel('t [ns]')
    ax.legend()

rms_fig.suptitle('Charge Spread During Propagation')
rms_axs[0].set_title('x-direction')
rms_axs[1].set_title('z-direction')

# Fitting
if False:

    f = e_z_rms

    a, n = rms_fit(t,f, min_t=0.2, max_t=19)
    function_name = "Electron (z)"

    fit_fig, fit_ax = plt.subplots()
    
    fit_ax.plot(t, f, color='k', linestyle='-', label=function_name)
    fit_ax.plot(t, a*(t**n), color='b', linestyle='--', label=f'fit: {np.round(a,5)}*x^{np.round(n,4)}')

    fit_ax.set_xlabel('time, t [ns]')
    fit_ax.set_ylabel('rms [mm]')

    fit_fig.suptitle(f'Charge Spread for {function_name}')

    # Get initial slope
        
    def fit_line(x,m,b):
        return m*x + b
    
    slope, intercept = rms_initial_slope(t,f)

    x_slope_max = (np.max(f)-intercept)/slope * 0.8 # Find x location to stop plotting slope
    slope_i_max = np.where(t>=x_slope_max)[0][0]
    # try:
    #     slope_i_max = min(slope_i_max[0][0], len(t)-1)
    # except:
    #     slope_i_max = len(t)-1
    t_slope = t[1:3]

    fit_ax.plot(t_slope, fit_line(t_slope,slope,intercept), color='r', linestyle=':', label=f'initial slope: {np.round(slope,5)}')

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
        fit_ax.plot(x_fit, fit_exponential(x_fit,a_exp,0.5), color='r', linestyle=':', label=f'expected diffusion: sqrt({dim}*2Dt) with a=sqrt({dim}*2D)={np.round(a_exp,5)}')

    fit_ax.legend()

plt.show()