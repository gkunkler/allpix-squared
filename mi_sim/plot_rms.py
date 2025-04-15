import ROOT
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

ROOT.gSystem.Load("lib/libAllpixObjects.so") # Load the dictionaries for Allpix objects

# File Input
filePath = "output/modules.root" # when run from allpix-squared directory
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

# Plot in two adjacent subplots
rms_fig, rms_axs = plt.subplots(1,2,constrained_layout=True, figsize=(10,5))

rms_axs[0].plot(t,e_rms, color='k', linestyle='--', label='e-')
rms_axs[0].plot(t,h_rms, color='k', linestyle='-', label='hole')

rms_axs[1].plot(t,e_x_rms, color='r', linestyle='-', label='x')
rms_axs[1].plot(t,e_y_rms, color='g', linestyle='-.', label='y')
rms_axs[1].plot(t,e_z_rms, color='b', linestyle=':', label='z')
rms_axs[1].plot(t,e_rms, color='k', linestyle='--', label='total')

for ax in rms_axs:
    ax.set_xlabel('t [ns]')
    ax.set_ylabel('rms [mm]')
    ax.legend()

rms_fig.suptitle('Charge Spread During Propagation')
rms_axs[0].set_title('Electron and Holes')
rms_axs[1].set_title('Electrons in XYZ')

# Fitting
if True:

    # Fit configuration

    min_t = 0 # Pick the region we are fitting
    max_t = 99

    f = e_z_rms # Set the function we are fitting
    function_name = "Electron (z)"
    # f = e_rms # Set the function we are fitting
    # function_name = "Electron (3D RMS)"
    
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

    # if not min_i: 
    #     min_i = 0
    # else:
    #     min_i = max(min_i[0][0], 0)
    # if not max_i: 
    #     max_i = len(t)-1
    # else:
    #     max_i = min(max_i[0][0], len(t)-1)

    # Fit inside fit region
    y_fit = f[min_i:max_i] 
    x_fit = t[min_i:max_i]

    def fit_exponential(x, a, n):
        return a * (x**n)
    
    params, covariance = curve_fit(fit_exponential, x_fit, y_fit)

    a, n = params
    a_v, n_v = covariance

    print(f'params: a={a}, n={n}')
    print(f'covariance: a={a_v}, n={n_v}')

    fit_fig, fit_ax = plt.subplots()
    
    fit_ax.plot(t, f, color='k', linestyle='-', label=function_name)
    fit_ax.plot(x_fit, fit_exponential(x_fit,a,n), color='b', linestyle='--', label=f'fit: {np.round(a,5)}*x^{np.round(n,4)}')

    fit_ax.set_xlabel('time, t [ns]')
    fit_ax.set_ylabel('rms [mm]')

    fit_fig.suptitle(f'Charge Spread for {function_name}')

    # Compare to diffusion const.
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