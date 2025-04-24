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
    coulomb_mag_histo = dir.Get("coulomb_mag_histo")
except:
    print("Failed to read the plots. Exiting.")
    exit()

# Extract data from the histogram
nbins = coulomb_mag_histo.GetNbinsX()
x = np.array([coulomb_mag_histo.GetBinCenter(i) for i in range(1, nbins + 1)])
y = np.array([coulomb_mag_histo.GetBinContent(i) for i in range(1, nbins + 1)])

# Plot the histogram data using matplotlib
plt.figure(figsize=(10, 6))
plt.plot(x, y, drawstyle='steps-mid', label="Coulomb Magnitude")
plt.yscale('log')
plt.xlabel("Electric Field Magnitude [V/cm]")
plt.ylabel("Counts")
plt.title("Coulomb Interaction Magnitudes")
plt.legend()
plt.grid()
plt.show()
