import ROOT
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

ROOT.gSystem.Load("lib/libAllpixObjects.so") # Load the dictionaries for Allpix objects

# File Input
filePath = "output/test_output.root" # when run from allpix-squared directory
print(f'Reading data from {filePath}')
file = ROOT.TFile(filePath) # read the file from the specified file path

# Set conditionals for 4 potential objects in the output file
containsDC = True
containsPC = True

# Get the ROOT tree objects
detectorName = "DetectorModel"
#detectorName = "detector1"
print_output = "Objects detected:"
try:
    dcTree = file.Get("DepositedCharge")
    dcBranch = dcTree.GetBranch(detectorName)
    print_output += " DepositedCharge"
except:
    containsDC = False

try:
    pcTree = file.Get("PropagatedCharge")
    pcBranch = pcTree.GetBranch(detectorName)
    print_output += " PropagatedCharge"
except:
    containsPC = False

if containsDC or containsPC:
    print(print_output)
    if not containsPC:
        print("Requires at minimum PropagatedCharge. Quitting")
        exit()
    
else:
    print("Found no applicable objects in root file. Quitting")
    exit()

# Master arrays to store separate events in the same list
deposited_data = []
propagated_data = []

# Utility function to map list of 255 and 1 values to -1 and 1 values
def typeToSign(list):
    if type(list) is int: 
        return 1 if list==1 else -1
    return [1 if item==1 else -1 for item in list]

# Loop over events in tree (same number of event for each)
num_events = pcTree.GetEntries()
for iEvent in range(num_events):
    
    # Define objects for each event
    deposit = {'x': [], 'y': [], 'z': [], 'q':[], 'type':[]} # Deposited Charges Global Positions
    propagation = {'x': [], 'y': [], 'z': [], 'q':[], 'type':[]} # Propagated Charges Global Positions

    # Update each tree to the current event
    if containsDC: dcTree.GetEntry(iEvent)
    if containsPC: pcTree.GetEntry(iEvent)

    # May need to get the branch here if they need to be updated for each event
    if containsDC: dcBranch = dcTree.GetBranch(detectorName)
    if containsPC: pcBranch = pcTree.GetBranch(detectorName)

    # Get ROOT branch as python vector (object?)
    if containsDC: br_dc = getattr(dcTree, dcBranch.GetName())
    if containsPC: br_pc = getattr(pcTree, pcBranch.GetName())

    # Get the locations and amounts of deposited charges
    if containsDC:
        for dc in br_dc:
            deposit['x'].append(dc.getGlobalPosition().x())
            deposit['y'].append(dc.getGlobalPosition().y())
            deposit['z'].append(dc.getGlobalPosition().z())
            deposit['q'].append(abs(dc.getCharge()))
            deposit['type'].append(typeToSign(dc.getType()))

    # Get the locations of propagated charges
    if containsPC:
        for pc in br_pc:
            propagation['x'].append(pc.getGlobalPosition().x())
            propagation['y'].append(pc.getGlobalPosition().y())
            propagation['z'].append(pc.getGlobalPosition().z())
            propagation['q'].append(abs(pc.getCharge())) # Not the actual charge but the induced charge from each PC
            propagation['type'].append(typeToSign(pc.getType()))

    # Add objects to master lists
    deposited_data.append(deposit)
    propagated_data.append(propagation)

num_pixels_x = 1
num_pixels_y = 1
pixel_width = 0.330 #mm
pixel_height = 2 #mm

# Plot Deposited & Propagated Charge Distributions
if (containsDC or containsPC) and num_events <= 5:

    num_bins = 30
    # Plot deposited and propagated charge distributions
    if containsDC and containsPC:
        
        charge_fig, charge_axs = plt.subplots(1,2)
        for i in range(num_events):
            dcGPos = deposited_data[i]
            pcGPos = propagated_data[i]

            charge_axs[0].hist2d(dcGPos['x'], dcGPos['y'], bins=num_bins)
            charge_axs[1].hist2d(pcGPos['x'], pcGPos['y'], bins=num_bins)
        for ax in charge_axs:
            ax.set_aspect('equal')
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')

        charge_fig.suptitle("Charge Distributions")
        charge_axs[0].set_title('Deposited Charges')
        charge_axs[1].set_title('Propagated Charges')

        charge_axs[0].set_xlim(-num_pixels_x/2*pixel_width, num_pixels_x/2*pixel_width)
        charge_axs[0].set_ylim(-num_pixels_y/2*pixel_width, num_pixels_y/2*pixel_width)
        charge_axs[1].set_xlim(-num_pixels_x/2*pixel_width, num_pixels_x/2*pixel_width)
        charge_axs[1].set_ylim(-num_pixels_y/2*pixel_width, num_pixels_y/2*pixel_width)

    # Plot separate 2D distributions for propagated holes and electrons
    if containsPC:
        dist_fig, dist_axs = plt.subplots(1,2, constrained_layout=True)
        for i in range(num_events):
            pcGPos = propagated_data[i]
            e_selector = np.array(pcGPos['type'])==-1
            h_selector = np.array(pcGPos['type'])==1

            dist_axs[0].hist2d(np.array(pcGPos['x'])[e_selector], np.array(pcGPos['y'])[e_selector], bins=num_bins)
            dist_axs[1].hist2d(np.array(pcGPos['x'])[h_selector], np.array(pcGPos['y'])[h_selector], bins=num_bins)
        for ax in dist_axs:
            ax.set_aspect('equal')
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
        
        dist_fig.suptitle('Propagated Charge Distributions')
        dist_axs[0].set_title('Propagated Electrons')
        dist_axs[1].set_title('Propagated Holes')

        dist_axs[0].set_xlim(-num_pixels_x/2*pixel_width, num_pixels_x/2*pixel_width)
        dist_axs[0].set_ylim(-num_pixels_y/2*pixel_width, num_pixels_y/2*pixel_width)
        dist_axs[1].set_xlim(-num_pixels_x/2*pixel_width, num_pixels_x/2*pixel_width)
        dist_axs[1].set_ylim(-num_pixels_y/2*pixel_width, num_pixels_y/2*pixel_width)

    # Plot scatter with one or both of Deposited & Propagated Charge Counts
    scatter_fig = plt.figure()
    scatter_ax = scatter_fig.add_subplot(projection='3d')
    opacity_scaling = 0.1/num_events 

    for i in range(num_events): # Add the charges from each event
        if containsDC: 
            dcGPos = deposited_data[i]
            scatter_ax.scatter(dcGPos['x'], dcGPos['y'], dcGPos['z'], marker='.', alpha=opacity_scaling)
        if containsPC:
            pcGPos = propagated_data[i]
            scatter_ax.scatter(pcGPos['x'], pcGPos['y'], pcGPos['z'], marker='.', alpha=opacity_scaling)

    # Add lines indicating the center pixel location
    pixel_x = [pixel_width*k for k in [-0.5, 0.5, 0.5, -0.5, -0.5]]
    pixel_y = [pixel_width*k for k in [-0.5, -0.5, 0.5, 0.5, -0.5]] 
    pixel_z = [pixel_height*k for k in [0.5, 0.5, 0.5, 0.5, 0.5]] # pixel is center in z
    scatter_ax.plot(pixel_x, pixel_y, pixel_z, linestyle='--', color='k')

    scatter_ax.set_xlabel('x [mm]')
    scatter_ax.set_ylabel('y [mm]')
    scatter_ax.set_zlabel('z [mm]')
    scatter_ax.set_xlim(-num_pixels_x/2*pixel_width, num_pixels_x/2*pixel_width) # bounds of 5x5 are -0.825,0.825
    scatter_ax.set_ylim(-num_pixels_y/2*pixel_width, num_pixels_y/2*pixel_width) # bounds of 5x5 are -0.825,0.825
    scatter_ax.set_zlim(-1, 1)

    # Plot histogram
    hist_fig, hist_ax = plt.subplots()
    bins = np.linspace(-1,1,200)

    dcG_z = {'e':[],'h':[]}
    pcG_z = {'e':[],'h':[]}
    for i in range(num_events): # Add the charges from each event to a single array with only z values
        if containsDC: 
            dcGPos = deposited_data[i]
            dcG_z['e'].append(np.array(dcGPos['z'])[np.array(dcGPos['type'])==-1]) 
            dcG_z['h'].append(np.array(dcGPos['z'])[np.array(dcGPos['type'])==1]) 
        if containsPC:
            pcGPos = propagated_data[i]
            pcG_z['e'].append(np.array(pcGPos['z'])[np.array(pcGPos['type'])==-1]) 
            pcG_z['h'].append(np.array(pcGPos['z'])[np.array(pcGPos['type'])==1]) 
    
    if containsDC: 
        hist_ax.hist(dcG_z['e'], bins, alpha=0.5, label='Deposited Electrons')
        hist_ax.hist(dcG_z['h'], bins, alpha=0.5, label='Deposited Holes')
    if containsPC: 
        hist_ax.hist(pcG_z['e'], bins, alpha=0.5, label='Propagated Electrons')
        hist_ax.hist(pcG_z['h'], bins, alpha=0.5, label='Propagated Holes')

    hist_ax.legend()

    # Set titles depending on charge objects available
    if containsDC and containsPC: 
        scatter_ax.set_title("Locations of Deposited and Propagated Charges")
        hist_ax.set_title("Z Positions of Deposited and Propagated Charges")
    elif containsDC: 
        scatter_ax.set_title("Locations of Deposited Charges")
        hist_ax.set_title("Z Positions of Deposited Charges")
    else: 
        scatter_ax.set_title("Locations of Propagated Charges")
        hist_ax.set_title("Z Positions of Propagated Charges")

#TODO: Take the final locations of the charges and calculate the electric field (potential is probably easier) in the detector. Plot slices or level curves of the potential

plt.show()


