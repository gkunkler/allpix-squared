# Run from output/multi

import ROOT
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

ROOT.gSystem.Load("../../lib/libAllpixObjects.so") # Load the dictionaries for Allpix objects

# Plot in two adjacent subplots
# rms_fig, rms_axs = plt.subplots(1,2,constrained_layout=True, figsize=(10,5), sharey=True)

# configs = ['d1_cr0_e_-50V_5ns','d1_cr0_e_-50V_10ns','d1_cr0_e_-50V_25ns']
# names = ['5ns','10ns','25ns']
# title_addition = "Diffusion only"
configs = ['d1_cr0_e_-100V_5ns', 'd1_cr1_e_-100V_5ns', 'd1_cr1_eh_-100V_5ns']
names = ['Diffusion Only', 'CR, Electrons only', 'CR, Electrons & Holes']
title_addition = "100V Bias Voltage, 25ns"

hist_fig, hist_axs = plt.subplots(3,1,constrained_layout=True, figsize=(8,6), sharex=True, sharey=True, dpi=200)

# File Input
for fileIndex in range(len(configs)):

    filePath = f"raw/{configs[fileIndex]}/data.root"
    print(f'Reading data from {filePath}')
    file = ROOT.TFile(filePath) # read the file from the specified file path

    # Set conditionals for 2 potential objects in the output file
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

    pixel_height = 0.65 #mm

    # Plot Deposited & Propagated Charge Distributions
    if (containsDC or containsPC) and num_events <= 5:

        # Plot histogram
        # hist_fig, hist_ax = plt.subplots()
        bins = np.linspace(-pixel_height/2,pixel_height/2,200)

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
        #     hist_ax.hist(dcG_z['e'], bins, alpha=0.5, label='Deposited Electrons')
        #     hist_ax.hist(dcG_z['h'], bins, alpha=0.5, label='Deposited Holes')
            for z in np.unique(dcG_z['h'] + dcG_z['e']):
                hist_axs[fileIndex].axvline(z, linestyle='dotted', linewidth=2, color='k', label='Deposited Charge Locations')

        if containsPC: 
            if pcG_z['e']:
                hist_axs[fileIndex].hist(pcG_z['e'], bins, alpha=0.5, label='Propagated Electrons')
            if pcG_z['h']:
                hist_axs[fileIndex].hist(pcG_z['h'], bins, alpha=0.5, label='Propagated Holes')

        # Filter out duplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        hist_axs[fileIndex].legend(handles, labels, loc='best')

        # Set titles depending on charge objects available
        hist_axs[fileIndex].set_ylabel(names[fileIndex])
        if containsDC and containsPC: 
            hist_fig.suptitle(f"Z Positions of Deposited and Propagated Charges ({title_addition})")
        elif containsDC: 
            hist_fig.suptitle(f"Z Positions of Deposited Charges ({title_addition})")
        else: 
            hist_fig.suptitle(f"Z Positions of Propagated Charges ({title_addition})")

    #TODO: Take the final locations of the charges and calculate the electric field (potential is probably easier) in the detector. Plot slices or level curves of the potential

plt.show()


