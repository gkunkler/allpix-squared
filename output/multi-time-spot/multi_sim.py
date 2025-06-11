#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:52:20 2024

@author: rickard
"""

from tools.AllpixSim import AllpixObject
import numpy as np
import time
import ROOT
import os
import pickle
import shutil # Used to copy files

total_start_time = time.time()

ROOT.gSystem.Load("lib/libAllpixObjects.so") # Load the dictionaries for Allpix objects

output_folder = "output/multi/raw"
output_object_folder = "output/multi"
output_list = []

# Constant Values
num_pulses = 100 # Number of particles per event
keV = 1
# enable_coulomb = 1
charge_per_step = 1
coulomb_field_limit = 4e5
max_charge_groups = 2000
time_step = 0.02
integration_time = 10 # ns
# spot_size = 0 #um
num_charges = 1000 # number of 
# coulomb_distance_limit = 4e-5 #cm
bias_voltage = 0 #V

# Changing Values
# integration_times = [5,10,25] # ns
# bias_voltages = [0,-50,-100,-200,-300] # V
# charges_per_steps = {0.5:14, 1:28, 3:83, 5:138, 10:275} # 0.1% of total charge (num_pulses*keV*1000/ionization_energy * 0.001) -> approx. 1000 charge groups
# max_charge_groups_list = [1000]
# keVs = [0.5,5,10] # ionization energy = 3.64eV
coulomb_distance_limits = [2e-4, 1e-1] #cm
repeats = 1
time_steps = [0.005,0.01,0.05,0.1] #ns
spot_sizes = [0.1, 0.5, 1] #um
# coulomb_field_limits = [4e5] #[5e4,7e4,1e5,3e5,5e5,7e5,1e6,2e6]

for i in range(repeats):

    # for keV in keVs:
    for coulomb_distance_limit in coulomb_distance_limits:
        # charge_per_step = charges_per_steps[keV]

        # for coulomb_field_limit in coulomb_field_limits:
        # for enable_diffusion in [0,1]:
        # for propagation_type in [[1,0], [0,1], [1,1]]: # [d,cr]
            # enable_diffusion = propagation_type[0]
            # enable_coulomb = propagation_type[1]
        for spot_size in spot_sizes:

            for time_step in time_steps:

                # for max_charge_groups in max_charge_groups_list:
                # for time_step in time_steps:
                # for propagation_charges in [[1,0,0], [1,1,0], [1,0,1], [1,1,1]]: # [e,h,m]
                # for propagation_config in [[1,1,1,0,0]]: #[d,cr,e,h,m]
                for propagation_config in [[1,1,1,0,0], [1,0,1,0,0]]:

                    enable_diffusion = propagation_config[0]
                    enable_coulomb = propagation_config[1]
                    propagate_electrons = propagation_config[2]
                    propagate_holes = propagation_config[3]
                    include_mirror = propagation_config[4]
                        
                    allpix = AllpixObject()

                    allpix.LOG_LEVEL            = "INFO" #STATUS, INFO, WARNING, DEBUG
                    allpix.LOG_FORMAT           = "DEFAULT"
                    allpix.DETECTOR_FILE        = "detector.conf"
                    # allpix.SOURCE_FILE          = "macro_multi.mac"
                    # allpix.CHANGE_DETECTOR      = False
                    # allpix.PIXEL_SIZE_X         = 1000 #um
                    # allpix.PIXEL_SIZE_Y         = 1000#3078 #um
                    # allpix.ELECTRODE_WIDTH      = 900 #um
                    # allpix.ELECTRODE_DEPTH      = 900 #um
                    # allpix.ELECTRODE_THICKNESS  = 0.5 #um
                    # allpix.SILICON_THICKNESS    = 650 #um
                    # allpix.N_PIXELS             = 5 # number of pixels
                    # allpix.N_ROWS               = 5 # number of pixel rows

                    # allpix.USE_CUSTOM_SPECTRUM  = False
                    # allpix.SOURCE_ENERGY        = keV #keV
                    # allpix.SOURCE_POS_X         = 0 #um
                    # allpix.SOURCE_POS_Y         = 0
                    # allpix.SOURCE_POS_Z         = -20000 #um
                    # allpix.SOURCE_TYPE          = "beam"
                    # allpix.BEAM_SHAPE           = "ellipse"
                    # allpix.BEAM_SIZE_X          = np.round(2*np.sqrt(2*np.log(2))*6.06,4) #14.13 #um FWHM
                    # allpix.BEAM_SIZE_Y          = np.round(2*np.sqrt(2*np.log(2))*4.57,4) #14.13 #um FWHM 

                    # allpix.MAX_STEP_LENGTH      = 0.1 #um
                    # allpix.CHARGE_PER_STEP      = charge_per_step
                    allpix.TIME_STEP            = time_step #ns
                    allpix.BIAS_VOLTAGE         = bias_voltage # V
                    allpix.INTEGRATION_TIME     = integration_time # Generally much less is needed, this is for the prototype with incomplete E-field
                    # allpix.INDUCED_DISTANCE     = 1
                    
                    allpix.WORKERS              = 1 # default 4
                    # allpix.USE_ANGLE            = False # Not working i think
                    
                    allpix.ENABLE_DIFFUSION     = enable_diffusion
                    allpix.ENABLE_COULOMB       = enable_coulomb
                    allpix.COULOMB_FIELD_LIMIT  = coulomb_field_limit
                    allpix.COULOMB_DISTANCE_LIMIT = coulomb_distance_limit
                    allpix.PROPAGATE_ELECTRONS = propagate_electrons
                    allpix.PROPAGATE_HOLES = propagate_holes
                    allpix.INCLUDE_MIRROR = include_mirror

                    allpix.SOURCE_ENERGY = keV
                    allpix.NUMBER_OF_PARTICLES     = num_pulses
                    allpix.MAX_CHARGE_GROUPS = max_charge_groups
                    allpix.NUMBER_OF_EVENTS = 1
                    allpix.SPOT_SIZE = spot_size
                    allpix.DEPOSITION_TYPE = "fixed" if spot_size == 0 else "spot"
                    allpix.NUMBER_OF_CHARGES = num_charges

                    # allpix.RECOMBINATION_MODEL = recombination_model #SRH, Auger, Langevin

                    #Custom box/sheet of tungsten
                    # allpix.USE_BOX              = False
                    # allpix.BOX_X                = 7000 #um
                    # allpix.BOX_Y                = 7000 #um
                    # allpix.BOX_Z                = 100 #um

                    # allpix.BOX_POS_X            = allpix.BOX_X/2-7*14 -20000#um
                    # allpix.BOX_POS_Y            = 0 #um
                    # allpix.BOX_POS_Z            = -3*504/2-3*504-279-10000 #um

                    # allpix.E_FIELD_FILE         = "pathToEFieldInitFile"
                    # allpix.W_POTENTIAL_FILE     = "pathToWPotentialInitFile"

                    # allpix.CONFIGURATION_DESCRIPTION = f"d{enable_diffusion}_cr{enable_coulomb}_{"e" if propagate_electrons == 1 else ""}{"h" if propagate_holes == 1 else ""}{"m" if include_mirror == 1 else ""}_{bias_voltage}V_{integration_time}x{time_step}ns".replace(".","p") #replace if there are "." in the path
                    allpix.CONFIGURATION_DESCRIPTION = f"{"e" if propagate_electrons == 1 else ""}{"h" if propagate_holes == 1 else ""}{"m" if include_mirror == 1 else ""}_{bias_voltage}V_{integration_time}x{time_step}ns_{num_charges}-{spot_size}um_{coulomb_distance_limit}cm_{coulomb_field_limit}Vcm_{i}".replace(".","p") #replace if there are "." in the path

                    allpix.OUTPUT_FOLDER = output_folder
                    allpix.OUTPUT_FOLDER += f"/{allpix.CONFIGURATION_DESCRIPTION}"
                    allpix.OUTPUT_FILE = allpix.OUTPUT_FOLDER.replace("output/", "") + "/data.root"

                    print(f"Running sim with parameters: {allpix.CONFIGURATION_DESCRIPTION}")
                    
                    start_time = time.time()
                    allpix.run_sim()
                    end_time = time.time()

                    # Read the RMS Plots
                    filePath = "output/modules.root" # when run from allpix-squared directory
                    print(f'Reading data from {filePath}')
                    file = ROOT.TFile(filePath) # read the file from the specified file path

                    # Copy the modules.root file to the specified output folder
                    # output_file_path = os.path.join(allpix.OUTPUT_FOLDER, f"modules_{allpix.CONFIGURATION_DESCRIPTION}.root")
                    # file.Copy(file)
                    shutil.copy(filePath,allpix.OUTPUT_FOLDER)
                    print(f"Copied {filePath} to {allpix.OUTPUT_FOLDER}")

                    # Copy and Read the log file
                    shutil.copy("mi_sim/log.txt",allpix.OUTPUT_FOLDER)
                    print(f"Copied log.txt to {allpix.OUTPUT_FOLDER}")

                    # Extract InteractivePropagation time from log.txt
                    interactive_time = None
                    try:
                        with open("mi_sim/log.txt", "r") as logf:
                            for line in reversed(logf.readlines()):
                                if "Module InteractivePropagation:DetectorModel took" in line:
                                    # Example: |16:06:26.819|    (INFO)  Module InteractivePropagation:DetectorModel took 2.68764s
                                    interactive_time = line.strip().split("took")[-1].strip()
                                    print(f"Extracted InteractivePropagation time as {interactive_time}")
                                    break
                    except Exception as e:
                        print(f"Could not extract InteractivePropagation time: {e}")

                    detectorName = "DetectorModel"
                    try:
                        dir = file.Get("InteractivePropagation").Get(detectorName) # Access the plots from InteractivePropagation
                        total_multigraph = dir.Get("rms_total_graph")
                        e_multigraph = dir.Get("rms_e_graph")

                        total_graph_list = total_multigraph.GetListOfGraphs()
                        e_graph_list = e_multigraph.GetListOfGraphs()

                        

                        rms_object = {
                            "name": allpix.CONFIGURATION_DESCRIPTION,
                            "repeat_index": i,
                            "num_charges":num_charges,
                            "spot_size": spot_size,
                            "bias_voltage":bias_voltage,
                            "coulomb_field_limit":coulomb_field_limit,
                            "coulomb_distance_limit": coulomb_distance_limit,
                            "enable_diffusion": enable_diffusion,
                            "enable_coulomb": enable_coulomb,
                            "propagate_electrons": propagate_electrons,
                            "propagate_holes": propagate_holes,
                            "include_mirror": include_mirror,
                            "max_charge_groups": max_charge_groups,
                            "sim_time": end_time - start_time,
                            "interactive_time": interactive_time,
                            "charge_per_step": charge_per_step,
                            "time_step": time_step,
                            "t_rms": np.array(total_graph_list[0].GetX()),
                            "e_rms": np.array(total_graph_list[0].GetY()),
                            "h_rms": np.array(total_graph_list[1].GetY()),
                            "e_x_rms": np.array(e_graph_list[0].GetY()),
                            "e_y_rms": np.array(e_graph_list[1].GetY()),
                            "e_z_rms": np.array(e_graph_list[2].GetY()),
                            }

                        # rms_object = {
                        #     "name": allpix.CONFIGURATION_DESCRIPTION,
                        #     "keV":keV,
                        #     "bias_voltage":bias_voltage,
                        #     "coulomb_field_limit":coulomb_field_limit,
                        #     "enable_diffusion": enable_diffusion,
                        #     "enable_coulomb": enable_coulomb,
                        #     "max_charge_groups": max_charge_groups,
                        #     "sim_time": end_time - start_time,
                        #     "charge_per_step": charge_per_step,
                        #     "time_step": time_step,
                        #     "t_rms": np.array(total_graph_list[0].GetX()),
                        #     "e_rms": np.array(total_graph_list[0].GetY()),
                        #     "h_rms": np.array(total_graph_list[1].GetY()),
                        #     "e_x_rms": np.array(e_graph_list[0].GetY()),
                        #     "e_y_rms": np.array(e_graph_list[1].GetY()),
                        #     "e_z_rms": np.array(e_graph_list[2].GetY())
                        #     }
                        
                        output_list.append(rms_object)
                    except:
                        print("Failed to read the rms plots from InteractivePropagation")

                    # TODO: Read the other data (more convenient here)

# Ensure the output folder exists
if not os.path.exists(output_object_folder):
    os.makedirs(output_object_folder)

# Define the output file path
output_file_path = os.path.join(output_object_folder, "output_lists.pkl")

# Write the output_lists dictionary to the file
with open(output_file_path, "wb") as f:
    pickle.dump(output_list, f)

print(f"Output lists saved to {output_file_path}")

# Copy this script (multi_sim.py) to the output directory
shutil.copy(__file__, os.path.join(output_object_folder, "multi_sim.py"))

total_end_time = time.time()

print(f'Total time: {total_end_time-total_start_time}')

