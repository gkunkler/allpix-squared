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

total_start_time = time.time()

ROOT.gSystem.Load("lib/libAllpixObjects.so") # Load the dictionaries for Allpix objects

output_folder = "output/multi/raw"
output_object_folder = "output/multi"
output_list = []

# e_field_mapping = "PIXEL_FULL"
# recombination_model = "srh" #SRH, Auger, Langevin

num_pulses = 100 # Number of particles per event

keVs = [0.5,5,10] # ionization energy = 3.64eV

# charges_per_steps = {0.5:14, 1:28, 3:83, 5:138, 10:275} # 0.1% of total charge (num_pulses*keV*1000/ionization_energy * 0.001) -> approx. 1000 charge groups
charge_per_step = 1

coulomb_field_limit = 4e5 #[5e4,7e4,1e5,3e5,5e5,7e5,1e6,2e6]
# max_charge_groups_list = [1000]
max_charge_groups = 1000

time_step = 0.1

# enable_coulomb = 1

for keV in keVs:
    # charge_per_step = charges_per_steps[keV]

    # for coulomb_field_limit in coulomb_field_limits:
    for enable_diffusion in [0,1]:

        nPulses = num_pulses

        # for max_charge_groups in max_charge_groups_list:
        # for time_step in time_steps:
        for enable_coulomb in [0,1]:
            allpix = AllpixObject()

            allpix.LOG_LEVEL            = "STATUS" #STATUS, INFO, WARNING, DEBUG
            allpix.LOG_FORMAT           = "DEFAULT"
            allpix.DETECTOR_FILE        = "detector.conf"
            allpix.SOURCE_FILE          = "macro_multi.mac"
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
            # allpix.INTEGRATION_TIME     = 20 #ns # Generally much less is needed, this is for the prototype with incomplete E-field
            # allpix.INDUCED_DISTANCE     = 1
            
            allpix.WORKERS              = 1 # default 4
            # allpix.USE_ANGLE            = False # Not working i think
            
            allpix.ENABLE_DIFFUSION     = enable_diffusion
            allpix.ENABLE_COULOMB       = enable_coulomb
            allpix.COULOMB_FIELD_LIMIT  = coulomb_field_limit

            allpix.SOURCE_ENERGY = keV
            allpix.NUMBER_OF_PARTICLES     = nPulses
            allpix.MAX_CHARGE_GROUPS = max_charge_groups
            allpix.NUMBER_OF_EVENTS = 1

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

            allpix.OUTPUT_FOLDER = output_folder
            allpix.OUTPUT_FOLDER += f"/{keV}_d{enable_diffusion}_cr{enable_coulomb}".replace(".","p") #replace if there are "." in the path
            allpix.OUTPUT_FILE = allpix.OUTPUT_FOLDER.replace("output/", "")

            print(f"Running sim with parameters: {keV}_d{enable_diffusion}_cr{enable_coulomb}")
            
            start_time = time.time()
            allpix.run_sim()
            end_time = time.time()

            # Read the RMS Plots
            filePath = "output/modules.root" # when run from allpix-squared directory
            print(f'Reading data from {filePath}')
            file = ROOT.TFile(filePath) # read the file from the specified file path

            detectorName = "DetectorModel"
            try:
                dir = file.Get("InteractivePropagation").Get(detectorName) # Access the plots from InteractivePropagation
                total_multigraph = dir.Get("rms_total_graph")
                e_multigraph = dir.Get("rms_e_graph")

                total_graph_list = total_multigraph.GetListOfGraphs()
                e_graph_list = e_multigraph.GetListOfGraphs()

                rms_object = {
                    "keV":keV,
                    "coulomb_field_limit":coulomb_field_limit,
                    "enable_diffusion": enable_diffusion,
                    "enable_coulomb": enable_coulomb,
                    "max_charge_groups": max_charge_groups,
                    "sim_time": end_time - start_time,
                    "charge_per_step": charge_per_step,
                    "time_step": time_step,
                    "t_rms": np.array(total_graph_list[0].GetX()),
                    "e_rms": np.array(total_graph_list[0].GetY()),
                    "h_rms": np.array(total_graph_list[1].GetY()),
                    "e_x_rms": np.array(e_graph_list[0].GetY()),
                    "e_y_rms": np.array(e_graph_list[1].GetY()),
                    "e_z_rms": np.array(e_graph_list[2].GetY())
                    }
                
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

total_end_time = time.time()

print(f'Total time: {total_end_time-total_start_time}')

