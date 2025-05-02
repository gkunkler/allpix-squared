#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 21:08:03 2024

@author: rickard
"""
import os
import pickle

class AllpixObject:
    def __init__(self):
        self.LOG_LEVEL = "STATUS"
        self.LOG_FORMAT = "DEFAULT"

        self.PATH_TO_SCRIPT = "mi_sim/"
        self.CONFIG_FILE = "multi_simulation.conf"
        self.OUTPUT_FILE = "multi_output.root"

        self.DETECTOR_FILE = "detector.conf"
        self.NUMBER_OF_EVENTS = 1
        self.OUTPUT_FOLDER = "output/multi/raw"
        self.SOURCE_ENERGY = 10 #keV
        self.SOURCE_POS_X = None #um
        self.SOURCE_POS_Y = None #um
        self.SOURCE_POS_Z = None #um
        self.SOURCE_TYPE = None
        self.BEAM_SHAPE = None
        self.BEAM_SIZE_X = None #um FWHM
        self.BEAM_SIZE_Y = None #um FWHM 
        self.BEAM_DIR_X = None
        self.BEAM_DIR_Y = None
        self.BEAM_DIR_Z = None
        self.NUMBER_OF_PARTICLES = 1 #particles per event
        self.MAX_STEP_LENGTH = 0.1
        
        
        self.INDUCED_DISTANCE = 1
        self.PIXEL_SIZE_X         = None #um
        self.PIXEL_SIZE_y         = None #um
        self.ELECTRODE_WIDTH      = None #um
        self.ELECTRODE_DEPTH      = None #um
        self.ELECTRODE_THICKNESS  = None #um
        self.SILICON_THICKNESS    = None #um
        self.N_PIXELS             = None # number of pixels
        self.N_ROWS               = None # number of pixel rows
        self.USE_BOX = False
        self.BOX_X = None
        self.BOX_Y = None
        self.BOX_Z = None
        self.BOX_POS_X = None
        self.BOX_POS_Y = None
        self.BOX_POS_Z = None
        self.USE_ANGLE = False
        self.ROT_X = None
        self.ROT_Y = None
        self.ROT_Z = None
        self.SOURCE_FILE = "macro_multi.mac"
        self.E_FIELD_FILE = None
        self.W_POTENTIAL_FILE = "weighting_potential3.apf"
        self.BOX_STEP_SIZE = None
        
        
        self.E_FIELD_MAPPING = "PIXEL_FULL"
        self.INTERACTIVE_MODULE = "[Ignore]"
        self.TRANSIENT_MODULE = "[TransientPropagation]"
        self.RECOMBINATION_MODEL = "none"

        self.BIAS_VOLTAGE = 0
        self.MAX_CHARGE_GROUPS = 1000
        self.CHARGE_PER_STEP = 1
        self.INTEGRATION_TIME = 20 #ns
        self.TIME_STEP = 0.2 #ns
        self.COULOMB_FIELD_LIMIT = 5e5 #V/cm
        self.ENABLE_DIFFUSION = 1
        self.ENABLE_COULOMB = 1
        self.PROPAGATE_ELECTRONS = 1
        self.PROPAGATE_HOLES = 0
        self.INCLUDE_MIRROR = 0
        # self.MOBILITY_E = 1000 #cm*cm/V/s
        # self.MOBILITY_H = 100 #cm*cm/V/s
        self.RELATIVE_PERMITIVITY = 11.7 #10.2 for CZT, 11.7 for Si
        
        self.WORKERS = 4
        
        # THe following variables are not included in the file only used in 
        # this class
        self.CONFIGURATION_DESCRIPTION = ""
        self.OUTPUT_FOLDER_TEMP = self.OUTPUT_FOLDER + "_{}"        
        self.FOLDER_COUNTER = 1
        self.excluded = ["OUTPUT_FOLDER_TEMP", "FOLDER_COUNTER", "excluded", "CONFIGURATION_DESCRIPTION"]
        
    def check_folder(self):
        if os.path.exists(self.OUTPUT_FOLDER):
            if self.FOLDER_COUNTER > 1:
                self.OUTPUT_FOLDER = self.OUTPUT_FOLDER.strip("_{}".format(self.FOLDER_COUNTER-1))
            self.OUTPUT_FOLDER = self.OUTPUT_FOLDER + "_{}".format(self.FOLDER_COUNTER)
            self.FOLDER_COUNTER += 1
            self.check_folder()
        else:
            os.makedirs(self.OUTPUT_FOLDER)
            
    def saveSimulationParameters(self, file = None):
        if file is None:
            file = r"{}".format(self.OUTPUT_FOLDER+f"/parameters.txt")
        variables = {attr:value for (attr,value) in self.__dict__.items()}
        
        
        with open(file, 'wb') as f:
            pickle.dump(variables,f)
            # for attr, value in self.__dict__.items():
            #     f.write(f"{attr}, {value}\n")
        # print("---- Parameters saved ----")
        
    def run_sim(self):
        
        self.check_folder()
        self.saveSimulationParameters()

        # self.TRANSIENT_MODULE = "[Ignore]" if self.ENABLE_COULOMB else "[TransientPropagation]"
        self.TRANSIENT_MODULE = "[Ignore]"
        self.INTERACTIVE_MODULE = "[InteractivePropagation]" # if self.ENABLE_COULOMB else "[Ignore]"

        self.ENABLE_DIFFUSION = 1 if self.ENABLE_DIFFUSION else 0
        self.ENABLE_COULOMB = 1 if self.ENABLE_COULOMB else 0

        self.MULTITHREADING = "false" if self.WORKERS < 2 else "true"
        
        # Change the source
        
        with open(self.PATH_TO_SCRIPT + self.SOURCE_FILE, 'r') as f:
            file_data = f.read()
            
            if self.USE_ANGLE:
                file_data = file_data.replace("#", "")
            for variable in vars(self):
                if variable not in self.excluded:
                    file_data = file_data.replace(variable, str(vars(self)[variable]))
            self.SOURCE_FILE = self.SOURCE_FILE.replace(".mac", "_temp.mac")
            with open(self.PATH_TO_SCRIPT + self.SOURCE_FILE, 'w') as file:
                file.write(file_data)
            
        # Change the detector
        
        # with open("micronDetector.conf", 'r') as f:
        #     file_data = f.read()
            
        # for variable in vars(self):
        #     if variable not in self.excluded:
        #         file_data = file_data.replace(variable, str(vars(self)[variable]))
        
        # with open('micronDetector_temp.conf', 'w') as file:
        #     file.write(file_data)
        # with open(self.OUTPUT_FOLDER+'/micronDetector_temp.conf', 'w') as file:
        #     file.write(file_data)
        
        # # Change the box
        # if self.USE_BOX:
        #     self.USE_BOX = "sheet"
        #     with open("detector.conf", 'r') as f:
        #         file_data = f.read()
                
        #     file_data = file_data.replace("#", "")
                
        #     for variable in vars(self):
        #         if variable not in self.excluded:
        #             file_data = file_data.replace(variable, str(vars(self)[variable]))
            
        #     with open('detector_temp.conf', 'w') as file:
        #         file.write(file_data)
        #     with open(self.OUTPUT_FOLDER+'/detector_temp.conf', 'w') as file:
        #         file.write(file_data)
        # else:
            
        #     with open("detector.conf", 'r') as f:
        #         file_data = f.read()
        
        #         file_data = file_data.replace('USE_BOX', 'Ignore')
        #     for variable in vars(self):
        #         if variable not in self.excluded:
        #             file_data = file_data.replace(variable, str(vars(self)[variable]))
            
        #     with open('detector_temp.conf', 'w') as file:
        #         file.write(file_data)
        #     with open(self.OUTPUT_FOLDER+'/detector_temp.conf', 'w') as file:
        #         file.write(file_data)
                
        # self.DETECTOR_FILE = self.DETECTOR_FILE.replace(".conf", "_temp.conf")
            
        # Change the configuration

        with open(self.PATH_TO_SCRIPT + self.CONFIG_FILE, 'r') as f:
            file_data = f.read()

        for variable in vars(self):
            if variable not in self.excluded:
                file_data = file_data.replace(variable, str(vars(self)[variable]))

        self.CONFIG_FILE = self.CONFIG_FILE.replace(".conf", "_temp.conf")
        
        with open(self.PATH_TO_SCRIPT + self.CONFIG_FILE, 'w') as file:
            file.write(file_data)
        with open(self.OUTPUT_FOLDER+f"/configuration.conf", 'w') as file:
            file.write(file_data)

        # Run it

        # print(f"Running command: {"allpix -c "+self.PATH_TO_SCRIPT + self.CONFIG_FILE}")
        
        if self.WORKERS > 1:
            os.system("bin/allpix -c " + self.PATH_TO_SCRIPT + self.CONFIG_FILE+" -j {}".format(self.WORKERS))
        else:
            os.system("bin/allpix -c " + self.PATH_TO_SCRIPT + self.CONFIG_FILE)

        
if __name__ == "__main__":
    allpix = AllpixObject()
    allpix.run_sim()






























