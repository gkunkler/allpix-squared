# copy into the output/multi... directory and run from there

import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

def rms_initial_slope(t,f,i_range = [1,3]):

    x_slope = t[i_range[0]:i_range[1]]
    y_slope = f[i_range[0]:i_range[1]] 
    
    slope, intercept, r, p, std_err = linregress(x_slope, y_slope)

    return slope,intercept

# Read the output_lists.pkl file
# output_file_path = "output/multi/output_lists.pkl"
output_file_path = "output_lists.pkl"
with open(output_file_path, "rb") as f:
    data = pickle.load(f)

# Primary Values from RMS plots
spot_sizes = []
num_charges_values = []
coulomb_field_limits = []
coulomb_distance_limits = []
enable_diffusion_values = []
enable_coulomb_values = []
include_mirror_values = []
propagate_electrons_values = []
propagate_holes_values = []
max_charge_group_values = []
sim_time_values = []
interactive_time_values = []
charge_per_step_values = []
max_charge_groups_values = []
time_step_values = []
t_rms = []
e_rms = []
h_rms = []
e_x_rms = []
e_y_rms = []
e_z_rms = []

# Secondary Values from RMS plots
e_rms_final = []
e_x_rms_initial_slope = []

for rms_object in data:
    spot_sizes.append(rms_object['spot_size'])
    num_charges_values.append(rms_object['num_charges'])
    coulomb_field_limits.append(rms_object['coulomb_field_limit'])
    coulomb_distance_limits.append(rms_object['coulomb_distance_limit'])
    enable_diffusion_values.append(rms_object['enable_diffusion'])
    enable_coulomb_values.append(rms_object['enable_coulomb'])
    include_mirror_values.append(rms_object['include_mirror'])
    propagate_electrons_values.append(rms_object['propagate_electrons'])
    propagate_holes_values.append(rms_object['propagate_holes'])
    max_charge_group_values.append(rms_object['max_charge_groups'])
    sim_time_values.append(rms_object['sim_time'])
    
    charge_per_step_values.append(rms_object['charge_per_step'])
    max_charge_groups_values.append(rms_object['max_charge_groups'])
    time_step_values.append(rms_object['time_step'])
    t_rms.append(np.array(rms_object['t_rms']))
    e_rms.append(np.array(rms_object['e_rms']))
    h_rms.append(np.array(rms_object['h_rms']))
    e_x_rms.append(np.array(rms_object['e_x_rms']))
    e_y_rms.append(np.array(rms_object['e_y_rms']))
    e_z_rms.append(np.array(rms_object['e_z_rms']))
    e_rms_final.append(rms_object['e_rms'][-1])
    e_x_rms_initial_slope.append(rms_initial_slope(rms_object['t_rms'],rms_object['e_x_rms'])[0])

    if 'ms' in rms_object['interactive_time']:
        interactive_time = float(rms_object['interactive_time'][0:-2])
    else:
        interactive_time = float(rms_object['interactive_time'][0:-1])*1000
        
        
    interactive_time_values.append(interactive_time) # ms

spot_sizes = np.array(spot_sizes)
num_charges_values = np.array(num_charges_values)
coulomb_distance_limits = np.array(coulomb_distance_limits)
interactive_time_values = np.array(interactive_time_values)
coulomb_field_limits = np.array(coulomb_field_limits)
enable_diffusion_values = np.array(enable_diffusion_values)
enable_coulomb_values = np.array(enable_coulomb_values)
max_charge_group_values = np.array(max_charge_group_values)
sim_time_values = np.array(sim_time_values)
charge_per_step_values = np.array(charge_per_step_values)
time_step_values = np.array(time_step_values)
e_rms_final = np.array(e_rms_final)
e_x_rms_initial_slope = np.array(e_x_rms_initial_slope)

# Pick which variables to change in plots

# analysis_values = coulomb_field_limits
# unique_analysis_values = np.unique(analysis_values)
# num_analysis_values = len(unique_analysis_values)
# analysis_name = "Field Limit"
# analysis_units = "V/cm"

# analysis_2_values = max_charge_group_values
# unique_analysis_2_values = np.unique(analysis_2_values)
# num_analysis_2_values = len(unique_analysis_2_values)
# analysis_2_name = "Max Charge Groups"
# analysis_2_units = ""

# analysis_values = max_charge_group_values
# unique_analysis_values = np.unique(analysis_values)
# num_analysis_values = len(unique_analysis_values)
# analysis_name = "Max Charge Groups"
# analysis_units = ""

# analysis_values = time_step_values
# unique_analysis_values = np.unique(analysis_values)
# num_analysis_values = len(unique_analysis_values)
# analysis_name = "Time Step"
# analysis_units = "ns"

# analysis_2_values = coulomb_field_limits
# unique_analysis_2_values = np.unique(analysis_2_values)
# num_analysis_2_values = len(unique_analysis_2_values)
# analysis_2_name = "Field Limit"
# analysis_2_units = "V/cm"

analysis_values = coulomb_field_limits
unique_analysis_values = np.unique(analysis_values)
num_analysis_values = len(unique_analysis_values)
analysis_name = "Coulomb Field Limit"
analysis_units = "V/cm"
linestyles = ['-','--', '-.', ':']

analysis_2_values = time_step_values
unique_analysis_2_values = np.unique(analysis_2_values)
num_analysis_2_values = len(unique_analysis_2_values)
analysis_2_name = "Time Step"
analysis_2_units = "ns"

# Set a filter on non-analysis (independent) variables to limit
nonanalysis_values = max_charge_group_values
nonanalysis_filter = max_charge_group_values
num_nonanalysis_filter = len(nonanalysis_filter)
nonanalysis_units = " charge groups"

# nonanalysis_values = coulomb_field_limits
# nonanalysis_filter = [5e5]
# num_nonanalysis_filter = len(nonanalysis_filter)
# nonanalysis_units = "V/cm"

# Create a rms time plot for all cases

fig_rms_t, axs_rms_t = plt.subplots(1, num_analysis_2_values, figsize=(5*num_analysis_2_values,4), dpi=150)

for i in range(num_analysis_2_values):
    
    indices = [j for j, value in enumerate(analysis_2_values) if value == unique_analysis_2_values[i] and nonanalysis_values[j] in nonanalysis_filter]

    if num_analysis_2_values == 1:
        for k in indices:
            axs_rms_t.plot(t_rms[k], e_rms[k],linestyle=linestyles[k],label=f'{analysis_name}: {np.format_float_scientific(analysis_values[k],2)} {f"({nonanalysis_values[k]} {nonanalysis_units})" if num_nonanalysis_filter > 1 else ""}')
        axs_rms_t.set_xlabel('t [ns]')
        axs_rms_t.set_ylabel('rms [mm]')
        axs_rms_t.set_title(f'{analysis_2_name}: {unique_analysis_2_values[i]} {analysis_2_units}')
        axs_rms_t.grid(True)
        axs_rms_t.legend()
        continue

    for k in indices:
        axs_rms_t[i].plot(t_rms[k], e_rms[k], linestyle=linestyles[k], label=f'{analysis_name}: {np.format_float_scientific(analysis_values[k],2)} {f"({nonanalysis_values[k]} {nonanalysis_units})" if num_nonanalysis_filter > 1 else ""}')
    axs_rms_t[i].set_xlabel('t [ns]')
    axs_rms_t[i].set_ylabel('rms [mm]')
    axs_rms_t[i].set_title(f'{analysis_2_name}: {unique_analysis_2_values[i]} {analysis_2_units}')
    axs_rms_t[i].grid(True)
    axs_rms_t[i].legend()

# fig_rms_t.suptitle(f'RMS Spread during Propagation {f'({nonanalysis_filter[0]} {nonanalysis_units})' if num_nonanalysis_filter==1 else ''}')

# Create a set of 2 plots
fig_secondary, axs_secondary = plt.subplots(1, 2, figsize=(12, 6))

for analysis_2_value in unique_analysis_2_values:

    indices = [i for i, value in enumerate(analysis_2_values) if value == analysis_2_value]

    for nonanalysis_value in nonanalysis_filter:

        subindices = [i for i in indices if nonanalysis_values[i] == nonanalysis_value]

        # Left plot: Initial slope vs. Coulomb field limit
        axs_secondary[0].plot(analysis_values[subindices], e_x_rms_initial_slope[subindices], label=f'{analysis_2_name}: {analysis_2_value} {analysis_2_units} {f"({nonanalysis_value} {nonanalysis_units})" if num_nonanalysis_filter > 1 else ""}')
        # Right plot: Final RMS value vs. Coulomb field limit
        axs_secondary[1].plot(analysis_values[subindices], e_rms_final[subindices], label=f'{analysis_2_name}: {analysis_2_value} {analysis_2_units} {f"({nonanalysis_value} {nonanalysis_units})" if num_nonanalysis_filter > 1 else ""}')


axs_secondary[0].set_xlabel(f'{analysis_name} [{analysis_units}]')
axs_secondary[0].set_ylabel('Initial Slope [mm/ns]')
axs_secondary[0].set_title(f'Initial Slope vs. {analysis_name}')
axs_secondary[0].grid(True)
axs_secondary[0].legend()
    
axs_secondary[1].set_xlabel(f'{analysis_name} [{analysis_units}]')
axs_secondary[1].set_ylabel('Final RMS Value [mm]')
axs_secondary[1].set_title(f'Final RMS Value vs. {analysis_name}')
axs_secondary[1].grid(True)
axs_secondary[1].legend()

# fig_secondary.suptitle(f'RMS Curve Characterization {f'({nonanalysis_filter[0]} {nonanalysis_units})' if num_nonanalysis_filter==1 else ''}')

plt.tight_layout()
plt.show()
