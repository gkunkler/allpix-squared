import pickle
import matplotlib.pyplot as plt
from tools.fitting_functions import rms_initial_slope, rms_fit
import numpy as np

# Read the output_lists.pkl file
output_file_path = "output/multi/output_lists.pkl"
with open(output_file_path, "rb") as f:
    data = pickle.load(f)

# Primary Values from RMS plots
keV_values = []
coulomb_field_limits = []
enable_coulomb_values = []
max_charge_group_values = []
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
    keV_values.append(rms_object['keV'])
    coulomb_field_limits.append(rms_object['coulomb_field_limit'])
    enable_coulomb_values.append(rms_object['enable_coulomb'])
    max_charge_group_values.append(rms_object['max_charge_groups'])
    t_rms.append(np.array(rms_object['t_rms']))
    e_rms.append(np.array(rms_object['e_rms']))
    h_rms.append(np.array(rms_object['h_rms']))
    e_x_rms.append(np.array(rms_object['e_x_rms']))
    e_y_rms.append(np.array(rms_object['e_y_rms']))
    e_z_rms.append(np.array(rms_object['e_z_rms']))
    e_rms_final.append(rms_object['e_rms'][-1])
    e_x_rms_initial_slope.append(rms_initial_slope(rms_object['t_rms'],rms_object['e_x_rms'])[0])

keV_values = np.array(keV_values)
coulomb_field_limits = np.array(coulomb_field_limits)
max_charge_group_values = np.array(max_charge_group_values)
e_rms_final = np.array(e_rms_final)
e_x_rms_initial_slope = np.array(e_x_rms_initial_slope)


# Pick which variables to change in plots
analysis_values = max_charge_group_values
unique_analysis_values = np.unique(analysis_values)
num_analysis_values = len(unique_analysis_values)
analysis_name = "Max Charge Groups"
analysis_units = ""

analysis_2_values = keV_values
unique_analysis_2_values = np.unique(analysis_2_values)
num_analysis_2_values = len(unique_analysis_2_values)
analysis_2_name = "Energy"
analysis_2_units = "keV"

# Create a rms time plot for all cases

fig_rms_t, axs_rms_t = plt.subplots(1, num_analysis_2_values, figsize=(5*num_analysis_2_values,4))

for i in range(num_analysis_2_values):
    
    indices = [j for j, value in enumerate(analysis_2_values) if value == unique_analysis_2_values[i]]

    if num_analysis_2_values == 1:
        for k in indices:
            axs_rms_t.plot(t_rms[k], e_rms[k],label=f'{analysis_name}: {np.format_float_scientific(analysis_values[k],2)}')
        axs_rms_t.set_xlabel('t [ns]')
        axs_rms_t.set_ylabel('rms [mm]')
        axs_rms_t.set_title(f'{analysis_2_name}: {unique_analysis_2_values[i]} {analysis_2_units}')
        axs_rms_t.grid(True)
        axs_rms_t.legend()
        continue

    for k in indices:
        axs_rms_t[i].plot(t_rms[k], e_rms[k],label=f'{analysis_name}: {np.format_float_scientific(analysis_values[k],2)}')
    axs_rms_t[i].set_xlabel('t [ns]')
    axs_rms_t[i].set_ylabel('rms [mm]')
    axs_rms_t[i].set_title(f'{analysis_2_name}: {unique_analysis_2_values[i]} {analysis_2_units}')
    axs_rms_t[i].grid(True)
    axs_rms_t[i].legend()

fig_rms_t.suptitle('RMS Spread during Propagation')

# Create a set of 2 plots
fig_secondary, axs_secondary = plt.subplots(1, 2, figsize=(12, 6))

for analysis_2_value in unique_analysis_2_values:

    indices = [i for i, value in enumerate(analysis_2_values) if value == analysis_2_value]

    # Left plot: Initial slope vs. Coulomb field limit
    axs_secondary[0].plot(analysis_values[indices], e_x_rms_initial_slope[indices], label=f'{analysis_2_value} {analysis_2_units}')
    # Right plot: Final RMS value vs. Coulomb field limit
    axs_secondary[1].plot(analysis_values[indices], e_rms_final[indices], label=f'{analysis_2_value} {analysis_2_units}')


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

plt.tight_layout()
plt.show()
