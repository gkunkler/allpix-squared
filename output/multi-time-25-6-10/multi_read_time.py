import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import matplotlib.cm as cm

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

# Primary values from rms_object
num_charges_values = []
coulomb_distance_limits = []
time_step_values = []
# Secondary values from rms_object
sim_time_values = []
interactive_time_values = []
# Tertiary values from RMS plots
e_rms_final = []

for rms_object in data:

    # if rms_object['include_mirror']:
    if rms_object['name'][1] == "m":
        continue

    num_charges_values.append(rms_object['num_charges'])
    coulomb_distance_limits.append(rms_object['coulomb_distance_limit'])
    sim_time_values.append(rms_object['sim_time'])
    time_step_values.append(rms_object['time_step'])
    e_rms_final.append(rms_object['e_rms'][-1])

    if 'ms' in rms_object['interactive_time']:
        interactive_time = float(rms_object['interactive_time'][0:-2])
    else:
        interactive_time = float(rms_object['interactive_time'][0:-1])*1000
        
        
    interactive_time_values.append(interactive_time) # ms

num_charges_values = np.array(num_charges_values)
coulomb_distance_limits = np.array(coulomb_distance_limits)
interactive_time_values = np.array(interactive_time_values)
sim_time_values = np.array(sim_time_values)
time_step_values = np.array(time_step_values)

# Identify unique combinations of primary variables
primary_keys = list(zip(num_charges_values, coulomb_distance_limits, time_step_values))

unique_keys = []
key_indices = {}

for idx, key in enumerate(primary_keys):
    if key not in key_indices:
        unique_keys.append(key)
        key_indices[key] = []
    key_indices[key].append(idx)

# Prepare arrays for averaged results
avg_num_charges = []
avg_coulomb_distance_limits = []
avg_time_step_values = []
avg_sim_time_values = []
avg_e_rms_final = []
avg_interactive_time_values = []

for key in unique_keys:
    indices = key_indices[key]
    # Average t_rms and e_rms element-wise (assume all t_rms/e_rms for duplicates are the same shape)
    avg_e_rms_final.append(np.mean([e_rms_final[i] for i in indices]))
    avg_num_charges.append(key[0])
    avg_coulomb_distance_limits.append(key[1])
    avg_time_step_values.append(key[2])
    avg_interactive_time_values.append(np.mean([interactive_time_values[i] for i in indices]))

# Convert to numpy arrays
num_charges_values = np.array(avg_num_charges)
coulomb_distance_limits = np.array(avg_coulomb_distance_limits)
time_step_values = np.array(avg_time_step_values)
sim_time_values = np.array(avg_sim_time_values)
e_rms_final = np.array(avg_e_rms_final)
interactive_time_values = np.array(avg_interactive_time_values)



unique_time_steps = np.unique(time_step_values)
unique_coulomb_limits = np.unique(coulomb_distance_limits)
colors = cm.viridis(np.linspace(0, 1, len(unique_coulomb_limits)))

fig, axes = plt.subplots(1, len(unique_time_steps), figsize=(6 * len(unique_time_steps), 5), sharey=True)

if len(unique_time_steps) == 1:
    axes = [axes]

for ax, t_step in zip(axes, unique_time_steps):
    for idx, c_limit in enumerate(unique_coulomb_limits):
        mask = (time_step_values == t_step) & (coulomb_distance_limits == c_limit)
        if np.any(mask):
            ax.plot(
                num_charges_values[mask],
                interactive_time_values[mask],
                marker='o',
                label=f'Coulomb limit: {c_limit}',
                color=colors[idx]
            )
    ax.set_title(f'Time step: {t_step} ns')
    ax.set_xlabel('Num Charges')
    ax.grid(True)
    ax.legend()
axes[0].set_ylabel('Simulation Time [ms]')
fig.suptitle("Simulation Time Spent on InteractivePropagation")

fig2, axes2 = plt.subplots(1, len(unique_time_steps), figsize=(7 * len(unique_time_steps), 5), sharey=True)

if len(unique_time_steps) == 1:
    axes2 = [axes2]
for ax, t_step in zip(axes2,unique_time_steps):
    mask = (time_step_values == t_step)
    for jdx, n_charge in enumerate(np.unique(num_charges_values)):
        submask = mask & (num_charges_values == n_charge)
        if np.any(submask):
            ax.plot(
                coulomb_distance_limits[submask],
                e_rms_final[submask],
                marker='o',
                label=f'Num charges: {n_charge}'
            )
    ax.set_xlabel('Coulomb Distance Limit [cm]')
    ax.set_title(f'Time step: {t_step} ns')
    ax.grid(True)
    ax.legend()
axes2[0].set_ylabel('Final RMS [mm]')
fig2.suptitle("Final RMS Dependence on Coulomb Distance Limit")

plt.tight_layout()
plt.show()
