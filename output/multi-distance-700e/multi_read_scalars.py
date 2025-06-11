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

def average_y_by_x(x, y):
    """
    Given arrays x and y, returns arrays of unique x values and the average y for each unique x.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    unique_x = np.unique(x)
    avg_y = np.array([np.mean(y[x == ux]) for ux in unique_x])
    return unique_x, avg_y

def generate_title(name, units, value = ""):
    if value == "":
        return f'{name}{f' [{units}]' if not units == '' else ''}' 
    else:
        return f'{name}: {value}{f' {units}' if not units == '' else ''}' 

# Read the output_lists.pkl file
# output_file_path = "output/multi/output_lists.pkl"
output_file_path = "output_lists.pkl"
with open(output_file_path, "rb") as f:
    data = pickle.load(f)

# Arrays to store the scalar values from 
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
e_rms_final_values = []
e_x_rms_initial_slopes = []

for rms_object in data:

    spot_sizes.append(rms_object['spot_size'])
    num_charges_values.append(rms_object['num_charges'])
    coulomb_field_limits.append(rms_object['coulomb_field_limit'])
    coulomb_distance_limits.append(rms_object['coulomb_distance_limit'])
    enable_diffusion_values.append(bool(rms_object['enable_diffusion']))
    enable_coulomb_values.append(bool(rms_object['enable_coulomb']))
    include_mirror_values.append(bool(rms_object['include_mirror']))
    propagate_electrons_values.append(bool(rms_object['propagate_electrons']))
    propagate_holes_values.append(bool(rms_object['propagate_holes']))
    max_charge_group_values.append(rms_object['max_charge_groups'])
    sim_time_values.append(rms_object['sim_time'])
    charge_per_step_values.append(rms_object['charge_per_step'])
    max_charge_groups_values.append(rms_object['max_charge_groups'])
    time_step_values.append(rms_object['time_step'])
    e_rms_final_values.append(rms_object['e_rms'][-1])
    e_x_rms_initial_slopes.append(rms_initial_slope(rms_object['t_rms'],rms_object['e_x_rms'])[0])

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
include_mirror_values = np.array(include_mirror_values)
propagate_electrons_values = np.array(propagate_electrons_values)
propagate_holes_values = np.array(propagate_holes_values)
max_charge_group_values = np.array(max_charge_group_values)
sim_time_values = np.array(sim_time_values)
charge_per_step_values = np.array(charge_per_step_values)
time_step_values = np.array(time_step_values)
e_rms_final_values = np.array(e_rms_final_values)
e_x_rms_initial_slopes = np.array(e_x_rms_initial_slopes)

# Generalize the arrays
x_values = coulomb_distance_limits
x_name = "Coulomb Distance Limit"
x_units = "cm"
# x_values = num_charges_values
# x_name = "Number of Electrons"
# x_units = ""

# y_values = e_rms_final_values
# y_name = "Final RMS Values"
# y_units = "mm"
y_values = interactive_time_values
y_name = "Time Spent on InteractivePropagation"
y_units = "ms"

line_values = num_charges_values # Separated by color in each subplot
line_name = "Number of Electrons"
line_units = ""
# line_values = coulomb_distance_limits # Separated by color in each subplot
# line_name = "Coulomb Distance Limit"
# line_units = "cm"

filter_1_values = time_step_values # separates into horizontal subplots (can be filtered)
filter_1 = [0.1]
filter_1_name = "Time Step"
filter_1_units = "ns"
filter_2_values = include_mirror_values # separates into vertical subplots (can be filtered)
filter_2 = [0]
filter_2_name = "Mirror"
filter_2_units = ""

markers = ['o', 's','^', '*', 'D', '+', 'x']

fig, axes = plt.subplots(len(filter_2), len(filter_1), figsize=(6 * len(filter_1), 5 * len(filter_2)), sharey=True, sharex=True)

if len(filter_1) == 1 and len(filter_2) == 1:
    axes = np.array([[axes]])

both_indices = np.array([[i,j] for j in range(len(filter_2)) for i in range(len(filter_1))]).transpose()
both_filters = np.array([[value1, value2] for value2 in filter_2 for value1 in filter_1]).transpose()
for ax, i, j, filter_1_value, filter_2_value in zip(axes.flatten(), both_indices[0], both_indices[1], both_filters[0], both_filters[1]):
    print(f'({i}, {j}): {filter_1_value}, {filter_2_value}')
    for idx_line, line_value in enumerate(np.unique(line_values)):
        print(f'{idx_line}, {line_value}')
        mask = (filter_1_values == filter_1_value) & (filter_2_values == filter_2_value) & (line_values == line_value)
        if np.any(mask):
            legend_value = generate_title(line_name, line_units, value = line_value)
            ax.scatter(x_values[mask], y_values[mask], marker = markers[idx_line], label=legend_value)
            avg_x_values, avg_y_values = average_y_by_x(x_values[mask], y_values[mask])
            ax.plot(avg_x_values, avg_y_values, marker = 'none', color = 'k', linestyle = '--')

    filter_1_title = generate_title(filter_1_name, filter_1_units, value = filter_1_value) 
    filter_2_title = generate_title(filter_2_name, filter_2_units, value = filter_2_value) 

    ax.set_title(f'{filter_1_title}, {filter_2_title}')
    ax.set_xscale('log')
    ax.grid(True)
    ax.legend()

x_label = generate_title(x_name, x_units)
y_label = generate_title(y_name, y_units)
fig.supxlabel(x_label)
fig.supylabel(y_label)


fig.suptitle("")

plt.tight_layout()

plt.savefig('image.png', dpi=200)

plt.show()


