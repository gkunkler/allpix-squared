
import matplotlib.pyplot as plt
import numpy as np

max_charge_groups = [1000,900,800,600]
av_charge_per_step = [46,50,57,72]
time = np.array([37007,30968,23382,15821])
plt.plot(max_charge_groups, time*0.001, marker='o', label="Without Distance Limit")

max_charge_groups = [500,600,800,900,1000,1200]
av_charge_per_step = [80,72,57,50,46,39]

time = np.array([10539,16591,23379,31696,46669,49595])
plt.plot(max_charge_groups, time*0.001, marker='^', label="Distance Limit of 40um")

time = np.array([8810,14083,17780,24049,32196,37595])
plt.plot(max_charge_groups, time*0.001, marker='*', label="Distance Limit of 4um")

time = np.array([4171,5126,10398,12135,14325,16877])
plt.plot(max_charge_groups, time*0.001, marker='+', label="Distance Limit of 0.4um")

plt.xlabel('Max Charge Groups')
plt.ylabel('Time [s]')
plt.ylim([0,50])
plt.title('Time vs Max Charge Groups')
plt.grid(True)
plt.legend()
plt.show()