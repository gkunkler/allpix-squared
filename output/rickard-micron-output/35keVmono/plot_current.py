import ROOT
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

# This file plots some histograms, scatter, and current plots
# See plot_hist.py for a more detailed look at the histograms

ROOT.gSystem.Load("../../../lib/libAllpixObjects.so") # Load the dictionaries for Allpix objects

# File Input
filePath = "16-perStep_withCR_multithread_1e-06limit/35keV/250V/m250um/data.root"

print(f'Reading data from {filePath}')
file = ROOT.TFile(filePath) # read the file from the specified file path

# Get the ROOT tree objects
detectorName = "detector1"
try:
    pxcTree = file.Get("PixelCharge")
    pxcBranch = pxcTree.GetBranch(detectorName)
except:
    print("Found no PixelCharge objects in root file")
    quit()
   

# Master object to separate events
all_events = []

# Loop over events in tree (same number of event for each)
num_events = pxcTree.GetEntries()
for iEvent in range(num_events):

    event_data = []
    
    # Define objects for each event
    dcGPos = {'x': [], 'y': [], 'z': [], 'q':[], 'type':[]} # Deposited Charges Global Positions
    pcGPos = {'x': [], 'y': [], 'z': [], 'q':[], 'type':[]} # Propagated Charges Global Positions
    pcPulse = {'q': [], 'x': [], 'y': [], 't': []} # Pulse of charge from propagated charges (contains global x and y)
    pxcPulse = {'q': [], 'i_x': -1, 'i_y': -1, 't_bin': -1} # Pulse of charge from pixel charges (stores the pixel indices as well)
    pxhPulse = {'q': [], 't': [], 'i_x': [], 'i_y': []} # Pulse of charge from pixel hits (stores the pixel indices as well)

    pxcTree.GetEntry(iEvent)
    pxcBranch = pxcTree.GetBranch(detectorName) # May need to get the branch here if they need to be updated for each event

    # Get ROOT branch as python vector (object?)
    br_pxc = getattr(pxcTree, pxcBranch.GetName())

    # Get charge pulse from pixel charge
    for pxc in br_pxc:
        
        if pxc.getCharge() == 0:
            continue

        pxcPulse = {  # Create a new dictionary in each loop iteration
            'q': list(pxc.getPulse()).copy(), # the copy() prevents the array of data from changing after it is added to the dictionary
            'i_x': pxc.getIndex().x(),
            'i_y': pxc.getIndex().y(),
            't_bin': pxc.getPulse().getBinning(),
            'total_q': pxc.getCharge()
        }    

        event_data.append(pxcPulse)

    all_events.append(event_data)

channels = [5,6,7,8,9] # 7,8,9 are middle of detector

pxc_fig, pxc_ax = plt.subplots(1,len(channels), sharey=True, sharex=True, figsize=(20,5))

q_fig, q_ax = plt.subplots(dpi=150)
charges_collected = []
channels_for_charge = []

q_avg = np.zeros(len(channels))

print(f'number of events: {len(all_events)}')
num_active_events = 0
for event_data in all_events:

    if len(event_data) == 0: # Skip events with no measured pulses
        continue

    # Skip events with net zero charge
    total = 0
    for pxcPulse in event_data:
        total += pxcPulse["total_q"]
    if total == 0 or total < 1:
        continue

    num_active_events += 1 # Increment non-filtered events

    for pxcPulse in event_data:
        
        # Get the associated pixel (Note that (1,1) is the center for a 3x3 grid)
        i = pxcPulse['i_x']

        print(f"Pixel Pulse ({i},{0}): {pxcPulse["total_q"]}")

        i -= channels[0] # Start channel indexing at 0 for subplots
    
    avg_set = False # Whether these arrays have been defined a length yet
    pxc_current = []
    pxc_avg = []
    total_charge = np.zeros(len(channels))

    current_integral = np.zeros(len(channels))
    current_integral_e = np.zeros(len(channels))

    bin_width = pxcPulse['t_bin']
    t = np.linspace(0, (len(pxcPulse['q']) - 1) * bin_width, len(pxcPulse['q']))

    # Determine rebinning (must loop through each event and pixel to get the longest q array)
        # do_rebinning = True # toggle rebinning behavior
        # default_bin_width = 1 # Used if do_rebinning is set to True
        # num_bins = 0
        # for pxcPulse in event_data:
        #     if do_rebinning:
        #         new_num_bins = math.ceil(len(pxcPulse['q']) * pxcPulse['t_bin'] / default_bin_width)
        #         num_bins = max(num_bins, new_num_bins)
        #     else:
        #         num_bins = max(num_bins, len(pxcPulse['q']))
        #         bin_width = pxcPulse['t_bin']
        # if do_rebinning: bin_width = default_bin_width
        # t = np.arange(0, num_bins * bin_width, bin_width)

    for pxcPulse in event_data:

        i = pxcPulse['i_x']
        if i in channels:
            i -= channels[0]
        else:
            continue # Skip pulses detected outside of selected channels
        
        # Preform the rebinning
        # unbinned_q = pxcPulse['q'] # e
        # if do_rebinning:
        #     q = np.zeros(num_bins)
        #     for j in range(len(unbinned_q)):
        #         index = math.floor(j * pxcPulse['t_bin'] / bin_width)
        #         if index < num_bins: q[index] += unbinned_q[j]
        # else:
        #     q = np.array(unbinned_q)
        if len(t) > len(pxcPulse['q']):
            q = np.zeros(len(t))
            q[0:len(pxcPulse['q'])] = np.array(pxcPulse['q'])
        elif len(t) < len(pxcPulse['q']):
            q = np.array(pxcPulse['q'])[0:len(t)]
        else:
            q = np.array(pxcPulse['q'])
            
        if len(q) != len(t):
            print(q)
            print(t)
            exit()

        # Add to the total charge for output
        total_charge[i] += sum(q) # e
        q_avg[i] += pxcPulse['total_q'] # e
        charges_collected.append(pxcPulse['total_q'])
        channels_for_charge.append(channels[i])

        # Set size of current and average arrays to the length of the q array
        if not avg_set:
            if len(pxc_current) == 0: pxc_current = np.zeros((len(channels),len(t)))
            if len(pxc_avg) == 0: pxc_avg = np.zeros((len(channels),len(t)))
            avg_set = True

        # Convert charge to current by dividing by bin width (dq/dt) 
        #  no added negative sign since current is from holes
        pxc_current[i] = 1 * q / bin_width # * (1.609*10**-19) * (10**9) * (10**6) # e/ns * 1.609e-19 C/e * 10e9 ns/s * 10e6 uA/A = uA

        pxc_ax[i].plot(t, pxc_current[i], linewidth=0.4, alpha=min(0.8, max(0.5, 3/num_events))) # For plotting each event with low opacity

        # Add to the average for each pixel
        pxc_avg[i] += pxc_current[i] # [pxc_avg[i][j] + pxc_current[i][j] for j in range(len(t))] # Division happens later

        current_integral[i] = bin_width*sum(pxc_current[i]) #/ (10**9) / (10**6) * -1 # ns*uA * 1s/1e9ns * 1A/1e6uA = C
        current_integral_e[i] = current_integral[i] #/ (1.609*10**-19)

    # plt.show()


    # # Get the cumulative charge curves
    # pxc_cum = np.zeros((num_pixels_x, num_pixels_y, num_bins))
    # pxc_cum_fig, pxc_cum_ax = plt.subplots(1,num_pixels_y)
    # for i_y in range(num_pixels_y):
    #     for i_t in range(num_bins):
    #         previous_value = 0 if i_t == 0 else pxc_cum[i_x][i_y][i_t-1]
    #         pxc_cum[i_x][i_y][i_t] = previous_value + bin_width * pxc_avg[i_x][i_y][i_t] / (10**9) / (10**6) / (1.609*10**-19) # ns*uA * 1s/1e9ns * 1A/1e6uA * 1e/1.609e19C = e
    #     pxc_cum_ax[i_x][i_y].plot(t,pxc_cum[i_x][i_y], linestyle='-', color='k')

    # Take the integral to help determine physical interpretation
    # The sum of all charges is close to half the depositied charges, so probably total electrons that make it to the plate.
    # They are just split up based on the weighting potential
    #print(f'Time bin width: {bin_width} ns')
    print(f'=== Results from Pixel Pulse ===')
    print(f'Sum of all charges in center pixel pulse is {total_charge[1]}e (Average per event: {total_charge[1]/num_events})')
    print(f'Average charge per event (e):\n{total_charge/num_events}')
        
    #print(f'Integral of the center pixel current pulse is {current_integral[1][1]}C = {current_integral_e[1][1]}e')
    # print(f'Integrals of each pixel current (C):\n{current_integral}')
    print(f'Integrals of each pixel current (e):\n{current_integral_e}')

    # print(np.shape(pxc_cum))
    # print(f'Final values of cumulative charge in each pixel (e):\n{pxc_cum[:,:,-1]}')
    # print(f'Final slopes (current) in each pixel (e/ns):\n{pxc_current[:,:,-1] / (10**6) / (10**9) / (1.609*10**-19) }') # uA * 1A/1e6 * 1s/1e9ns * 1e/1.609e-19C = e/ns

    # print(f'Sum of integrals over the entire sensor (e): {sum(sum(current_integral_e))}')

    # pxc_fig.suptitle(f'Current pulse in each pixel (averaged over {num_events} events)')
    # pxc_cum_fig.suptitle(f'Cumulative charge detected on each pixel (averaged over {num_events} events)')

    # Determine bounds for the pulse and cumulative charge plots
    # min_t = 0
    # max_t = 10 #max(t)

    # min_I = -2000 # e/ns
    # max_I = 10000

    # max_I = np.max(pxc_current)
    # min_I = np.min(pxc_current)
    # diff_I = max_I-min_I
    # max_I += diff_I*0.1
    # min_I -= diff_I*0.1

    # min_q = np.min(pxc_cum)
    # max_q = np.max(pxc_cum)
    # diff_q = max_q - min_q
    # min_q -= diff_q*0.1
    # max_q += diff_q*0.1

    # for i_y in range(len(pxc_ax)): 
    #     min_y = -0.02
    #     max_y = 0.2 # -1 * min_y if i_x != 1 else 0.2
        
    #     pxc_ax[i_y].set_xlim((min_t, max_t))
    #     pxc_ax[i_y].set_ylim((min_I, max_I)) # The plots might be transposed from the actual orientation

        #min_y = min(pxc_cum[i_x][i_y])-100
        #max_y = max(pxc_cum[i_x][i_y])+100
        #pxc_cum_ax[i_x][i_y].set_xlim((min_x, max_x))
        # pxc_cum_ax[i_x][i_y].set_ylim((min_q, max_q)) # The plots might be transposed from the actual orientation
            
    #pxc_ax[1][0].set_title(f'Current pulse from average of {num_events} events')
    # pxc_ax[num_pixels_x-1][1].set_xlabel(f"Time [ns] (bin_width={bin_width}ns)")
    # pxc_ax[1][0].set_ylabel("Current [uA]")

    # pxc_cum_ax[num_pixels_x-1][1].set_xlabel(f"Time [ns] (bin_width={bin_width}ns)")
    # pxc_cum_ax[1][0].set_ylabel("Charge [e]")

    #pxc_ax[1].set_xlabel(f"Time [ns] (time_step={bin_width})")
    #pxc_ax[1].set_ylabel("Induced Current (dq/dt) [e/ns]")

print(f"Total Events Analyzed: {num_active_events}")

# pxc_avg = pxc_avg/num_active_events
q_avg = q_avg/num_active_events

# Add the average plot    
for i in range(len(pxc_ax)): 

    min_t = 0
    max_t = 65 #max(t)

    min_I = -1000 # e/ns
    max_I = 4000


    pxc_ax[i].set_xlim((min_t, max_t))
    pxc_ax[i].set_ylim((min_I, max_I))

    pxc_ax[i].set_xlabel('t [ns]')

    # pxc_ax[i].plot(t,pxc_avg[i], linestyle='--', color='k', label='Average of all events')
    pxc_ax[i].set_title(f"Channel {channels[i]} (Average of {np.round(q_avg[i],2)}e)")

pxc_ax[0].set_ylabel('Current [e/ns]')

pxc_fig.suptitle("Currents")

q_ax.plot(channels, [350.5,1817.98,3282.4,1727.57,263.97], label='Diffusion only', color='r', linestyle = '--',alpha=0.5)
# q_ax.scatter(channels_for_charge, charges_collected, marker='.', s=0.5, color='k')
q_ax.plot(channels, q_avg, label='Diffusion & CR', color='r')

q_ax.set_xlabel("Channel")
q_ax.set_ylabel("Charge Collected [e]")
q_ax.set_xticks(channels) # TODO
q_fig.suptitle("Average Charge Collected in Each Channel")
q_fig.legend()

plt.show()

# input("Keeping window open...") # Keeps plots visible until user presses enter


