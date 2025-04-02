import ROOT
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

# This file plots some histograms, scatter, and current plots
# See plot_hist.py for a more detailed look at the histograms

ROOT.gSystem.Load("lib/libAllpixObjects.so") # Load the dictionaries for Allpix objects

# File Input
filePath = "output/test_output.root" # when run from allpix-squared directory
print(f'Reading data from {filePath}')
file = ROOT.TFile(filePath) # read the file from the specified file path

# Set conditionals for 4 potential objects in the output file
containsDC = True
containsPC = True
containsPXC = True
containsPXH = True

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

try:
    pxcTree = file.Get("PixelCharge")
    pxcBranch = pxcTree.GetBranch(detectorName)
    print_output += " PixelCharge"
except:
    containsPXC = False

try:
    pxhTree = file.Get("PixelHit")
    pxhBranch = pxhTree.GetBranch(detectorName)
    print_output += " PixelHit"
except:
    containsPXH = False

if containsDC or containsPC or containsPXC or containsPXH:
    print(print_output)
else:
    print("Found no applicable objects in root file")
    quit()

# Master object to separate events
event_data = {'dc_positions':[], 'pc_positions':[],'pc_pulses':[], 'pxc_pulses':[]}

# Loop over events in tree (same number of event for each)
num_events = pxcTree.GetEntries()
for iEvent in range(num_events):
    
    # Define objects for each event
    dcGPos = {'x': [], 'y': [], 'z': [], 'q':[], 'type':[]} # Deposited Charges Global Positions
    pcGPos = {'x': [], 'y': [], 'z': [], 'q':[], 'type':[]} # Propagated Charges Global Positions
    pcPulse = {'q': [], 'x': [], 'y': [], 't': []} # Pulse of charge from propagated charges (contains global x and y)
    pxcPulse = {'q': [], 'i_x': -1, 'i_y': -1, 't_bin': -1} # Pulse of charge from pixel charges (stores the pixel indices as well)
    pxhPulse = {'q': [], 't': [], 'i_x': [], 'i_y': []} # Pulse of charge from pixel hits (stores the pixel indices as well)

    # Update each tree to the current event
    if containsDC: dcTree.GetEntry(iEvent)
    if containsPC: pcTree.GetEntry(iEvent)
    if containsPXC: pxcTree.GetEntry(iEvent)
    if containsPXH: pxhTree.GetEntry(iEvent)

    # May need to get the branch here if they need to be updated for each event
    if containsDC: dcBranch = dcTree.GetBranch(detectorName)
    if containsPC: pcBranch = pcTree.GetBranch(detectorName)
    if containsPXC: pxcBranch = pxcTree.GetBranch(detectorName)
    if containsPXH: pxhBranch = pxhTree.GetBranch(detectorName)

    # Get ROOT branch as python vector (object?)
    if containsDC: br_dc = getattr(dcTree, dcBranch.GetName())
    if containsPC: br_pc = getattr(pcTree, pcBranch.GetName())
    if containsPXC: br_pxc = getattr(pxcTree, pxcBranch.GetName())
    if containsPXH: br_pxh = getattr(pxhTree, pxhBranch.GetName())

    # Get the locations of deposited charges
    if containsDC:
        for dc in br_dc:
            dcGPos['x'].append(dc.getGlobalPosition().x())
            dcGPos['y'].append(dc.getGlobalPosition().y())
            dcGPos['z'].append(dc.getGlobalPosition().z())
            dcGPos['q'].append(abs(dc.getCharge()))
            dcGPos['type'].append(dc.getType())

    # Get the locations of propagated charges
    if containsPC:
        for pc in br_pc:
            pcGPos['x'].append(pc.getGlobalPosition().x())
            pcGPos['y'].append(pc.getGlobalPosition().y())
            pcGPos['z'].append(pc.getGlobalPosition().z())
            pcGPos['q'].append(abs(pc.getCharge())) # Not the actual charge but the induced charge from each PC
            pcGPos['type'].append(pc.getType())

            pcPulse['q'].append(pc.getCharge())
            pcPulse['x'].append(pc.getGlobalPosition().x())
            pcPulse['y'].append(pc.getGlobalPosition().y())
            #pcPulse['t'].append(pc.getLocalTime())
            pcPulse['t'].append(pc.getGlobalTime())

    # Get charge pulse from pixel charge
    if containsPXC:
        for pxc in br_pxc:
            #print(f'Length of pxc.getPulse(): {len(pxc.getPulse())} (x={pxc.getIndex().x()},y={pxc.getIndex().y()}) bin={pxc.getPulse().getBinning()}')
            pxcPulse = {  # Create a new dictionary in each loop iteration
                'q': list(pxc.getPulse()).copy(), # the copy() prevents the array of data from changing after it is added to the dictionary
                'i_x': pxc.getIndex().x(),
                'i_y': pxc.getIndex().y(),
                't_bin': pxc.getPulse().getBinning(),
            }    
            #print(len(pxcPulse['q']))
            #pxcPulse['q'] = pxc.getPulse()
            #pxcPulse['i_x'] = pxc.getIndex().x()
            #pxcPulse['i_y'] = pxc.getIndex().y()
            #pxcPulse['t_bin'] = pxc.getPulse().getBinning()
            event_data['pxc_pulses'].append(pxcPulse)
            #print(len(pxcPulse['q']))
            #print(len(event_data['pxc_pulses'][0]['q']))
            #pxcPulse = {}

    # Get data from pixel hit (which is created by a digitizer)
    if containsPXH:
        for pxh in br_pxh:
            #pxhPulse['q'].append(pxh.getPixelPulse())
            #pxhPulse['q'].append(pxh.getPixelPulse().getCharge())
            #pxhPulse['t'].append(pxh.getPixelPulse().getLocalTime())

            pass
    # Add objects to event_data
    event_data['dc_positions'].append(dcGPos)
    event_data['pc_positions'].append(pcGPos)
    event_data['pc_pulses'].append(pcPulse)

# Plotting in ROOT (don't use)
    #dcHist = ROOT.TH2F("dcHist", "Deposited Counts;x[mm];y[mm]", 30, -1, 1, 30, -1, 1)
    #for x,y in zip(dcGPos['x'],dcGPos['y']):
    #    dcHist.Fill(x,y)
    #canvas = ROOT.TCanvas("depositedCanvas", "2D Histogram", 800, 600)
    #dcHist.Draw("COLZ")

num_pixels_x = 3
num_pixels_y = 3
pixel_width = 0.330 #mm
pixel_height = 2 #mm

# Plot Deposited & Propagated Charge Distributions
if (containsDC or containsPC) and num_events <= 5:

    # Plot deposited and propagated charge distributions
    #if containsDC and containsPC:
    if False: # Don't plot
        charges_fig, charge_axs = plt.subplots(1,2)
        for i in range(num_events):
            dcGPos = event_data['dc_positions'][i]
            pcGPos = event_data['pc_positions'][i]
            charge_axs[0].hist2d(dcGPos['x'], dcGPos['y'], bins=100)
            charge_axs[1].hist2d(pcGPos['x'], pcGPos['y'], bins=100)
        for ax in charge_axs:
            ax.set_aspect('equal')
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
        charge_axs[0].set_title('Deposited Charges')
        charge_axs[1].set_title('Propagated Charges')

    # Plot scatter with one or both of Deposited & Propagated Charge Counts
    scatter_fig = plt.figure()
    scatter_ax = scatter_fig.add_subplot(projection='3d')
    opacity_scaling = 0.1/num_events 

    for i in range(num_events): # Add the charges from each event
        if containsDC: 
            dcGPos = event_data['dc_positions'][i]
            scatter_ax.scatter(dcGPos['x'], dcGPos['y'], dcGPos['z'], marker='.', alpha=opacity_scaling)
        if containsPC:
            pcGPos = event_data['pc_positions'][i]
            scatter_ax.scatter(pcGPos['x'], pcGPos['y'], pcGPos['z'], marker='.', alpha=opacity_scaling)

    # Plot histogram
    hist_fig, hist_ax = plt.subplots()
    bins = np.linspace(-1,1,200)

    dcG_z = {'e':[],'h':[]}
    pcG_z = {'e':[],'h':[]}
    for i in range(num_events): # Add the charges from each event to a single array with only z values
        if containsDC: 
            dcGPos = event_data['dc_positions'][i]
            # print(dcGPos['type'])
            dcG_z['e'].append(np.array(dcGPos['z'])[np.array(dcGPos['type'])==255]) 
            dcG_z['h'].append(np.array(dcGPos['z'])[np.array(dcGPos['type'])==1]) 
        if containsPC:
            pcGPos = event_data['pc_positions'][i]
            pcG_z['e'].append(np.array(pcGPos['z'])[np.array(pcGPos['type'])==255]) 
            pcG_z['h'].append(np.array(pcGPos['z'])[np.array(pcGPos['type'])==1]) 
    
    if containsDC: 
        hist_ax.hist(dcG_z['e'], bins, alpha=0.5, label='Deposited Electrons')
        hist_ax.hist(dcG_z['h'], bins, alpha=0.5, label='Deposited Holes')
    if containsPC: 
        hist_ax.hist(pcG_z['e'], bins, alpha=0.5, label='Propagated Electrons')
        hist_ax.hist(pcG_z['h'], bins, alpha=0.5, label='Propagated Holes')

    hist_ax.legend()

    # hist_ax.set_yscale('log')
    
    # Formatting for scatter
    # Add lines indicating the center pixel location
    pixel_x = [pixel_width*k for k in [-0.5, 0.5, 0.5, -0.5, -0.5]]
    pixel_y = [pixel_width*k for k in [-0.5, -0.5, 0.5, 0.5, -0.5]] 
    pixel_z = [pixel_height*k for k in [0.5, 0.5, 0.5, 0.5, 0.5]] # pixel is center in z
    scatter_ax.plot(pixel_x, pixel_y, pixel_z, linestyle='--', color='k')

    scatter_ax.set_xlabel('x [mm]')
    scatter_ax.set_ylabel('y [mm]')
    scatter_ax.set_zlabel('z [mm]')
    scatter_ax.set_xlim(-num_pixels_x/2*pixel_width, num_pixels_x/2*pixel_width) # bounds of 5x5 are -0.825,0.825
    scatter_ax.set_ylim(-num_pixels_y/2*pixel_width, num_pixels_y/2*pixel_width) # bounds of 5x5 are -0.825,0.825
    scatter_ax.set_zlim(-1, 1)

    # Set titles depending on charge objects available
    if containsDC and containsPC: 
        scatter_ax.set_title("Locations of Deposited and Propagated Charges")
        hist_ax.set_title("Z Positions of Deposited and Propagated Charges")
    elif containsDC: 
        scatter_ax.set_title("Locations of Deposited Charges")
        hist_ax.set_title("Z Positions of Deposited Charges")
    else: 
        scatter_ax.set_title("Locations of Propagated Charges")
        hist_ax.set_title("Z Positions of Propagated Charges")

if containsPC:

    # Create binning for charge
    width = 250 #ns , initial total width of the time binning (width = num_bins * width of each bin)
    #width = max(chargePulse['t'])+1 #ns
    num_bins = 500
    threshold = 5 # e , threshold for determining width of plot
    buffer = 2 # ns , number of nanoseconds on either side of the pulse to plot

    # Sum the Shockley-Ramo induced charges manually using bins defined above
    pulse_fig, pulse_ax = plt.subplots()
    avg_charge = np.zeros(num_bins) # stores charges from all events
 
    num_pixels_x = 3
    num_pixels_y = 3
    total_charge_pixeled = np.zeros((num_pixels_x,num_pixels_y)) # PC has no way of knowing the number of pixels
    for pcPulse in event_data['pc_pulses']:

        t_binning = np.arange(0,width,width/num_bins)
        summed_charge = np.zeros(num_bins) #e
        for i in range(len(pcPulse['q'])):

            # Sort the charges (all pixels combined) over time bins
            index = math.floor(pcPulse['t'][i]*num_bins/width)
            if index<num_bins: summed_charge[index]+=pcPulse['q'][i]
            #if index>0: print(summed_charge[index])
            
            # Get indices to total the charge in each pixel separately
            i_x = math.floor(num_pixels_x / 2 + pcPulse['x'][i] / pixel_width)
            i_y = math.floor(num_pixels_y / 2 + pcPulse['y'][i] / pixel_width)
            if i_x > 0 and i_x < num_pixels_x and i_y > 0 and i_y < num_pixels_y: 
                total_charge_pixeled[i_x][i_y] += pcPulse['q'][i] # Add the charge into its pixel
                #if pcPulse['q'][i] > 0 and i_x != 1 and i_y != 1: print(f'({i_x}{i_y}) recieved {pcPulse['q'][i]}e at time')

        avg_charge = [avg_charge[i]+summed_charge[i]/num_events for i in range(num_bins)] # add summed charge to avg_charge element-wise

        # Determine range for histogram
        min_time = -1
        max_time = -1
        for i in range(len(summed_charge)):
            if summed_charge[i] > threshold:
                min_time = i*width/num_bins
                break

        for i in range(len(summed_charge)-1, -1, -1):
            if summed_charge[i] > threshold:
                max_time = i*width/num_bins
                break

        min_index = max(0, math.floor((min_time-buffer)*num_bins/width))
        max_index = min(num_bins-1, math.floor((max_time+buffer)*num_bins/width))
        
        #charge_ax.hist(summed_charge)
        #time_ax.hist(chargePulse['t'])
        pulse_ax.plot(t_binning[min_index:max_index+1], summed_charge[min_index:max_index+1], alpha=min(0.5,3/num_events))

    # Get averaged charge between all events (assumes they are lined up)
    #avg_charge = [q/num_events for q in avg_charge]
    pulse_ax.plot(t_binning[min_index:max_index+1],avg_charge[min_index: max_index+1], linestyle="--", color='k')
    
    print(f'=== Results from Propagated Charge ===')
    print(f'Total magnitude of propagated charge: {sum(avg_charge)*num_events} (Average per event: {sum(avg_charge)})')
    print(f'Magnitude of propagated charges by pixel:\n{total_charge_pixeled}\n - Average per event:\n{total_charge_pixeled/num_events}')
    
    # Format and show pulse plots
    pulse_ax.set_title(f'Charges sorted by arrival time (average of {num_events} events)')
    pulse_ax.set_xlabel("Time [ns]")
    pulse_ax.set_ylabel("Charge [e]")

if containsPXC:

    num_pixels_x = max([x['i_x'] for x in event_data['pxc_pulses']])+1
    num_pixels_y = max([x['i_y'] for x in event_data['pxc_pulses']])+1

    #print(f'number of pulses: {len(event_data['pxc_pulses'])}')
    pxc_fig, pxc_ax = plt.subplots(num_pixels_x,num_pixels_y)
    pxc_current = []
    pxc_avg = []
    total_charge = np.zeros((num_pixels_x,num_pixels_y))
    
    # Determine rebinning (must loop through each event and pixel to get the longest q array)
    do_rebinning = True # toggle rebinning behavior
    default_bin_width = 0.5 # Used if do_rebinning is set to True
    num_bins = 0
    for pxcPulse in event_data['pxc_pulses']:
        if do_rebinning:
            new_num_bins = math.ceil(len(pxcPulse['q']) * pxcPulse['t_bin'] / default_bin_width)
            num_bins = max(num_bins, new_num_bins)
        else:
            num_bins = max(num_bins, len(pxcPulse['q']))
            bin_width = pxcPulse['t_bin']
    if do_rebinning: bin_width = default_bin_width
    t = np.arange(0, num_bins * bin_width, bin_width)
    
    # Loop through each event to get the charge and current pulses
    for pxcPulse in event_data['pxc_pulses']:
        
        # Get the associated pixel (Note that (1,1) is the center for a 3x3 grid)
        i_x = pxcPulse['i_x']
        i_y = pxcPulse['i_y']

        # Preform the rebinning
        unbinned_q = pxcPulse['q'] # e
        if do_rebinning:
            q = np.zeros(num_bins)
            for i in range(len(unbinned_q)):
                index = math.floor(i * pxcPulse['t_bin'] / bin_width)
                if index < num_bins: q[index] += unbinned_q[i]
        else:
            q = unbinned_q

        # Add to the total charge for output
        total_charge[i_x][i_y] += sum(q) # e

        # Set size of current and average arrays based on the potentially rebinned num_bins
        if len(pxc_current) == 0: pxc_current = np.zeros((num_pixels_x,num_pixels_y,num_bins))
        if len(pxc_avg) == 0: pxc_avg = np.zeros((num_pixels_x,num_pixels_y,num_bins))

        # Convert charge to current by dividing by bin width (dq/dt) 
        #  with added negative sign since current is defined opposite of e- movement
        pxc_current[i_x][i_y] = -1 * q / bin_width * (1.609*10**-19) * (10**9) * (10**6) # e/ns * 1.609e-19 C/e * 10e9 ns/s * 10e6 uA/A = uA

        pxc_ax[i_x][i_y].plot(t, pxc_current[i_x][i_y], alpha=min(0.5, 3/num_events)) # For plotting each event with low opacity

        # Add to the average for each pixel
        pxc_avg[i_x][i_y] = [pxc_avg[i_x][i_y][i] + pxc_current[i_x][i_y][i]/num_events for i in range(len(t))]

    for i_x in range(num_pixels_x):
        for i_y in range(num_pixels_y):
            pxc_ax[i_x][i_y].plot(t,pxc_avg[i_x][i_y], linestyle='--', color='k')
    
    # Get the cumulative charge curves
    pxc_cum = np.zeros((num_pixels_x, num_pixels_y, num_bins))
    pxc_cum_fig, pxc_cum_ax = plt.subplots(num_pixels_x,num_pixels_y)
    for i_x in range(num_pixels_x):
        for i_y in range(num_pixels_y):
            for i_t in range(num_bins):
                previous_value = 0 if i_t == 0 else pxc_cum[i_x][i_y][i_t-1]
                pxc_cum[i_x][i_y][i_t] = previous_value + bin_width * pxc_avg[i_x][i_y][i_t] / (10**9) / (10**6) / (1.609*10**-19) # ns*uA * 1s/1e9ns * 1A/1e6uA * 1e/1.609e19C = e
            pxc_cum_ax[i_x][i_y].plot(t,pxc_cum[i_x][i_y], linestyle='-', color='k')

    # Take the integral to help determine physical interpretation
    # The sum of all charges is close to half the depositied charges, so probably total electrons that make it to the plate.
    # They are just split up based on the weighting potential
    #print(f'Time bin width: {bin_width} ns')
    print(f'=== Results from Pixel Pulse ===')
    print(f'Sum of all charges in center pixel pulse is {total_charge[1][1]}e (Average per event: {total_charge[1][1]/num_events})')
    print(f'Average charge per event (e):\n{total_charge/num_events}')

    current_integral = np.zeros((num_pixels_x, num_pixels_y))
    current_integral_e = np.zeros((num_pixels_x, num_pixels_y))
    for i_x in range(num_pixels_x):
        for i_y in range(num_pixels_y):
            current_integral[i_x][i_y] = bin_width*sum(pxc_current[i_x][i_y]) / (10**9) / (10**6) * -1 # ns*uA * 1s/1e9ns * 1A/1e6uA = C
            current_integral_e[i_x][i_y] = current_integral[i_x][i_y] / (1.609*10**-19)
    #print(f'Integral of the center pixel current pulse is {current_integral[1][1]}C = {current_integral_e[1][1]}e')
    print(f'Integrals of each pixel current (C):\n{current_integral}')
    print(f'Integrals of each pixel current (e):\n{current_integral_e}')

    print(np.shape(pxc_cum))
    print(f'Final values of cumulative charge in each pixel (e):\n{pxc_cum[:,:,-1]}')
    print(f'Final slopes (current) in each pixel (e/ns):\n{pxc_current[:,:,-1] / (10**6) / (10**9) / (1.609*10**-19) }') # uA * 1A/1e6 * 1s/1e9ns * 1e/1.609e-19C = e/ns

    print(f'Sum of integrals over the entire sensor (e): {sum(sum(current_integral_e))}')

    pxc_fig.suptitle(f'Current pulse in each pixel (averaged over {num_events} events)')
    pxc_cum_fig.suptitle(f'Cumulative charge detected on each pixel (averaged over {num_events} events)')

    # Determine bounds for the pulse and cumulative charge plots
    min_t = 0
    max_t = 300 #max(t)

    max_I = np.max(pxc_current)
    min_I = np.min(pxc_current)
    diff_I = max_I-min_I
    max_I += diff_I*0.1
    min_I -= diff_I*0.1

    min_q = np.min(pxc_cum)
    max_q = np.max(pxc_cum)
    diff_q = max_q - min_q
    min_q -= diff_q*0.1
    max_q += diff_q*0.1

    for i_x in range(num_pixels_x):
        for i_y in range(num_pixels_y): 
            min_y = -0.02
            max_y = 0.2 # -1 * min_y if i_x != 1 else 0.2
            
            pxc_ax[i_x][i_y].set_xlim((min_t, max_t))
            pxc_ax[i_x][i_y].set_ylim((min_I, max_I)) # The plots might be transposed from the actual orientation

            #min_y = min(pxc_cum[i_x][i_y])-100
            #max_y = max(pxc_cum[i_x][i_y])+100
            #pxc_cum_ax[i_x][i_y].set_xlim((min_x, max_x))
            pxc_cum_ax[i_x][i_y].set_ylim((min_q, max_q)) # The plots might be transposed from the actual orientation
            
    #pxc_ax[1][0].set_title(f'Current pulse from average of {num_events} events')
    pxc_ax[num_pixels_x-1][1].set_xlabel(f"Time [ns] (bin_width={bin_width}ns)")
    pxc_ax[1][0].set_ylabel("Current [uA]")
    
    pxc_cum_ax[num_pixels_x-1][1].set_xlabel(f"Time [ns] (bin_width={bin_width}ns)")
    pxc_cum_ax[1][0].set_ylabel("Charge [e]")
    
    #pxc_ax[1].set_xlabel(f"Time [ns] (time_step={bin_width})")
    #pxc_ax[1].set_ylabel("Induced Current (dq/dt) [e/ns]")

plt.show()

# input("Keeping window open...") # Keeps plots visible until user presses enter


