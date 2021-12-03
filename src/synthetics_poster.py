"""
Script to make synthetic slow slip events and run MODWT on them
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

import MODWT

# Time between events
timestep = 500
# Durations of slow slip events
durations = [2, 5, 10, 20]
# MODWT wavelet filter
name = 'LA8'
# Duration of recording
N = 2000
# MODWT level
J = 10

# Create time vector
time = np.arange(0, N)

# Set random seed
np.random.seed(0)

# Start figure
params = {'xtick.labelsize':20,
          'ytick.labelsize':20}
pylab.rcParams.update(params)   
fig = plt.figure(1, figsize=(5 * len(durations), 3 * (2)))

# Loop on duration
for i, duration in enumerate(durations):
    # Create displacements vectors
    part1 = np.arange(0, timestep - duration) / (timestep - duration)
    part2 = (timestep - np.arange(timestep - duration, timestep)) / duration
    signal = np.concatenate([part1, part2])
    disp = np.tile(signal, int(N / timestep))
    # Compute MODWT
    (W, V) = MODWT.pyramid(disp, name, J)
    (D, S) = MODWT.get_DS(disp, W, name, J)
    # Plot data
    ax = plt.subplot2grid((2, len(durations)), (0, i))
    plt.plot(time, disp, 'k', label='Data')
    plt.ylim(-2.0, 2.0)
    plt.legend(loc=3, fontsize=20, frameon=False)
    if i != 0:
        ax.axes.yaxis.set_ticks([])
    title = 'Duration of event = ' + str(duration) + ' days'
    plt.title(title, fontsize=20)
    # Plot sum of details 6, 7 and 8
    ax = plt.subplot2grid((2, len(durations)), (1, i))
    plt.plot(time, D[5] + D[6] + D[7], 'k', label='D6 + D7 + D8')
    plt.ylim(-2.0, 2.0)
    plt.legend(loc=3, fontsize=20, frameon=False)
    if i != 0:
        ax.axes.yaxis.set_ticks([])
    plt.xlabel('Time (days)', fontsize=20)

plt.tight_layout()
plt.savefig('synthetics/D678.eps', format='eps')
plt.close(1)
