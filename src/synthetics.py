"""
Script to make synthetic slow slip events and run MODWT on them
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

import MODWT

# Time between events
#timesteps = [100, 200, 500, 1000]
timesteps = [500]
# Durations of slow slip events
#durations = [5, 10, 20, 40]
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

# Loop on time between events
for timestep in timesteps:

    # Start figure
    params = {'xtick.labelsize':20,
              'ytick.labelsize':20}
    pylab.rcParams.update(params)   
#    fig = plt.figure(1, figsize=(5 * len(durations), 3 * (J + 2)))
    fig = plt.figure(1, figsize=(5 * len(durations), 3 * 3))

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
        maxD = max([np.max(Dj) for Dj in D])
        minD = min([np.min(Dj) for Dj in D])
        # Plot data
#        ax = plt.subplot2grid((J + 2, len(durations)), (0, i))
#        ax = plt.subplot2grid((3, len(durations)), (0, i))
#        plt.plot(time, disp, 'k', label='Data')
#        plt.ylim(-2.0, 2.0)
#        plt.legend(loc=3, fontsize=20)
#        if i != 0:
#            ax.axes.yaxis.set_ticks([])
#        title = 'Duration of event = ' + str(duration) + ' days'
#        plt.title(title, fontsize=20)
        # Plot details
        for j in range(8, J):
#            ax = plt.subplot2grid((J + 2, len(durations)), (j + 1, i))
            ax = plt.subplot2grid((3, len(durations)), (j - 8, i))
            plt.plot(time, D[j], 'k', label='D' + str(j + 1))
            plt.ylim(minD, maxD)    
            plt.legend(loc=3, fontsize=20)
            if i != 0:
                ax.axes.yaxis.set_ticks([])
            if j == 8:
                title = 'Duration of event = ' + str(duration) + ' days'
                plt.title(title, fontsize=20)
#            if j == 7:
#                plt.xlabel('Time (days)', fontsize=20)
        # Plot smooth
#        ax = plt.subplot2grid((J + 2, len(durations)), (J + 1, i))
        ax = plt.subplot2grid((3, len(durations)), (2, i))
        plt.plot(time, S[J], 'k', label='S' + str(J))
        plt.ylim(-2.0, 2.0)
        plt.xlabel('Time (days)', fontsize=20)
        plt.legend(loc=3, fontsize=20)
        if i != 0:
            ax.axes.yaxis.set_ticks([])

    # Save figure
    plt.tight_layout()
    plt.savefig('synthetics/' + str(timestep) + '_DS_4.eps', format='eps')
    plt.close(1)
