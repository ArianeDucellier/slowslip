"""
Script to make synthetic slow slip events and run MODWT on them
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

import MODWT

# Time between events
timesteps = [100, 200, 500, 1000]
# Durations of slow slip events
durations = [2, 5, 10, 20]
# MODWT wavelet filter
name = 'LA8'
# Duration of recording
N = 2000
# MODWT level
J = 8

# Create time vector
time = np.arange(0, N)

# Set random seed
np.random.seed(0)

# Loop on time between events
for timestep in timesteps:
    # Loop on duration
    for duration in durations:
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
        # Plot MODWT
        params = {'xtick.labelsize':24,
                  'ytick.labelsize':24}
        pylab.rcParams.update(params)   
        fig = plt.figure(1, figsize=(15, 5 * (J + 3)))
        # Plot data
        plt.subplot2grid((J + 2, 1), (J + 1, 0))
        plt.plot(time, disp, 'k', label='Data')
        plt.ylim(-2.0, 2.0)
        plt.legend(loc=3, fontsize=14)
        # Plot details
        for j in range(0, J):
            plt.subplot2grid((J + 2, 1), (J - j, 0))
            plt.plot(time, D[j], 'k', label='D' + str(j + 1))
            plt.ylim(minD, maxD)
            plt.legend(loc=3, fontsize=14)
        # Plot smooth
        plt.subplot2grid((J + 2, 1), (0, 0))
        plt.plot(time, S[J], 'k', label='S' + str(J))
        plt.ylim(-2.0, 2.0)
        plt.legend(loc=3, fontsize=14)
        # Save figure
        title = 'Time between events = ' + str(timestep) + ' - Duration of events = ' + str(duration)
        plt.suptitle(title, fontsize=30)
        plt.savefig('synthetics/' + str(timestep) + '_' + str(duration) + '_DS.eps', format='eps')
        plt.close(1)
