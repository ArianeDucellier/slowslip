"""
Script to make synthetic slow slip events and run MODWT on them
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

import MODWT

# Time between events
timesteps = [100, 300, 500]
# Durations of slow slip events
durations = [2, 5, 10, 20, 50, 100]
# MODWT wavelet filter
name = 'LA8'
# Duration of recording
N = 5000
# MODWT level
J = 8

# Create time vector
time = np.arange(0, N + 1)

# Set random seed
np.random.seed(0)

# Loop on time between events
for timestep in timesteps:
    # Loop on duration
    for duration in durations:
        # Create displacements vectors
        signal = np.zeros(N + 1)
        for i in range(0, N + 1):
            if (time[i] <= 0.5 * (N - duration)):
                signal[i] = time[i] / (N - duration)
            elif (time[i] >= 0.5 * (N + duration)):
                signal[i] = (time[i] - N) / (N - duration)
            else:
                signal[i] = (0.5 * N - time[i]) / duration           
        disp = signal + noise
        # Compute MODWT
        (W, V) = MODWT.pyramid(disp, name, J)
        (D, S) = MODWT.get_DS(disp, W, name, J)
        maxD = max([np.max(Dj) for Dj in D])
        minD = min([np.min(Dj) for Dj in D])
        if SNR != 0.0:
            # Thresholding of MODWT wavelet coefficients
            sigE = 1.0 / SNR
            Wt = []
            for j in range(1, J + 1):
                Wj = W[j - 1]
                deltaj = sqrt(2.0 * (sigE ** 2.0) * log(N + 1) / (2.0 ** j))
                Wjt = np.where(np.abs(Wj) >= deltaj, Wj, 0.0)
                if (j == J):
                    Vt = np.where(np.abs(V) >= deltaj, V, 0.0)
                Wt.append(Wjt)
            dispt = MODWT.inv_pyramid(Wt, Vt, name, J)
            # Low-pass filter
            dispf = lfilter(b, a, disp)
        # Plot MODWT
        params = {'xtick.labelsize':24,
                  'ytick.labelsize':24}
        pylab.rcParams.update(params)   
        fig = plt.figure(1, figsize=(15, 5 * (J + 3)))
        # Plot denoised data
        plt.subplot2grid((J + 3, 1), (J + 2, 0))
        plt.plot(time, signal, 'k', label='Signal', linewidth=3)
        if SNR != 0.0:
            plt.plot(time, dispt, 'r', label='Denoised', linewidth=3)
            plt.plot(time, dispf, 'grey', label='Low-pass filtered')
        plt.xlabel('Time (days)', fontsize=30)
        plt.ylim(-2.0, 2.0)
        plt.legend(loc=3, fontsize=14)
        # Plot data
        plt.subplot2grid((J + 3, 1), (J + 1, 0))
        plt.plot(time, disp, 'k', label='Data')
        plt.ylim(-2.0, 2.0)
        plt.legend(loc=3, fontsize=14)
        # Plot details
        for j in range(0, J):
            plt.subplot2grid((J + 3, 1), (J - j, 0))
            plt.plot(time, D[j], 'k', label='D' + str(j + 1))
            plt.ylim(minD, maxD)
            plt.legend(loc=3, fontsize=14)
        # Plot smooth
        plt.subplot2grid((J + 3, 1), (0, 0))
        plt.plot(time, S[J], 'k', label='S' + str(J))
        plt.ylim(-2.0, 2.0)
        plt.legend(loc=3, fontsize=14)
        # Save figure
        title = 'Wavelet = ' + name + ' - SNR = ' + str(SNR) + ' - Duration = ' + str(duration)
        plt.suptitle(title, fontsize=50)
        plt.savefig('synthetics/' + name + '_' + str(SNR) + '_' + \
            str(duration) + '_DS.eps', format='eps')
        plt.close(1)
