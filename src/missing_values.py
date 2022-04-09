""" Script to look at MODWT of GPS data
with different methods to fill in missing values """

import matplotlib.pyplot as plt
import numpy as np

from math import log, sqrt
from matplotlib import ticker

import DWT, MODWT
from MODWT import get_DS, get_scaling, inv_pyramid, pyramid

# Choose the station
station = 'PGC5'
direction = 'lon'
dataset = 'cleaned'
filename = '../data/PANGA/' + dataset + '/' + station + '.' + direction
# Load the data
data = np.loadtxt(filename, skiprows=26)
time = data[:, 0]
disp = data[:, 1]
# Correct for the repeated value
dt = np.diff(time)
gap = np.where(dt < 1.0 / 365.0 - 0.001)[0]
time[gap[0] + 1] = time[gap[0] + 2]
time[gap[0] + 2] = 0.5 * (time[gap[0] + 2] + time[gap[0] + 3])
# Look for gaps greater than 1 day
days = 1
dt = np.diff(time)
gap = np.where(dt > days / 365.0 + 0.001)[0]
# Select a subset of the data without gaps
ibegin = 2943
iend = 4333
time = time[ibegin + 1 : iend + 1]
disp = disp[ibegin + 1 : iend + 1]
N = np.shape(disp)[0]

# Parameters
name = 'LA8'
g = MODWT.get_scaling(name)
L = len(g)
J = 10
nmax = 42
nstep = 3
filling = 'both'

# Set random seed
np.random.seed(0)

# Compute MODWT
[W0, V0] = pyramid(disp, name, J)
(D0, S0) = get_DS(disp, W0, name, J)

# Loop on missing values
gaps = [605, 790]

# Loop on length of gap
for n in range(1, nmax, nstep):

    plt.figure(1, figsize=(len(gaps) * 3, (J + 2) * 2))

    for count, gap in enumerate(gaps):
        disp_interp = np.copy(disp)
        # Remove points, replace by interpolation or noise
        if (filling == 'interpolation'):
            for i in range(0, n):
                before = np.mean(disp_interp[gap - 6 : gap - 1])
                after = np.mean(disp_interp[gap + n : gap + n + 5])
                disp_interp[gap + i] = before + (after - before) * (i + 1) / (n + 1)
        elif (filling == 'noise'):
            sigma = np.std(disp)
            disp_interp[gap : (gap + n)] = np.random.normal(0.0, sigma, n)
        else:
            sigma = np.std(disp)
            for i in range(0, n):
                before = np.mean(disp_interp[gap - 6 : gap - 1])
                after = np.mean(disp_interp[gap + n : gap + n + 5])
                disp_interp[gap + i] = before + (after - before) * (i + 1) / (n + 1) + \
                    np.random.normal(0.0, sigma, 1)
        # Compute MODWT
        [W, V] = pyramid(disp_interp, name, J)
        (D, S) = get_DS(disp_interp, W, name, J)

        # Plot data
        ax = plt.subplot2grid((J + 2, len(gaps)), (0, count))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        plt.plot(time, disp_interp, 'r')
        plt.plot(time, disp, 'k', label='Data')
        plt.xlim(np.min(time[gap - 100]), np.max(time[gap + n + 99]))
        plt.legend(loc=1)        
        # Plot details at each level
        for j in range(0, J):
            ax = plt.subplot2grid((J + 2, len(gaps)), (j + 1, count))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            plt.plot(time, D[j], 'r')
            plt.plot(time, D0[j], 'k', label='D' + str(j + 1))
            plt.xlim(np.min(time[gap - 100]), np.max(time[gap + n + 99]))
            plt.legend(loc=1)
        # Plot scaling coefficients for the last level
        ax = plt.subplot2grid((J + 2, len(gaps)), (J + 1, count))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        plt.plot(time, S[J], 'r')
        plt.plot(time, S0[J], 'k', label='S' + str(J))
        plt.xlim(np.min(time[gap - 100]), np.max(time[gap + n + 99]))
        plt.legend(loc=1)
        plt.xlabel('Time (years)', fontsize=14)

    plt.tight_layout()
    plt.savefig('missing_values/' + filling + '/DS_' + str(n) + '.eps', format='eps')
    plt.close(1)
