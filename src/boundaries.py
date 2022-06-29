""" Script to look at MODWT of GPS data
and check the effect of boundaries on the MODWT """

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

from math import log, sqrt
from matplotlib import ticker

import DWT, MODWT
from MODWT import get_DS, get_scaling, inv_pyramid, pyramid

# MODWT parameters
wavelet = 'LA8'
J = 10

# Choose the station
station = 'PGC5'
direction = 'lon'
dataset = 'cleaned'
filename = '../data/PANGA/' + dataset + '/' + station + '.' + direction

# Load the data
data = np.loadtxt(filename, skiprows=26)
time = data[:, 0]
disp = data[:, 1]
error = data[:, 2]
sigma = np.std(disp)

# Correct for the repeated values
dt = np.diff(time)
gap = np.where(dt < 1.0 / 365.0 - 0.0001)[0]
for i in range(0, len(gap)):
    if (gap[i] + 2 < len(time)):
        if ((time[gap[i] + 2] - time[gap[i] + 1] > 2.0 / 365.0 - 0.0001) \
        and (time[gap[i] + 2] - time[gap[i] + 1] < 2.0 / 365.0 + 0.0001)):
            time[gap[i] + 1] = 0.5 * (time[gap[i] + 2] + time[gap[i]])
        elif (gap[i] + 3 < len(time)):
            if ((time[gap[i] + 2] - time[gap[i] + 1] > 1.0 / 365.0 - 0.0001) \
            and (time[gap[i] + 2] - time[gap[i] + 1] < 1.0 / 365.0 + 0.0001) \
            and (time[gap[i] + 3] - time[gap[i] + 2] > 2.0 / 365.0 - 0.0001) \
            and (time[gap[i] + 3] - time[gap[i] + 2] < 2.0 / 365.0 + 0.0001)):
                time[gap[i] + 1] = time[gap[i] + 2]
                time[gap[i] + 2] = 0.5 * (time[gap[i] + 2] + time[gap[i] + 3])

# Look for gaps greater than 1 day
days = 2
dt = np.diff(time)
gap = np.where(dt > days / 365.0 - 0.0001)[0]
duration = np.round((time[gap + 1] - time[gap]) * 365).astype(np.int)

# Fill the gaps by interpolation
for j in range(0, len(gap)):
    time = np.insert(time, gap[j] + 1, \
    time[gap[j]] + np.arange(1, duration[j]) / 365.0)
    if gap[j] >= 4:
        before = np.mean(disp[gap[j] - 4 : gap[j] + 1])
    else:
        before = np.mean(disp[0 : gap[j] + 1])
    if len(disp) >= gap[j] + 6:
        after = np.mean(disp[gap[j] + 1 : gap[j] + 6])
    else:
        after = np.mean(disp[gap[j] + 1 :])
    disp_interp = before + (after - before) * (np.arange(0, duration[j] - 1) + 1) / duration[j] + \
        np.random.normal(0.0, sigma, duration[j] - 1)
    disp = np.insert(disp, gap[j] + 1, disp_interp)
    gap[j + 1 : ] = gap[j + 1 : ] + duration[j] - 1

# MODWT
(W, V) = pyramid(disp, wavelet, J)
(D, S) = get_DS(disp, W, wavelet, J)

# Cut the data on both sides
disp_cut = disp[(time >= 2008.0) & (time <= 2012.0)]
time_cut = time[(time >= 2008.0) & (time <= 2012.0)]

(W_cut, V_cut) = pyramid(disp_cut, wavelet, J)
(D_cut, S_cut) = get_DS(disp_cut, W_cut, wavelet, J)

# Start figure
params = {'xtick.labelsize':24,
          'ytick.labelsize':24}
pylab.rcParams.update(params) 

plt.figure(1, figsize=(30, 3 * 4))

maxD = max([np.max(Dj) for Dj in D])
minD = min([np.min(Dj) for Dj in D])

# Plot data
ax = plt.subplot2grid((4, 3), (0, 0))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.plot(time_cut, disp_cut, 'r')
plt.plot(time, disp, 'k', label='Data')
plt.xlim([2008.0, 2012.0])
plt.ylim([np.min(disp), np.max(disp)])
plt.legend(loc=3, fontsize=20)
ax.axes.xaxis.set_ticklabels([])       
ax.axes.set_xticks([2008, 2009, 2010, 2011, 2012])

# Plot details at each level
for j in range(0, J):
    if j < 3:
        ax = plt.subplot2grid((4, 3), (j + 1, 0))
    elif j < 7:
        ax = plt.subplot2grid((4, 3), (j - 3, 1))
    else:
        ax = plt.subplot2grid((4, 3), (j - 7, 2))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    plt.plot(time_cut, D_cut[j], 'r')
    plt.plot(time, D[j], 'k', label='D' + str(j + 1))
    plt.xlim([2008.0, 2012.0])
    plt.ylim(minD, maxD)
    plt.legend(loc=3, fontsize=20)
    if ((j == 2) or (j == 6)):
        plt.xlabel('Time (years)', fontsize=24)
    else:
        ax.axes.xaxis.set_ticklabels([])
    if j >= 3:
        ax.axes.yaxis.set_ticklabels([])
    ax.axes.set_xticks([2008, 2009, 2010, 2011, 2012])

# Plot scaling coefficients for the last level
ax = plt.subplot2grid((4, 3), (3, 2))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.plot(time_cut, S_cut[J], 'r')
plt.plot(time, S[J], 'k', label='S' + str(J))
plt.xlim([2008.0, 2012.0])
plt.ylim([np.min(disp), np.max(disp)])
plt.legend(loc=3, fontsize=20)
ax.axes.yaxis.set_ticklabels([])
plt.xlabel('Time (years)', fontsize=24)
ax.axes.set_xticks([2008, 2009, 2010, 2011, 2012])

plt.tight_layout()
plt.savefig('boundaries.eps', format='eps')
plt.close(1)
