"""
Script to plot a vespagram-like figure of slow slip
"""

import datetime
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from math import cos, pi, sin, sqrt
from scipy.io import loadmat

import date
import DWT
from MODWT import get_DS, get_scaling, pyramid

def vespagram(station_file, lat0, lon0, name, radius, direction, dataset, \
    wavelet, J, slowness):
    """
    """
    # Read station file
    stations = pd.read_csv(station_file, sep=r'\s{1,}', header=None, engine='python')
    stations.columns = ['name', 'longitude', 'latitude']

    # Keep only stations in a given radius
    a = 6378.136
    e = 0.006694470
    dx = (pi / 180.0) * a * cos(lat0 * pi / 180.0) / sqrt(1.0 - e * e * \
        sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0))
    dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * \
        pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)
    x = dx * (stations['longitude'] - lon0)
    y = dy * (stations['latitude'] - lat0)
    stations['distance'] = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0))
    mask = stations['distance'] <= radius
    stations = stations.loc[mask]

    # Wavelet initialization
    g = get_scaling(wavelet)
    L = len(g)
    (nuH, nuG) = DWT.get_nu(wavelet, J)

    # Wavelet vectors initialization
    times = []
    disps = []
    Ws = []
    Vs = []
    Ds = []
    Ss = []

    # Read GPS data and compute wavelet transform
    for station in stations['name']:
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
            if ((time[gap[i] + 2] - time[gap[i] + 1] > 2.0 / 365.0 - 0.0001) \
            and (time[gap[i] + 2] - time[gap[i] + 1] < 2.0 / 365.0 + 0.0001)):
                time[gap[i] + 1] = 0.5 * (time[gap[i] + 2] + time[gap[i]])
            elif ((time[gap[i] + 2] - time[gap[i] + 1] > 1.0 / 365.0 - 0.0001) \
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
            disp = np.insert(disp, gap[j] + 1, \
                np.random.normal(0.0, sigma, duration[j] - 1))
            gap[j + 1 : ] = gap[j + 1 : ] + duration[j] - 1
        times.append(time)
        disps.append(disp)
        # MODWT
        [W, V] = pyramid(disp, wavelet, J)
        Ws.append(W)
        Vs.append(V)
        (D, S) = get_DS(disp, W, wavelet, J)
        Ds.append(D)
        Ss.append(S)

    # Divide into time blocks
    tblocks = []
    for time in times:
        tblocks.append(np.min(time))
        tblocks.append(np.max(time))
    tblocks = sorted(set(tblocks))
    tbegin = tblocks[0 : -1]
    tend = tblocks[1 : ]

    time_vesps = []
    vesps = []

    # Loop on time blocks
    for t in range(0, len(tblocks) - 1):

        # Find time period
        for time in times:
            indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
            if (len(indices) > 0):
                time_subset = time[indices]
                break
        time_vesps.append(time_subset)

        # Initialize vespagram
        vespagram = np.zeros((len(slowness), len(time_subset), J))      

        # Loop on time scales
        for j in range(0, J):
            nsta = 0
            for (time, D, lat) in zip(times, Ds, stations['latitude']):
                indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
                if (len(indices) > 0):
                    nsta = nsta + 1
                    tj = time[indices]
                    Dj = D[j][indices]
                    for i in range(0, len(slowness)):
                        Dj_interp = np.interp(time_subset + \
                            slowness[i] * (lat - lat0), tj, Dj)
                        vespagram[i, :, j] = vespagram[i, :, j] + Dj_interp
            vespagram[:, :, j] = vespagram[:, :, j] / nsta
        vesps.append(vespagram)

    time_subset = np.concatenate(time_vesps)
    vespagram = np.concatenate(vesps, axis=1)

    # Plot
    for j in range(0, J):
        plt.figure(1, figsize=(15, 5))
        plt.contourf(time_subset, slowness * 365.25 / dy, \
            vespagram[:, :, j], cmap=plt.get_cmap('seismic'), \
            vmin=-2.0, vmax=2.0)
        plt.xlabel('Time (year)')
        plt.ylabel('Slowness (day / km)')
        plt.colorbar(orientation='horizontal')
        plt.savefig('vespagram/D' + str(j + 1) + '_' + name + '.eps', \
            format='eps')
        plt.close(1)

if __name__ == '__main__':

    station_file = '../data/PANGA/stations.txt'
    radius = 100
    direction = 'lon'
    dataset = 'cleaned'
    wavelet = 'LA8'
    J = 6
    slowness = np.arange(-0.1, 0.105, 0.005)

    for i in range(0, 10):
        lat0 = 47 + 0.2 * i
        lon0 = -123
        name = str(i)
        vespagram(station_file, lat0, lon0, name, radius, direction, \
            dataset, wavelet, J, slowness)
