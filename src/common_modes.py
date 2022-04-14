"""
Script to compute common modes for GPS stations
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from math import cos, floor, pi, sin, sqrt

import DWT
from MODWT import get_DS, get_scaling, pyramid

def detrend(time, disp):
    """
    Detrend the signal by applying a one-year moving average
    """
    N = len(disp)
    disp_detrend = np.zeros(N)
    for i in range(0, N):
        indices = np.where((time >= time[i] - 0.5) & (time <= time[i] + 0.5))[0]
        disp_detrend[i] = np.mean(disp[indices])
    return disp_detrend

def stack(station_file, dataset, direction, lat_min, lat_max, lon_min, lon_max):
    """
    Stack all GPS stations within a latitude bin
    """
    # Read station file
    stations = pd.read_csv(station_file, sep=r'\s{1,}', header=None, \
        engine='python')
    stations.columns = ['name', 'longitude', 'latitude']
    subset = stations.loc[(stations['latitude'] >= lat_min) & \
                          (stations['latitude'] <= lat_max) & \
                          (stations['longitude'] >= 0.5 * (lon_min + lon_max))]

    # Stack the detrended signal over GPS stations
    times = []
    disps = []
    for station in subset['name']:
        filename = '../data/PANGA/' + dataset + '/' + station + '.' + direction
        data = np.loadtxt(filename, skiprows=26)
        time = data[:, 0]
        disp = data[:, 1]
        times.append(time)
        disps.append(disp)

    # Divide into time blocks
    tblocks = []
    for time in times:
        tblocks.append(np.min(time))
        tblocks.append(np.max(time))
    tblocks = sorted(set(tblocks))
    tbegin = tblocks[0 : -1]
    tend = tblocks[1 : ]

    # Initializations
    time_stack = []
    disp_stack = []
    if len(tblocks) > 0:
        time_sta = np.zeros(2 * (len(tblocks) - 1))
        nb_sta = np.zeros(2 * (len(tblocks) - 1))

    # Loop on time blocks
    t0 = 0
    for t in range(0, len(tblocks) - 1):
        for time in times:
            indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
            if (len(indices) > 0):
                time_subset = time[indices]
                break
        nsta = 0
        disp_subset = np.zeros(len(time_subset))
        for (time, disp) in zip(times, disps):
            indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
            if (len(indices) > 0):
                nsta = nsta + 1
                disp_interp = np.interp(time_subset, time[indices], disp[indices])
                disp_subset = disp_subset + disp_interp
        disp_subset = disp_subset / nsta
        time_stack.append(time_subset)
        disp_stack.append(disp_subset)
    time_subset = np.concatenate(time_stack)
    disp_subset = np.concatenate(disp_stack)
    return(time_subset, disp_subset)

def compute_common_modes(station_file, dataset, direction, latitudes, longitudes):
    """
    """
    for i in range(0, len(latitudes) - 1):
        (time, disp) = stack(station_file, dataset, direction, latitudes[i], latitudes[i + 1], \
            longitudes[i], longitudes[i + 1])
        disp = detrend(time, disp)
        filename = 'common_modes/' + 'lat_' + str(i) + '.pkl'
        pickle.dump([time, disp, latitudes[i], latitudes[i + 1]], open(filename, 'wb'))

def compute_wavelets(station_file, lats, lons, radius, direction, dataset, \
    wavelet, J, latitudes):
    """
    """
    # Read station file
    stations = pd.read_csv(station_file, sep=r'\s{1,}', header=None, \
        engine='python')
    stations.columns = ['name', 'longitude', 'latitude']

    # Define subset of stations
    subset = pd.DataFrame(columns=['name', 'longitude', 'latitude'])

    # Loop on latitude and longitude
    a = 6378.136
    e = 0.006694470
    for (lat, lon) in zip(lats, lons):
        # Keep only stations in a given radius       
        dx = (pi / 180.0) * a * cos(lat * pi / 180.0) / sqrt(1.0 - e * e * \
            sin(lat * pi / 180.0) * sin(lat * pi / 180.0))
        dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * \
            sin(lat * pi / 180.0) * sin(lat * pi / 180.0)) ** 1.5)
        x = dx * (stations['longitude'] - lon)
        y = dy * (stations['latitude'] - lat)
        stations['distance'] = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0))
        mask = stations['distance'] <= radius
        sub_stations = stations.loc[mask]
        subset = pd.concat([subset, sub_stations], ignore_index=True)
    subset.drop(columns=['distance'], inplace=True)
    subset.drop_duplicates(ignore_index=True, inplace=True)

    # Wavelet initialization
    g = get_scaling(wavelet)
    L = len(g)
    (nuH, nuG) = DWT.get_nu(wavelet, J)

    # Read GPS data and compute wavelet transform
    for (station, latitude) in zip(subset['name'], subset['latitude']):
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

        # Remove common modes
        i0 = 0
        for i in range(0, len(latitudes) - 1):
            if ((latitude >= latitudes[i]) & (latitude <= latitudes[i + 1])):
                i0 = i
        filename = 'common_modes/' + 'lat_' + str(i0) + '.pkl'
        (time_mode, disp_mode, lat_min, lat_max) = pickle.load(open(filename, 'rb'))
        disp_mode = np.interp(time, time_mode, disp_mode)
        disp = disp - disp_mode
        
        # MODWT
        (W, V) = pyramid(disp, wavelet, J)
        (D, S) = get_DS(disp, W, wavelet, J)

        # Save wavelets into file
        filename = 'MODWT_GPS_mode/' + dataset + '_' + station + '_' + direction + '.pkl'
        pickle.dump([time, disp, W, V, D, S], open(filename, 'wb'))

        # Start figure
        params = {'xtick.labelsize':24,
                  'ytick.labelsize':24}
        pylab.rcParams.update(params)   
        fig = plt.figure(1, figsize=(10, 3 * (J + 3)))

        maxD = max([np.max(Dj) for Dj in D])
        minD = min([np.min(Dj) for Dj in D])

        # Plot data
        plt.subplot2grid((J + 2, 1), (0, 0))
        plt.plot(time, disp, 'k', label='Data')
        plt.xlim([2006.0, 2021.5])
        plt.ylim([np.min(disp), np.max(disp)])
        plt.legend(loc=3, fontsize=20)
        # Plot details
        for j in range(0, J):
            plt.subplot2grid((J + 2, 1), (j + 1, 0))
            plt.plot(time, D[j], 'k', label='D' + str(j + 1))
            plt.xlim([2006.0, 2021.5])
            plt.ylim(minD, maxD)
            plt.legend(loc=3, fontsize=20)
        # Plot smooth
        plt.subplot2grid((J + 2, 1), (J + 1, 0))
        plt.plot(time, S[J], 'k', label='S' + str(J))
        plt.xlim([2006.0, 2021.5])
        plt.ylim([np.min(disp), np.max(disp)])
        plt.legend(loc=3, fontsize=20)
        plt.xlabel('Time (years)', fontsize=24)
        
        # Save figure
        plt.tight_layout()
        plt.savefig('MODWT_GPS_mode/' + dataset + '_' + station + '_' + \
            direction + '.eps', format='eps')
        plt.close(1)

if __name__ == '__main__':

    station_file = '../data/PANGA/stations.txt'
    direction = 'lon'
    dataset = 'cleaned'
    latitudes = np.array([46, 46.5, 47, 47.5, 48, 48.5, 49, 49.5, 50])
    longitudes = np.array([-121.44977915, -121.43793693, -121.42577458, \
       -121.44646117, -121.76248528, -122.36442489, -122.66268742, \
       -122.64874964, -122.63441767])

#    compute_common_modes(station_file, dataset, direction, latitudes, longitudes)

    lats = [47.20000, 47.30000, 47.40000, 47.50000, 47.60000, 47.70000, \
        47.80000, 47.90000, 48.00000, 48.10000, 48.20000, 48.30000, 48.40000, \
        48.50000, 48.60000, 48.70000]
    lons = [-122.74294, -122.73912, -122.75036, -122.77612, -122.81591, \
        -122.86920, -122.93549, -123.01425, -123.10498, -123.20716, \
        -123.32028, -123.44381, -123.57726, -123.72011, -123.87183, \
        -124.03193]
    radius_GPS = 50
    wavelet = 'LA8'
    J = 10

    compute_wavelets(station_file, lats, lons, radius_GPS, direction, dataset, \
      wavelet, J, latitudes)