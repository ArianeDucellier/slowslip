"""
Script to compute MODWT of cumulative tremor count
"""

#import datetime
#import matplotlib.cm as cm
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

#from datetime import datetime
from math import cos, floor, pi, sin, sqrt
#from matplotlib.colors import Normalize
from scipy.io import loadmat
from scipy.signal import detrend

import date
#import DWT
from MODWT import get_DS, pyramid

def compute_wavelets_tremor(station_file, lats, lons, dataset, direction, \
    radius_GPS, radius_tremor, wavelet, J):
    """
    """
    # Read station file
    stations = pd.read_csv(station_file, sep=r'\s{1,}', header=None, engine='python')
    stations.columns = ['name', 'longitude', 'latitude']

    # Read tremor files (K. Creager)
    data = loadmat('../data/tremor/SummaryLatestMerge.mat')
    Summary = data['Summary']
    TREMall = data['TREMall']
    data = pd.DataFrame(columns=['lat', 'lon', 'depth', 'time'])
    for k in range(0, len(Summary[0][0][3])):
        indices = Summary[0][0][3][k]
        lon = np.reshape(TREMall[0][0][2][indices[0] - 1], (-1))
        lat = np.reshape(TREMall[0][0][1][indices[0] - 1], (-1))
        depth = np.reshape(TREMall[0][0][3][indices[0] - 1], (-1))
        time = np.reshape(TREMall[0][0][0][indices[0] - 1], (-1))
        df = pd.DataFrame({'lat': lat, 'lon': lon, 'depth': depth, 'time': time})
        data = pd.concat([data, df])
    data.reset_index(drop=True, inplace=True)

    # To convert lat/lon into kilometers
    a = 6378.136
    e = 0.006694470

    # Loop on latitude and longitude
    for index, (lat, lon) in enumerate(zip(lats, lons)):

        # Keep only stations in a given radius
        dx = (pi / 180.0) * a * cos(lat * pi / 180.0) / sqrt(1.0 - e * e * \
            sin(lat * pi / 180.0) * sin(lat * pi / 180.0))
        dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * \
            sin(lat * pi / 180.0) * sin(lat * pi / 180.0)) ** 1.5)
        x = dx * (stations['longitude'] - lon)
        y = dy * (stations['latitude'] - lat)
        stations['distance'] = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0))
        mask = stations['distance'] <= radius_GPS
        sub_stations = stations.loc[mask].copy()
        sub_stations.reset_index(drop=True, inplace=True)

        # Time vectors initialization
        times = []

        # Read output files from wavelet transform
        for (station, lon_sta, lat_sta) in zip(sub_stations['name'], sub_stations['longitude'], sub_stations['latitude']):
            filename = 'MODWT_GPS/' + dataset + '_' + station + '_' + direction + '.pkl'
            (time, disp, W, V, D, S) = pickle.load(open(filename, 'rb'))
            if ((np.min(time) <= 2021.5) and (np.max(time) >= 2006.0)):
                times.append(time)

        # Divide into time blocks
        tblocks = []
        for time in times:
            tblocks.append(np.min(time))
            tblocks.append(np.max(time))
        tblocks = sorted(set(tblocks))
        tbegin = tblocks[0 : -1]
        tend = tblocks[1 : ]

        # Initializations
        times_stacked = []

        # Loop on time blocks
        for t in range(0, len(tblocks) - 1):

            # Find time period
            for time in times:
                indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
                if (len(indices) > 0):
                    time_subset = time[indices]
                    break
            times_stacked.append(time_subset)

        # Concatenate times and disps
        times_stacked = np.concatenate(times_stacked)
        times_stacked = times_stacked[(times_stacked  >= 2006.0) & (times_stacked  <= 2021.5)]

        # Keep only tremor in a given radius
        dx = (pi / 180.0) * a * cos(lat * pi / 180.0) / sqrt(1.0 - e * e * \
            sin(lat * pi / 180.0) * sin(lat * pi / 180.0))
        dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * \
            sin(lat * pi / 180.0) * sin(lat * pi / 180.0)) ** 1.5)
        x = dx * (data['lon'] - lon)
        y = dy * (data['lat'] - lat)
        distance = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0))
        data['distance'] = distance
        tremor = data.loc[data['distance'] <= radius_tremor].copy()
        tremor.reset_index(drop=True, inplace=True)
 
        # Convert tremor time
        nt = len(tremor)
        time_tremor = np.zeros(nt)
        for i in range(0, nt):
            (year, month, day, hour, minute, second) = date.matlab2ymdhms(tremor['time'].loc[i])
            time_tremor[i] = date.ymdhms2day(year, month, day, hour, minute, second)

        # Interpolate
        tremor = np.interp(times_stacked, np.sort(time_tremor), (1.0 / nt) * np.arange(0, len(time_tremor)))
        tremor_detrend = detrend(tremor)

        # MODWT
        (W, V) = pyramid(tremor_detrend, wavelet, J)
        (D, S) = get_DS(tremor_detrend, W, wavelet, J)

        # Save wavelets
        pickle.dump([times_stacked, tremor_detrend, W, V, D, S], \
            open('MODWT_tremor_longer/tremor_' + str(index) + '.pkl', 'wb'))

        # Start figure
        params = {'xtick.labelsize':24,
                  'ytick.labelsize':24}
        pylab.rcParams.update(params)   
#        fig = plt.figure(1, figsize=(10, 3 * (J + 3)))
        fig = plt.figure(1, figsize=(30, 3 * 4))

        maxD = max([np.max(Dj) for Dj in D])
        minD = min([np.min(Dj) for Dj in D])

        # Plot data
#        plt.subplot2grid((J + 2, 1), (0, 0))
        ax = plt.subplot2grid((4, 3), (0, 0))
        plt.plot(times_stacked, tremor_detrend, 'k', label='Data')
        plt.xlim([2006.0, 2021.5])
        plt.ylim([np.min(tremor_detrend), np.max(tremor_detrend)])
        plt.legend(loc=3, fontsize=20)
        ax.axes.xaxis.set_ticklabels([])
        # Plot details
        for j in range(0, J):
#            plt.subplot2grid((J + 2, 1), (j + 1, 0))
            if j < 3:
                ax = plt.subplot2grid((4, 3), (j + 1, 0))
            elif j < 7:
                ax = plt.subplot2grid((4, 3), (j - 3, 1))
            else:
                ax = plt.subplot2grid((4, 3), (j - 7, 2))
            plt.plot(times_stacked, D[j], 'k', label='D' + str(j + 1))
            plt.xlim([2006.0, 2021.5])
            plt.ylim(minD, maxD)
            plt.legend(loc=3, fontsize=20)
            if ((j == 2) or (j == 6)):
                plt.xlabel('Time (years)', fontsize=24)
            else:
                ax.axes.xaxis.set_ticklabels([])
            if j >= 3:
                ax.axes.yaxis.set_ticklabels([])
        # Plot smooth
#        plt.subplot2grid((J + 2, 1), (J + 1, 0))
        ax = plt.subplot2grid((4, 3), (3, 2))
        plt.plot(times_stacked, S[J], 'k', label='S' + str(J))
        plt.xlim([2006.0, 2021.5])
        plt.ylim([np.min(tremor_detrend), np.max(tremor_detrend)])
        plt.legend(loc=3, fontsize=20)
        plt.xlabel('Time (years)', fontsize=24)
        ax.axes.yaxis.set_ticklabels([])
        
        # Save figure
        plt.tight_layout()
        plt.savefig('MODWT_tremor_longer/tremor_' + str(index) + '.eps', format='eps')
        plt.close(1)

if __name__ == '__main__':

    station_file = '../data/PANGA/stations.txt'
    lats = [47.20000, 47.30000, 47.40000, 47.50000, 47.60000, 47.70000, \
        47.80000, 47.90000, 48.00000, 48.10000, 48.20000, 48.30000, 48.40000, \
        48.50000, 48.60000, 48.70000]
    lons = [-122.74294, -122.73912, -122.75036, -122.77612, -122.81591, \
        -122.86920, -122.93549, -123.01425, -123.10498, -123.20716, \
        -123.32028, -123.44381, -123.57726, -123.72011, -123.87183, \
        -124.03193]
    direction = 'lon'
    dataset = 'cleaned'
    radius_GPS = 50
    radius_tremor = 50
    wavelet = 'LA8'
    J = 10

    compute_wavelets_tremor(station_file, lats, lons, dataset, direction, \
    radius_GPS, radius_tremor, wavelet, J)
