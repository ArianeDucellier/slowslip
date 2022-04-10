"""
Script to plot a vespagram-like figure of slow slip
"""

import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
import datetime
import matplotlib.cm as cm
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from math import cos, floor, pi, sin, sqrt
from matplotlib.colors import Normalize
from scipy.io import loadmat
from scipy.signal import detrend
from sklearn import linear_model

import date
import DWT
from MODWT import get_DS, get_scaling, pyramid

def compute_wavelets(station_file, direction, wavelet, J):
    """
    """
    # Read station file
    stations = pd.read_csv(station_file, sep=r'\s{1,}', header=None, \
        engine='python')
    stations.columns = ['name', 'longitude', 'latitude']

    # Time limits
    tmin = 2000 + 909 / 365.25
    tmax = 2000 + 8127 / 365.25

    # Wavelet initialization
    g = get_scaling(wavelet)
    L = len(g)
    (nuH, nuG) = DWT.get_nu(wavelet, J)

    # Read GPS data and compute wavelet transform
    for station in stations['name']:
        filename = '../data/GeoNet/FITS-' + station + '-' + direction + '.csv'

        # Load the data
        data = pd.read_csv(filename)
        data['date-time'] = pd.to_datetime(data['date-time'])
        time = np.array(data['date-time'].apply( \
            lambda x: datetime.date.toordinal(x)))
        cname = ' ' + direction + ' (mm)'

        time = 2000 + (time - 730120) / 365.25
        disp = np.array(data[cname])
        sigma = np.std(disp)

        # Detrend the data
        x = np.reshape(time, (len(time), 1))
        y = np.reshape(disp, (len(disp), 1))
        regr = linear_model.LinearRegression(fit_intercept=True)
        regr.fit(x, y)
        y_pred = regr.predict(x)
        disp = np.reshape(np.array(y - y_pred), (len(disp)))

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

        # Save wavelets into file
        filename = 'MODWT_GPS_NZ/' + station + '_' + direction + '.pkl'
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
        plt.xlim([tmin, tmax])
        plt.ylim([np.min(disp), np.max(disp)])
        plt.legend(loc=3, fontsize=20)
        # Plot details
        for j in range(0, J):
            plt.subplot2grid((J + 2, 1), (j + 1, 0))
            plt.plot(time, D[j], 'k', label='D' + str(j + 1))
            plt.xlim([tmin, tmax])
            plt.ylim(minD, maxD)
            plt.legend(loc=3, fontsize=20)
        # Plot smooth
        plt.subplot2grid((J + 2, 1), (J + 1, 0))
        plt.plot(time, S[J], 'k', label='S' + str(J))
        plt.xlim([tmin, tmax])
        plt.ylim([np.min(disp), np.max(disp)])
        plt.legend(loc=3, fontsize=20)
        plt.xlabel('Time (years)', fontsize=24)
        
        # Save figure
        plt.tight_layout()
        plt.savefig('MODWT_GPS_NZ/' + station + '_' + direction + '.pdf', format='pdf')
        plt.close(1)

if __name__ == '__main__':

    station_file = '../data/GeoNet/stations.txt'
    direction = 'e'
    wavelet = 'LA8'
    J = 10

    compute_wavelets(station_file, direction, wavelet, J)
