"""
Script to compute MODWT of cumulative tremor count
"""

import datetime
import matplotlib.cm as cm
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from datetime import datetime
from math import cos, floor, pi, sin, sqrt
from matplotlib.colors import Normalize
from scipy.io import loadmat
from scipy.signal import butter, detrend, lfilter

import date
import DWT
from MODWT import get_DS, get_scaling, pyramid

def compute_wavelets_tremor(lats, lons, radius_tremor, wavelet, J):
    """
    """
    # Read tremor files (A. Wech)
    data_2009 = pd.read_csv('../data/tremor/tremor_events-2009-08-06T00 00 00-2009-12-31T23 59 59.csv')
    data_2009['time '] = pd.to_datetime(data_2009['time '], format='%Y-%m-%d %H:%M:%S')
    data_2010 = pd.read_csv('../data/tremor/tremor_events-2010-01-01T00 00 00-2010-12-31T23 59 59.csv')
    data_2010['time '] = pd.to_datetime(data_2010['time '], format='%Y-%m-%d %H:%M:%S')
    data_2011 = pd.read_csv('../data/tremor/tremor_events-2011-01-01T00 00 00-2011-12-31T23 59 59.csv')
    data_2011['time '] = pd.to_datetime(data_2011['time '], format='%Y-%m-%d %H:%M:%S')
    data_2012 = pd.read_csv('../data/tremor/tremor_events-2012-01-01T00 00 00-2012-12-31T23 59 59.csv')
    data_2012['time '] = pd.to_datetime(data_2012['time '], format='%Y-%m-%d %H:%M:%S')
    data_2013 = pd.read_csv('../data/tremor/tremor_events-2013-01-01T00 00 00-2013-12-31T23 59 59.csv')
    data_2013['time '] = pd.to_datetime(data_2013['time '], format='%Y-%m-%d %H:%M:%S')
    data_2014 = pd.read_csv('../data/tremor/tremor_events-2014-01-01T00 00 00-2014-12-31T23 59 59.csv')
    data_2014['time '] = pd.to_datetime(data_2014['time '], format='%Y-%m-%d %H:%M:%S')
    data_2015 = pd.read_csv('../data/tremor/tremor_events-2015-01-01T00 00 00-2015-12-31T23 59 59.csv')
    data_2015['time '] = pd.to_datetime(data_2015['time '], format='%Y-%m-%d %H:%M:%S')
    data_2016 = pd.read_csv('../data/tremor/tremor_events-2016-01-01T00 00 00-2016-12-31T23 59 59.csv')
    data_2016['time '] = pd.to_datetime(data_2016['time '], format='%Y-%m-%d %H:%M:%S')
    data_2017 = pd.read_csv('../data/tremor/tremor_events-2017-01-01T00 00 00-2017-12-31T23 59 59.csv')
    data_2017['time '] = pd.to_datetime(data_2017['time '], format='%Y-%m-%d %H:%M:%S')
    data_2018 = pd.read_csv('../data/tremor/tremor_events-2018-01-01T00 00 00-2018-12-31T23 59 59.csv')
    data_2018['time '] = pd.to_datetime(data_2018['time '], format='%Y-%m-%d %H:%M:%S')
    data_2019 = pd.read_csv('../data/tremor/tremor_events-2019-01-01T00 00 00-2019-12-31T23 59 59.csv')
    data_2019['time '] = pd.to_datetime(data_2019['time '], format='%Y-%m-%d %H:%M:%S')
    data_2020 = pd.read_csv('../data/tremor/tremor_events-2020-01-01T00 00 00-2020-12-31T23 59 59.csv')
    data_2020['time '] = pd.to_datetime(data_2020['time '], format='%Y-%m-%d %H:%M:%S')
    data_2021 = pd.read_csv('../data/tremor/tremor_events-2021-01-01T00 00 00-2021-04-29T23 59 59.csv')
    data_2021['time '] = pd.to_datetime(data_2020['time '], format='%Y-%m-%d %H:%M:%S')
    data = pd.concat([data_2009, data_2010, data_2011, data_2012, data_2013, \
        data_2014, data_2015, data_2016, data_2017, data_2018, data_2019, \
        data_2020, data_2021])
    data.reset_index(drop=True, inplace=True)

    # To convert lat/lon into kilometers
    a = 6378.136
    e = 0.006694470

    # Time vector
    time = pickle.load(open('time.pkl', 'rb'))

    # Loop on latitude and longitude
    for index, (lat, lon) in enumerate(zip(lats, lons)):

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
            year = tremor['time '].loc[i].year
            month = tremor['time '].loc[i].month
            day = tremor['time '].loc[i].day
            hour = tremor['time '].loc[i].hour
            minute = tremor['time '].loc[i].minute
            second = tremor['time '].loc[i].second
            time_tremor[i] = date.ymdhms2day(year, month, day, hour, minute, second)

        # Interpolate
        tremor = np.interp(time, np.sort(time_tremor), (1.0 / nt) * np.arange(0, len(time_tremor)))
        tremor_detrend = detrend(tremor)

        # MODWT
        (W, V) = pyramid(tremor_detrend, wavelet, J)
        (D, S) = get_DS(tremor_detrend, W, wavelet, J)

        # Save wavelets
        pickle.dump([time, tremor_detrend, W, V, D, S], \
            open('MODWT_tremor/tremor_' + str(index) + '.pkl', 'wb'))

        # Start figure
        params = {'xtick.labelsize':24,
                  'ytick.labelsize':24}
        pylab.rcParams.update(params)   
        fig = plt.figure(1, figsize=(10, 3 * (J + 3)))

        maxD = max([np.max(Dj) for Dj in D])
        minD = min([np.min(Dj) for Dj in D])

        # Plot data
        plt.subplot2grid((J + 2, 1), (J + 1, 0))
        plt.plot(time, tremor_detrend, 'k', label='Data')
        plt.xlabel('Time (years)', fontsize=24)
        plt.xlim([2009.25, 2021.25])
        plt.ylim([np.min(tremor_detrend), np.max(tremor_detrend)])
        plt.legend(loc=3, fontsize=20)
        # Plot details
        for j in range(0, J):
            plt.subplot2grid((J + 2, 1), (J - j, 0))
            plt.plot(time, D[j], 'k', label='D' + str(j + 1))
            plt.xlim([2009.25, 2021.25])
            plt.ylim(minD, maxD)
            plt.legend(loc=3, fontsize=20)
        # Plot smooth
        plt.subplot2grid((J + 2, 1), (0, 0))
        plt.plot(time, S[J], 'k', label='S' + str(J))
        plt.xlim([2009.25, 2021.25])
        plt.ylim([np.min(tremor_detrend), np.max(tremor_detrend)])
        plt.legend(loc=3, fontsize=20)
        
        # Save figure
        plt.tight_layout()
        plt.savefig('MODWT_tremor/tremor_' + str(index) + '.eps', format='eps')
        plt.close(1)

if __name__ == '__main__':

    lats = [47.20000, 47.30000, 47.40000, 47.50000, 47.60000, 47.70000, \
        47.80000, 47.90000, 48.00000, 48.10000, 48.20000, 48.30000, 48.40000, \
        48.50000, 48.60000, 48.70000]
    lons = [-122.74294, -122.73912, -122.75036, -122.77612, -122.81591, \
        -122.86920, -122.93549, -123.01425, -123.10498, -123.20716, \
        -123.32028, -123.44381, -123.57726, -123.72011, -123.87183, \
        -124.03193]
    radius_tremor = 50
    wavelet = 'LA8'
    J = 8

    compute_wavelets_tremor(lats, lons, radius_tremor, wavelet, J)
