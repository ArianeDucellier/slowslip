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

from datetime import datetime
from math import cos, floor, pi, sin, sqrt
from matplotlib.colors import Normalize
from scipy.io import loadmat
from scipy.signal import detrend

import date
import DWT
from MODWT import get_DS, get_scaling, pyramid

def compute_wavelets(station_file, lats, lons, radius, direction, dataset, \
    wavelet, J):
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
    for station in subset['name']:
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
            disp = np.insert(disp, gap[j] + 1, \
                np.random.normal(0.0, sigma, duration[j] - 1))
            gap[j + 1 : ] = gap[j + 1 : ] + duration[j] - 1
        # MODWT
        (W, V) = pyramid(disp, wavelet, J)
        (D, S) = get_DS(disp, W, wavelet, J)
        # Save wavelets into file
        filename = 'tmp/' + dataset + '_' + station + '_' + direction + '.pkl'
        pickle.dump([time, disp, W, V, D, S], open(filename, 'wb'))

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
    time = pickle.load(open('tmp/time.pkl', 'rb'))

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
            open('tmp/tremor_' + str(index) + '.pkl', 'wb'))
        
        maxD = max([np.max(Dj) for Dj in D])
        minD = min([np.min(Dj) for Dj in D])

        # Plot MODWT
        params = {'xtick.labelsize':24,
                  'ytick.labelsize':24}
        pylab.rcParams.update(params)   
        fig = plt.figure(1, figsize=(15, 5 * (J + 3)))
        # Plot data
        plt.subplot2grid((J + 2, 1), (J + 1, 0))
        plt.plot(time, tremor_detrend, 'k', label='Data')
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
        plt.ylim(minD, maxD)
        plt.legend(loc=3, fontsize=14)
        # Save figure
        plt.savefig('tremor_' + str(index) + '.pdf', format='pdf')
        plt.close(1)

def vesp_tremor(station_file, tremor_file, lats, lons, dataset, direction, \
    radius_GPS, radius_tremor, tmin_GPS, tmax_GPS, J, slowness):
    """
    """
    # Read station file
    stations = pd.read_csv(station_file, sep=r'\s{1,}', header=None, engine='python')
    stations.columns = ['name', 'longitude', 'latitude']

    # Read tremor file (A. Ghosh)
#    data = loadmat(tremor_file)
#    mbbp = data['mbbp_cat_d']

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

    # Convert begin and end times for tremor
    (year1, month1, day1, hour1, minute1, second1) = date.day2ymdhms(tmin_GPS)
    (year2, month2, day2, hour2, minute2, second2) = date.day2ymdhms(tmax_GPS)

    # Start figure
    plt.style.use('bmh')
    fig = plt.figure()
    a = 6378.136
    e = 0.006694470

    # Part 1: vespagrams

    # Stations used for figure
    lon_stas = []
    lat_stas = []

    # Loop on latitude and longitude
    for index, (lat, lon) in enumerate(zip(lats, lons)):
        ax1 = plt.subplot2grid((len(lats) + 5, 1), (len(lats) - index - 1, 0))

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

        # Wavelet vectors initialization
        times = []
        disps = []
        Ws = []
        Vs = []
        Ds = []
        Ss = []

        # Read output files from wavelet transform
        for (station, lon_sta, lat_sta) in zip(sub_stations['name'], sub_stations['longitude'], sub_stations['latitude']):
            filename = 'tmp/' + dataset + '_' + station + '_' + direction + '.pkl'
            (time, disp, W, V, D, S) = pickle.load(open(filename, 'rb'))
            if ((np.min(time) < tmax_GPS) and (np.max(time) > tmin_GPS)):
                lon_stas.append(lon_sta)
                lat_stas.append(lat_sta)
                times.append(time)
                disps.append(disp)
                Ws.append(W)
                Vs.append(V)
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

        # Initializations
        time_vesps = []
        vesps = []
        if len(tblocks) > 0:
            time_sta = np.zeros(2 * (len(tblocks) - 1))
            nb_sta = np.zeros(2 * (len(tblocks) - 1))

        # Loop on time blocks
        t0 = 0
        for t in range(0, len(tblocks) - 1):

            # Find time where we look at number of stations
            if ((tbegin[t] <= 0.5 * (tmin_GPS + tmax_GPS)) and (tend[t] >= 0.5 * (tmin_GPS + tmax_GPS))):
                t0 = t

            # Find time period
            for time in times:
                indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
                if (len(indices) > 0):
                    time_subset = time[indices]
                    break
            time_vesps.append(time_subset)

            # Initialize vespagram
            vespagram = np.zeros((len(slowness), len(time_subset)))      

            # Loop on time scales
            nsta = 0
            for (time, D, lat_sta) in zip(times, Ds, sub_stations['latitude']):
                indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
                if (len(indices) > 0):
                    nsta = nsta + 1
                    tj = time[indices]
                    Dj = D[J][indices]
                    for i in range(0, len(slowness)):
                        Dj_interp = np.interp(time_subset + \
                            slowness[i] * (lat_sta - lat), tj, Dj)
                        vespagram[i, :] = vespagram[i, :] + Dj_interp
            vespagram[:, :] = vespagram[:, :] / nsta
            time_sta[2 * t] = tbegin[t]
            time_sta[2 * t + 1] = tend[t]
            nb_sta[2 * t] = nsta
            nb_sta[2 * t + 1] = nsta
            vesps.append(vespagram)

        # Figure
        if len(time_vesps) > 0:
            time_subset = np.concatenate(time_vesps)
            vespagram = np.concatenate(vesps, axis=1)
            plt.contourf(time_subset[(time_subset >= 2009.25) & (time_subset <= 2021.25)], slowness * 365.25 / dy, \
                vespagram[:, (time_subset >= 2009.25) & (time_subset <= 2021.25)], cmap=plt.get_cmap('seismic'), \
                norm=Normalize(vmin=-1.5, vmax=1.5))
            plt.axvline(0.5 * (tmin_GPS + tmax_GPS), color='grey', linewidth=1)
            plt.annotate('{:d} stations'.format(int(nb_sta[2 * t0])), \
                (tmin_GPS + 0.7 * (tmax_GPS - tmin_GPS), 0), fontsize=5)
        plt.xlim([tmin_GPS, tmax_GPS])
        plt.grid(b=None)
        ax1.axes.xaxis.set_ticks([])
        ax1.axes.yaxis.set_ticks([])

    # Part 2: tremor
    ax1 = plt.subplot2grid((len(lats) + 5, 1), (len(lats), 0), rowspan=5)

    # Loop on latitude and longitude
    for index, (lat, lon) in enumerate(zip(lats, lons)):
    
        # Keep only tremor in a given radius
        x = dx * (data['lon'] - lon)
        y = dy * (data['lat'] - lat)
        distance = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0))
        data['distance'] = distance
        tremor = data.loc[data['distance'] <= radius_tremor].copy()
        tremor.reset_index(drop=True, inplace=True)
 
        # Keep only tremor in time interval (A. Wech)
        mask = ((tremor['time '] >= datetime(year1, month1, day1, hour1, minute1, second1)) \
             & (tremor['time '] <= datetime(year2, month2, day2, hour2, minute2, second2)))
        tremor_sub = tremor.loc[mask].copy()
        tremor_sub.reset_index(drop=True, inplace=True)

        # Convert tremor time
        nt = len(tremor_sub)
        time_tremor = np.zeros(nt)
        for i in range(0, nt):
            year = tremor_sub['time '].loc[i].year
            month = tremor_sub['time '].loc[i].month
            day = tremor_sub['time '].loc[i].day
            hour = tremor_sub['time '].loc[i].hour
            minute = tremor_sub['time '].loc[i].minute
            second = tremor_sub['time '].loc[i].second
            time_tremor[i] = date.ymdhms2day(year, month, day, hour, minute, second)    

        # Number of tremors per day
        ntremor = np.zeros(int(floor((tmax_GPS - tmin_GPS) * 365.25)))
        for i in range(0, len(ntremor)):
            for j in range(0, nt):
                if ((time_tremor[j] >= tmin_GPS + (i - 0.5) / 365.25)  and \
                    (time_tremor[j] <= tmin_GPS + (i + 0.5) /365.25)):
                    ntremor[i] = ntremor[i] + 1

        plt.plot(tmin_GPS + (1.0 / 365.0) * np.arange(0, len(ntremor)), ntremor, \
            linewidth=0.5, color=(1 - index / len(lats), 0.8, index / len(lats)))
        plt.ylabel('Number of tremor')
    plt.xlim([tmin_GPS, tmax_GPS])
    plt.xlabel('Time (year)')

    plt.suptitle('Slow slip and tremor from {} to {}'.format(tmin_GPS, tmax_GPS))
    plt.savefig('vespagram_' + str(J + 1) + '_tremor.pdf', format='pdf')
    plt.close(1)

def vesp_map(station_file, tremor_file, tmin_tremor, tmax_tremor, lats, lons, \
    dataset, direction, radius_GPS, tmin_GPS, tmax_GPS, latmin, latmax, lonmin, lonmax, \
    J, slowness):
    """
    """
    # Read station file
    stations = pd.read_csv(station_file, sep=r'\s{1,}', header=None, engine='python')
    stations.columns = ['name', 'longitude', 'latitude']

    # Read tremor file (A. Ghosh)
#    data = loadmat(tremor_file)
#    mbbp = data['mbbp_cat_d']

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

    # Convert begin and end times for tremor
    (year1, month1, day1, hour1, minute1, second1) = date.day2ymdhms(tmin_tremor)
    day_begin = int(floor(date.ymdhms2matlab(year1, month1, day1, hour1, minute1, second1)))
    (year2, month2, day2, hour2, minute2, second2) = date.day2ymdhms(tmax_tremor)
    day_end = int(floor(date.ymdhms2matlab(year2, month2, day2, hour2, minute2, second2)))

    # Start figure
    plt.style.use('bmh')
    fig = plt.figure()
    a = 6378.136
    e = 0.006694470

    # Part 1: vespagrams

    # Stations used for figure
    lon_stas = []
    lat_stas = []

    # Loop on latitude and longitude
    for index, (lat, lon) in enumerate(zip(lats, lons)):
        ax1 = plt.subplot2grid((len(lats), 3), (len(lats) - index - 1, 0))

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

        # Wavelet vectors initialization
        times = []
        disps = []
        Ws = []
        Vs = []
        Ds = []
        Ss = []

        # Read output files from wavelet transform
        for (station, lon_sta, lat_sta) in zip(sub_stations['name'], sub_stations['longitude'], sub_stations['latitude']):
            filename = 'tmp/' + dataset + '_' + station + '_' + direction + '.pkl'
            (time, disp, W, V, D, S) = pickle.load(open(filename, 'rb'))
            if ((np.min(time) < tmax_GPS) and (np.max(time) > tmin_GPS)):
                lon_stas.append(lon_sta)
                lat_stas.append(lat_sta)
                times.append(time)
                disps.append(disp)
                Ws.append(W)
                Vs.append(V)
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

        # Initializations
        time_vesps = []
        vesps = []
        if len(tblocks) > 0:
            time_sta = np.zeros(2 * (len(tblocks) - 1))
            nb_sta = np.zeros(2 * (len(tblocks) - 1))

        # Loop on time blocks
        t0 = 0
        for t in range(0, len(tblocks) - 1):

            # Find time where we look at number of stations
            if ((tbegin[t] <= 0.5 * (tmin_GPS + tmax_GPS)) and (tend[t] >= 0.5 * (tmin_GPS + tmax_GPS))):
                t0 = t

            # Find time period
            for time in times:
                indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
                if (len(indices) > 0):
                    time_subset = time[indices]
                    break
            time_vesps.append(time_subset)

            # Initialize vespagram
            vespagram = np.zeros((len(slowness), len(time_subset)))      

            # Loop on time scales
            nsta = 0
            for (time, D, lat_sta) in zip(times, Ds, sub_stations['latitude']):
                indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
                if (len(indices) > 0):
                    nsta = nsta + 1
                    tj = time[indices]
                    Dj = D[J][indices]
                    for i in range(0, len(slowness)):
                        Dj_interp = np.interp(time_subset + \
                            slowness[i] * (lat_sta - lat), tj, Dj)
                        vespagram[i, :] = vespagram[i, :] + Dj_interp
            vespagram[:, :] = vespagram[:, :] / nsta
            time_sta[2 * t] = tbegin[t]
            time_sta[2 * t + 1] = tend[t]
            nb_sta[2 * t] = nsta
            nb_sta[2 * t + 1] = nsta
            vesps.append(vespagram)

        # Figure
        if len(time_vesps) > 0:
            time_subset = np.concatenate(time_vesps)
            vespagram = np.concatenate(vesps, axis=1)

            max_value = np.max(vespagram[:, (time_subset >= tmin_GPS) & (time_subset <= tmax_GPS)])
            min_value = np.min(vespagram[:, (time_subset >= tmin_GPS) & (time_subset <= tmax_GPS)])
            max_index = np.argmax(vespagram[:, (time_subset >= tmin_GPS) & (time_subset <= tmax_GPS)])
            min_index = np.argmin(vespagram[:, (time_subset >= tmin_GPS) & (time_subset <= tmax_GPS)])
            (imax, jmax) = np.unravel_index(max_index, np.array(vespagram[:, (time_subset >= tmin_GPS) & (time_subset <= tmax_GPS)]).shape)
            (imin, jmin) = np.unravel_index(min_index, np.array(vespagram[:, (time_subset >= tmin_GPS) & (time_subset <= tmax_GPS)]).shape)
            print(index, max_value, min_value, time_subset[(time_subset >= tmin_GPS) & (time_subset <= tmax_GPS)][jmax], time_subset[(time_subset >= tmin_GPS) & (time_subset <= tmax_GPS)][jmin])

            plt.contourf(time_subset[(time_subset >= 2009.25) & (time_subset <= 2021.25)], slowness * 365.25 / dy, \
                vespagram[:, (time_subset >= 2009.25) & (time_subset <= 2021.25)], cmap=plt.get_cmap('seismic'), \
                norm=Normalize(vmin=-1.5, vmax=1.5))
#                levels = np.linspace(-1.2, 1.2, 25))
            plt.axvline(0.5 * (tmin_GPS + tmax_GPS), color='grey', linewidth=1)
            plt.annotate('{:d} stations'.format(int(nb_sta[2 * t0])), \
                (tmin_GPS + 0.7 * (tmax_GPS - tmin_GPS), 0), fontsize=5)
        plt.xlim([tmin_GPS, tmax_GPS])
        plt.grid(b=None)
        if index == 0:
            plt.xlabel('Time (year)')
        if index != 0:    
            ax1.axes.xaxis.set_ticks([])
        ax1.axes.yaxis.set_ticks([])

    # Part 2: map of tremor
    ax2 = plt.subplot2grid((len(lats), 3), (0, 1), rowspan=len(lats), colspan=2, projection=ccrs.Mercator())
    shapename = 'ocean'
    ocean_shp = shapereader.natural_earth(resolution='10m',
                                          category='physical',
                                          name=shapename)
    shapename = 'land'
    land_shp = shapereader.natural_earth(resolution='10m',
                                       category='physical',
                                       name=shapename)
    ax2.set_extent([lonmin, lonmax, latmin, latmax], ccrs.Geodetic())
    ax2.gridlines(linestyle=":")
    for myfeature in shapereader.Reader(ocean_shp).geometries(): 
        ax2.add_geometries([myfeature], ccrs.PlateCarree(), facecolor='#E0FFFF', edgecolor='black', alpha=0.5)
    for myfeature in shapereader.Reader(land_shp).geometries(): 
        ax2.add_geometries([myfeature], ccrs.PlateCarree(), facecolor='#FFFFE0', edgecolor='black', alpha=0.5)

    # Keep only tremor on map (A. Ghosh)
#    lat_tremor = mbbp[:, 2]
#    lon_tremor = mbbp[:, 3]
#    find = np.where((lat_tremor >= latmin) & (lat_tremor <= latmax) \
#                  & (lon_tremor >= lonmin) & (lon_tremor <= lonmax))
#    tremor = mbbp[find, :][0, :, :]

    # Keep only tremor on map (A. Wech)
    mask = ((data['lat'] >= latmin) & (data['lat'] <= latmax) \
          & (data['lon'] >= lonmin) & (data['lon'] <= lonmax))
    tremor = data.loc[mask].copy()
    tremor.reset_index(drop=True, inplace=True)

    # Keep only tremor in time interval (A. Ghosh)
#    find = np.where((tremor[:, 0] >= day_begin) & (tremor[:, 0] < day_end))
#    tremor_sub = tremor[find, :][0, :, :]

   # Keep only tremor in time interval (A. Wech)
    mask = ((tremor['time '] >= datetime(year1, month1, day1, hour1, minute1, second1)) \
          & (tremor['time '] <= datetime(year2, month2, day2, hour2, minute2, second2)))
    tremor_sub = tremor.loc[mask].copy()
    tremor_sub.reset_index(drop=True, inplace=True)

    # Convert tremor time (A. Ghosh)
#    nt = np.shape(tremor_sub)[0]
#    time_tremor = np.zeros(nt)
#    for i in range(0, nt):
#        (year, month, day, hour, minute, second) = date.matlab2ymdhms(tremor_sub[i, 0])
#        time_tremor[i] = date.ymdhms2day(year, month, day, hour, minute, second)

    # Convert tremor time (A. Wech)
    nt = len(tremor_sub)
    time_tremor = np.zeros(nt)
    for i in range(0, nt):
        year = tremor_sub['time '].loc[i].year
        month = tremor_sub['time '].loc[i].month
        day = tremor_sub['time '].loc[i].day
        hour = tremor_sub['time '].loc[i].hour
        minute = tremor_sub['time '].loc[i].minute
        second = tremor_sub['time '].loc[i].second
        time_tremor[i] = date.ymdhms2day(year, month, day, hour, minute, second)    

    # Plot tremor on map (A. Ghosh)
#    ax2.scatter(tremor_sub[:, 3], tremor_sub[:, 2], c=time_tremor, s=2, transform=ccrs.PlateCarree())

    # Plot tremor on map (A. Wech)
    ax2.scatter(tremor_sub['lon'], tremor_sub['lat'], c=time_tremor, s=2, transform=ccrs.PlateCarree())

    # Plot GPS stations on map
    ax2.scatter(np.array(lon_stas), np.array(lat_stas), c='r', marker='^', s=20, transform=ccrs.PlateCarree())

    # Plot centers at which we look for GPS data on map
    ax2.scatter(np.array(lons), np.array(lats), c='k', marker='x', s=20, transform=ccrs.PlateCarree())

    ax2.set_title('Tremor from {:.2f} to {:.2f}: {:d} in {:d} days'.format( \
        tmin_tremor, tmax_tremor, np.shape(tremor_sub)[0], \
        int(floor(365.25 * (tmax_tremor - tmin_tremor)))), fontsize=10)

    plt.suptitle('Slow slip and tremor at {:.2f}'.format(0.5 * (tmin_GPS + tmax_GPS)))
    plt.savefig('vespagram_' + str(J + 1) + '.pdf', format='pdf')
    plt.close(1)

if __name__ == '__main__':

    station_file = '../data/PANGA/stations.txt'
    tremor_file = '../data/tremor/mbbp_cat_d_forHeidi'
    radius_GPS = 50
    radius_tremor = 50
    direction = 'lon'
    dataset = 'cleaned'
    wavelet = 'LA8'
    J = 8
    slowness = np.arange(-0.1, 0.105, 0.005)
    lats = [47.20000, 47.30000, 47.40000, 47.50000, 47.60000, 47.70000, \
        47.80000, 47.90000, 48.00000, 48.10000, 48.20000, 48.30000, 48.40000, \
        48.50000, 48.60000, 48.70000]
    lons = [-122.74294, -122.73912, -122.75036, -122.77612, -122.81591, \
        -122.86920, -122.93549, -123.01425, -123.10498, -123.20716, \
        -123.32028, -123.44381, -123.57726, -123.72011, -123.87183, \
        -124.03193]
    tmin_GPS = 2017.28
    tmax_GPS = 2018.28
    tmin_tremor = 2017.75
    tmax_tremor = 2017.81
    lonmin = -125.4
    lonmax = -121.4
    latmin = 46.3
    latmax = 49.6
    j = 4

#    compute_wavelets(station_file, lats, lons, radius_GPS, direction, dataset, \
#        wavelet, J)

    compute_wavelets_tremor(lats, lons, radius_tremor, wavelet, J)

#    vesp_tremor(station_file, tremor_file, lats, lons, dataset, direction, \
#        radius_GPS, radius_tremor, tmin_GPS, tmax_GPS, j - 1, slowness)

#    vesp_map(station_file, tremor_file, tmin_tremor, tmax_tremor, lats, lons, \
#        dataset, direction, radius_GPS, tmin_GPS, tmax_GPS, latmin, latmax, lonmin, lonmax, \
#        j - 1, slowness)
