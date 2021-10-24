"""
Script to cross-correlate GPS and tremor MODWT
"""

import datetime
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from datetime import datetime
from math import cos, floor, pi, sin, sqrt
from matplotlib.colors import Normalize

import date
import DWT
from MODWT import get_DS, get_scaling, pyramid

def correlate(X, Y, ncor):
    """
    """
    N = len(X)
    cc = np.zeros(2 * ncor + 1)
    for i in range(0, 2 * ncor + 1):
        if i < ncor:
            input1 = X[(ncor - i) : N]
            input2 = Y[0 : (N - ncor + i)]
        else:
            input1 = X[0 : (N - i + ncor)]
            input2 = Y[(i - ncor) : N]
        cc[i] = np.mean((input1 - np.mean(input1)) * (input2 - np.mean(input2))) / (np.std(input1) * np.std(input2))
    return cc

def compare_tremor_GPS(station_file, lats, lons, dataset, direction, \
    radius_GPS, radius_tremor, slowness, wavelet, J, J0):
    """
    """
    # Read station file
    stations = pd.read_csv(station_file, sep=r'\s{1,}', header=None, engine='python')
    stations.columns = ['name', 'longitude', 'latitude']

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

    a = 6378.136
    e = 0.006694470

    # Start figure
    plt.style.use('bmh')
    fig = plt.figure(figsize=(5, 16))

    # Loop on latitude and longitude
    for index, (lat, lon) in enumerate(zip(lats, lons)):
        ax1 = plt.subplot2grid((len(lats), 1), (len(lats) - index - 1, 0))

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
            if ((np.min(time) <= 2021.25) and (np.max(time) >= 2009.25)):
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
        times_stacked = []
        disps_stacked = []
        vesps = []

        # Loop on time blocks
        for t in range(0, len(tblocks) - 1):

            # Find time period
            for time in times:
                indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
                if (len(indices) > 0):
                    time_subset = time[indices]
                    break
            times_stacked.append(time_subset)

            # Initialize vespagram
            vespagram = np.zeros((len(slowness), len(time_subset)))      

            # Loop on stations
            nsta = 0
            Dj_stacked = np.zeros(len(time_subset))
            for (time, D, lat_sta) in zip(times, Ds, sub_stations['latitude']):
                indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
                if (len(indices) > 0):
                    nsta = nsta + 1
                    tj = time[indices]
                    Dj = D[J][indices]
                    Dj_interp = np.interp(time_subset, tj, Dj)
                    Dj_stacked =  Dj_stacked + Dj_interp
                    for i in range(0, len(slowness)):
                        Dj_interp = np.interp(time_subset + \
                            slowness[i] * (lat_sta - lat), tj, Dj)
                        vespagram[i, :] = vespagram[i, :] + Dj_interp
            Dj_stacked = Dj_stacked / nsta
            vespagram[:, :] = vespagram[:, :] / nsta
            disps_stacked.append(Dj_stacked)
            vesps.append(vespagram)

        # Concatenate times and disps
        times_stacked = np.concatenate(times_stacked)
        disps_stacked = np.concatenate(disps_stacked)
        vespagram = np.concatenate(vesps, axis=1)

        # Filter displacement
        disps_stacked = disps_stacked[(times_stacked  >= 2009.25) & (times_stacked <= 2021.25)]
        vespagram = vespagram[:, (times_stacked >= 2009.25) & (times_stacked <= 2021.25)]
        times_stacked = times_stacked[(times_stacked  >= 2009.25) & (times_stacked  <= 2021.25)]

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
        tremor = np.interp(times_stacked, np.sort(time_tremor), (1.0 / nt) * np.arange(0, len(time_tremor)))

        # Detrend
        coeffs = np.polyfit(times_stacked, tremor, 1)
        p = np.poly1d(coeffs)
        tremor_detrend = tremor - p(times_stacked)

        # MODWT
        (W, V) = pyramid(tremor_detrend, wavelet, J0)
        (D, S) = get_DS(tremor_detrend, W, wavelet, J0)

        # Save tremor MODWT
        filename = 'tremor/loc' + str(index) + '.pkl'
        pickle.dump([times_stacked, tremor, tremor_detrend, W, V, D, S], \
            open(filename, 'wb'))

        cc = np.zeros(len(D))
        # Cross-correlation
        for (j, Dj) in enumerate(D):
            cc[j] = np.max(np.abs(correlate(disps_stacked, Dj, 20)))
        best = np.argmax(cc)

        # Figure
        if len(times_stacked) > 0:
            plt.contourf(times_stacked, slowness * 365.25 / dy, \
                vespagram, cmap=plt.get_cmap('seismic'), \
                norm=Normalize(vmin=-1.5, vmax=1.5))
            norm_min = np.min(slowness * 365.25 / dy) / np.min(D[best])
            norm_max = np.max(slowness * 365.25 / dy) / np.max(D[best])
            plt.plot(times_stacked, min(norm_min, norm_max) * D[best], color='grey', linewidth=1)
            plt.annotate('{:d}th level: cc = {:0.2f}'.format(best + 1, np.max(cc)), \
                (2009.25 + 0.8 * (2021.25 - 2009.25), 0.06 * 365.25 / dy), fontsize=5)
        plt.xlim([2009.25, 2021.25])
        plt.grid(b=None)
        if index == 0:
            plt.xlabel('Time (year)')
        if index != 0:    
            ax1.axes.xaxis.set_ticks([])
        ax1.axes.yaxis.set_ticks([])

    plt.suptitle('Slow slip and tremor from {} to {}'.format(2009.25, 2021.25))
    plt.savefig('comparison_' + str(J + 1) + '.pdf', format='pdf')
    plt.close(1)

def correlate_tremor_GPS(station_file, lats, lons, dataset, direction, \
    radius_GPS, J, t0, nt):
    """
    """
    # Read station file
    stations = pd.read_csv(station_file, sep=r'\s{1,}', header=None, engine='python')
    stations.columns = ['name', 'longitude', 'latitude']

    a = 6378.136
    e = 0.006694470

    # Start figure
    plt.style.use('bmh')
    fig = plt.figure(figsize=(5, 32))

    # Loop on latitude and longitude
    for index, (lat, lon) in enumerate(zip(lats, lons)):
        ax1 = plt.subplot2grid((len(lats), 1), (len(lats) - index - 1, 0))

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
            if ((np.min(time) <= 2021.25) and (np.max(time) >= 2009.25)):
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
        times_stacked = []
        disps_stacked = []
        vesps = []

        # Loop on time blocks
        for t in range(0, len(tblocks) - 1):

            # Find time period
            for time in times:
                indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
                if (len(indices) > 0):
                    time_subset = time[indices]
                    break
            times_stacked.append(time_subset)

            # Loop on stations
            nsta = 0
            Dj_stacked = np.zeros(len(time_subset))
            for (time, D, lat_sta) in zip(times, Ds, sub_stations['latitude']):
                indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
                if (len(indices) > 0):
                    nsta = nsta + 1
                    tj = time[indices]
                    Dj = D[J][indices]
                    Dj_interp = np.interp(time_subset, tj, Dj)
                    Dj_stacked =  Dj_stacked + Dj_interp
            Dj_stacked = Dj_stacked / nsta
            disps_stacked.append(Dj_stacked)

        # Concatenate times and disps
        times_stacked = np.concatenate(times_stacked)
        disps_stacked = np.concatenate(disps_stacked)

        # Filter displacement
        disps_stacked = disps_stacked[(times_stacked  >= 2009.25) & (times_stacked <= 2021.25)]
        times_stacked = times_stacked[(times_stacked  >= 2009.25) & (times_stacked  <= 2021.25)]

        # Read MODWT of tremor
        filename = 'tremor/loc' + str(index) + '.pkl'
        MODWT_tremor = pickle.load(open(filename, 'rb'))
        D_tremor = MODWT_tremor[5]

        # Choose time
        i0 = np.argmin(np.abs(times_stacked - t0))
        input1 = disps_stacked[i0 - nt : i0 + nt]
        input2 = D_tremor[J][i0 - nt : i0 + nt]
        cc = correlate(input1, -input2, 20)[20]
        print(lat, lon, cc)

        # Figure
        if len(times_stacked) > 0:
            norm_min = np.min(input1)
            norm_max = np.max(input1)
            plt.plot(times_stacked[i0 - nt : i0 + nt], input1 / min(norm_min, norm_max), color='blue')
            norm_min = np.min(input2)
            norm_max = np.max(input2)
            plt.plot(times_stacked[i0 - nt : i0 + nt],  -input2 / min(norm_min, norm_max), color='red')
            plt.annotate('cc = {:0.2f}'.format(cc), \
                (times_stacked[i0], 0.5), fontsize=10)
        if index == 0:
            plt.xlabel('Time (year)')

    plt.suptitle('Slow slip and tremor from {} to {}'.format(times_stacked[i0 - nt], times_stacked[i0 + nt]))
    plt.savefig('correlation_' + str(J + 1) + '.pdf', format='pdf')
    plt.close(1)

def plot_correlations():
    """
    """
    times = [2009.754, 2010.631, 2011.629, 2012.234, 2012.693, 2013.755, \
             2014.564, 2014.927, 2015.992, 2017.244, 2017.713, 2018.508, \
             2019.214, 2019.762, 2020.018, 2020.791, 2021.118]

    for level in range(4, 9):
        
        df = pd.DataFrame(columns={'latitude', 'longitude', 'cc', 'time'})
        fig = plt.figure(1, figsize=(10, 10))

        ibegin = 1
        iend = 17
        if level == 8:
            ibegin = 2
            iend = 15
        elif (level == 7) or (level == 6):
            iend = 16

        for event in range(ibegin, iend + 1):
            df_event = pd.read_csv('events/event' + str(event) + '.txt', \
                skiprows=[0, 17, 18, 35, 36, 53, 54, 71, 72], sep=' ', header=None)
            df_event.columns = ['latitude', 'longitude', 'cc']
            if event == 17:
                first = (5 - level) * 16
                last = (6 - level) * 16 - 1
            elif (event == 1) or (event == 16):
                first = (7 - level) * 16
                last = (8 - level) * 16 - 1
            else:
                first = (8 - level) * 16
                last = (9 - level) * 16 - 1
            df_event = df_event.loc[first : last]
            time = np.repeat(times[event - 1], 16)
            df_event['time'] = time
            df = pd.concat([df, df_event])
        
        plt.scatter(df['time'], df['latitude'], s=50, c=-df['cc'], \
            cmap=plt.get_cmap('hot'), vmin=-1, vmax=1)
        plt.xlabel('Time (year)')
        plt.ylabel('Latitude')
        plt.title('Correlations for level {:d}'.format(level))
        plt.savefig('events_' + str(level) + '.pdf', format='pdf')
        plt.close(1)        

def plot_GPS(station_file, lats, lons, dataset, direction, radius_GPS, J, threshold, events, small_events):
    """
    """
    # Read station file
    stations = pd.read_csv(station_file, sep=r'\s{1,}', header=None, engine='python')
    stations.columns = ['name', 'longitude', 'latitude']

    a = 6378.136
    e = 0.006694470

    # Start figure
    fig, ax = plt.subplots(figsize=(16, 8))
    params = {'xtick.labelsize':24,
              'ytick.labelsize':24}
    pylab.rcParams.update(params)

    for event in events:
        plt.axvline(event, color='black', linewidth=1)
#    for event in small_events:
#        plt.axvline(event, color='blue', linewidth=1)

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

        # Wavelet vectors initialization
        times = []
        disps = []
        Ws = []
        Vs = []
        Ds = []
        Ss = []

        # Read output files from wavelet transform
        for (station, lon_sta, lat_sta) in zip(sub_stations['name'], sub_stations['longitude'], sub_stations['latitude']):
            filename = 'MODWT_GPS_longer/' + dataset + '_' + station + '_' + direction + '.pkl'
            (time, disp, W, V, D, S) = pickle.load(open(filename, 'rb'))
            if ((np.min(time) <= 2021.5) and (np.max(time) >= 2006.0)):
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
        times_stacked = []
        disps_stacked = []

        # Loop on time blocks
        for t in range(0, len(tblocks) - 1):

            # Find time period
            for time in times:
                indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
                if (len(indices) > 0):
                    time_subset = time[indices]
                    break
            times_stacked.append(time_subset)

            # Loop on stations
            nsta = 0
            Dj_stacked = np.zeros(len(time_subset))
            for (time, D, lat_sta) in zip(times, Ds, sub_stations['latitude']):
                indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
                if (len(indices) > 0):
                    nsta = nsta + 1
                    tj = time[indices]
#                    Dj = D[J][indices]
                    Dj = np.zeros(len(indices))
                    for j in J:
                        Dj = Dj + D[j - 1][indices]
                    Dj_interp = np.interp(time_subset, tj, Dj)
                    Dj_stacked =  Dj_stacked + Dj_interp
            Dj_stacked = Dj_stacked / nsta
            disps_stacked.append(Dj_stacked)

        # Concatenate times and disps
        times_stacked = np.concatenate(times_stacked)
        disps_stacked = np.concatenate(disps_stacked)

        # Filter displacement
        disps_stacked = disps_stacked[(times_stacked  >= 2006.0) & (times_stacked <= 2021.5)]
        times_stacked = times_stacked[(times_stacked  >= 2006.0) & (times_stacked  <= 2021.5)]

        # Figure
        if len(times_stacked) > 0:
            
            slowslip = np.where(np.abs(disps_stacked) >= threshold)[0]
            times_slowslip = times_stacked[slowslip]
            disps_slowslip = disps_stacked[slowslip]
            difference = np.diff(times_slowslip)
            jumps = np.where(difference > 1.5 / 365.25)[0]
            
            if len(jumps > 0):
                begin_jumps = np.insert(jumps + 1, 0, 0)
                end_jumps = np.append(jumps, len(times_slowslip) - 1)
                begin_times = times_slowslip[begin_jumps]
                end_times = times_slowslip[end_jumps]

                for i in range(0, len(jumps) + 1):
                    x0 = begin_times[i]
                    dx = end_times[i] - begin_times[i]
                    if disps_slowslip[begin_jumps[i]] > 0:
                        ax.add_patch(Rectangle((x0, lat + 0.01), dx, 0.03, facecolor='red'))
                    else:
                        ax.add_patch(Rectangle((x0, lat - 0.04), dx, 0.03, facecolor='blue'))

            plt.plot(times_stacked, lat + 0.1 * disps_stacked, color='black')

    plt.xlim([2006.0, 2021.5])
    plt.xlabel('Time (years)', fontsize=24)
    plt.xticks(fontsize=24)
    plt.ylim([min(lats) - 0.15, max(lats) + 0.15])
    plt.ylabel('Latitude', fontsize=24)
    plt.yticks(fontsize=24)
#    plt.title('Detail at level {:d} of MODWT of GPS data'. \
#        format(J + 1), fontsize=24)
    plt.title('Details at levels {} of MODWT of GPS data'. \
        format(J), fontsize=24)
#    plt.savefig('GPS_longer_detail_' + str(J + 1) + '.eps', format='eps')
    plt.savefig('GPS_longer_detail.eps', format='eps')
    plt.close(1)

def plot_tremor(lats, J, threshold, events, small_events):
    """
    """

    # Start figure
    fig, ax = plt.subplots(figsize=(16, 8))
    params = {'xtick.labelsize':24,
              'ytick.labelsize':24}
    pylab.rcParams.update(params)

    for event in events:
        plt.axvline(event, color='black', linewidth=1)
#    for event in small_events:
#        plt.axvline(event, color='blue', linewidth=1)

    # Loop on latitude and longitude
    for index, lat in enumerate(lats):

        # Read MODWT of tremor
        filename = 'MODWT_tremor_longer/tremor_' + str(index) + '.pkl'
        MODWT_tremor = pickle.load(open(filename, 'rb'))
        times_stacked = MODWT_tremor[0]
#        D_tremor = MODWT_tremor[4][J]
        D_tremor = np.zeros(len(times_stacked))
        for j in J:
            D_tremor = D_tremor + MODWT_tremor[4][j - 1]
        
        # Figure
        if len(times_stacked) > 0:
            
            slowslip = np.where(np.abs(D_tremor) >= threshold)[0]
            times_slowslip = times_stacked[slowslip]
            D_slowslip = D_tremor[slowslip]
            difference = np.diff(times_slowslip)
            jumps = np.where(difference > 1.5 / 365.25)[0]

            if len(jumps) > 0:
                begin_jumps = np.insert(jumps + 1, 0, 0)
                end_jumps = np.append(jumps, len(times_slowslip) - 1)
                begin_times = times_slowslip[begin_jumps]
                end_times = times_slowslip[end_jumps]

                for i in range(0, len(jumps) + 1):
                    x0 = begin_times[i]
                    dx = end_times[i] - begin_times[i]
                    if D_slowslip[begin_jumps[i]] < 0:
                        ax.add_patch(Rectangle((x0, lat + 0.01), dx, 0.03, facecolor='red'))
                    else:
                        ax.add_patch(Rectangle((x0, lat - 0.04), dx, 0.03, facecolor='blue'))

            plt.plot(times_stacked, lat - 5.0 * D_tremor, color='black')

    plt.xlim([2006.0, 2021.5])
    plt.xlabel('Time (years)', fontsize=24)
    plt.xticks(fontsize=24)
    plt.ylim([min(lats) - 0.15, max(lats) + 0.15])
    plt.ylabel('Latitude', fontsize=24)
    plt.yticks(fontsize=24)
#    plt.title('Detail at level {:d} of MODWT of tremor data'. \
#        format(J + 1), fontsize=24)
    plt.title('Details at levels {} of MODWT of tremor data'. \
        format(J), fontsize=24)
#    plt.savefig('tremor_longer_detail_' + str(J + 1) + '.eps', format='eps')
    plt.savefig('tremor_longer_detail.eps', format='eps')
    plt.close(1)

def find_false_detections(station_file, lats, lons, dataset, direction, \
    radius_GPS, J, thresh_GPS, thresh_tremor, events, plot=True):
    """
    """
    # Read station file
    stations = pd.read_csv(station_file, sep=r'\s{1,}', header=None, engine='python')
    stations.columns = ['name', 'longitude', 'latitude']

    a = 6378.136
    e = 0.006694470

    # Start figure
    if plot == True:
        fig, ax = plt.subplots(figsize=(16, 8))
        params = {'xtick.labelsize':24,
                  'ytick.labelsize':24}
        pylab.rcParams.update(params)

        for event in events:
            plt.axvline(event, color='black', linewidth=1)

    TP0 = np.zeros(len(lats))
    TN0 = np.zeros(len(lats))
    FP0 = np.zeros(len(lats))
    FN0 = np.zeros(len(lats))
    
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

        # Wavelet vectors initialization
        times = []
        disps = []
        Ws = []
        Vs = []
        Ds = []
        Ss = []

        # Read output files from wavelet transform
        for (station, lon_sta, lat_sta) in zip(sub_stations['name'], sub_stations['longitude'], sub_stations['latitude']):
            filename = 'MODWT_GPS_longer/' + dataset + '_' + station + '_' + direction + '.pkl'
            (time, disp, W, V, D, S) = pickle.load(open(filename, 'rb'))
            if ((np.min(time) <= 2021.5) and (np.max(time) >= 2006.0)):
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
        times_stacked = []
        disps_stacked = []

        # Loop on time blocks
        for t in range(0, len(tblocks) - 1):

            # Find time period
            for time in times:
                indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
                if (len(indices) > 0):
                    time_subset = time[indices]
                    break
            times_stacked.append(time_subset)

            # Loop on stations
            nsta = 0
            Dj_stacked = np.zeros(len(time_subset))
            for (time, D, lat_sta) in zip(times, Ds, sub_stations['latitude']):
                indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
                if (len(indices) > 0):
                    nsta = nsta + 1
                    tj = time[indices]
#                    Dj = D[J][indices]
                    Dj = np.zeros(len(indices))
                    for j in J:
                        Dj = Dj + D[j - 1][indices]
                    Dj_interp = np.interp(time_subset, tj, Dj)
                    Dj_stacked =  Dj_stacked + Dj_interp
            Dj_stacked = Dj_stacked / nsta
            disps_stacked.append(Dj_stacked)

        # Concatenate times and disps
        times_stacked = np.concatenate(times_stacked)
        disps_stacked = np.concatenate(disps_stacked)

        # Filter displacement
        disps_stacked = disps_stacked[(times_stacked  >= 2006.0) & (times_stacked <= 2021.5)]
        times_stacked = times_stacked[(times_stacked  >= 2006.0) & (times_stacked  <= 2021.5)]

        # Read MODWT of tremor
        filename = 'MODWT_tremor_longer/tremor_' + str(index) + '.pkl'
        MODWT_tremor = pickle.load(open(filename, 'rb'))
        time = MODWT_tremor[0]
#        D_tremor = - MODWT_tremor[4][J]
        D_tremor = np.zeros(len(times_stacked))
        for j in J:
            D_tremor = D_tremor - MODWT_tremor[4][j - 1]

        # Correlation
        if len(times_stacked) > 0:

            pos_GPS = np.where(disps_stacked >= thresh_GPS)[0]
            neg_GPS = np.where(disps_stacked <= - thresh_GPS)[0]
            pos_tremor = np.where(D_tremor >= thresh_tremor)[0]
            neg_tremor = np.where(D_tremor <= - thresh_tremor)[0]
            input1 = np.zeros(len(times_stacked))
            input1[pos_GPS] = 1
            input1[neg_GPS] = -1
            input2 = np.zeros(len(times_stacked))
            input2[pos_tremor] = 1
            input2[neg_tremor] = -1

            # True positives
            TP = np.where(input1 * input2 == 1)[0]
            TP0[index] = len(TP)

            if (plot == True) & (len(TP) > 0):
                times_TP = times_stacked[TP]
                difference = np.diff(times_TP)
                jumps = np.where(difference > 1.5 / 365.25)[0]
                begin_jumps = np.insert(jumps + 1, 0, 0)
                end_jumps = np.append(jumps, len(times_TP) - 1)
                begin_times = times_TP[begin_jumps]
                end_times = times_TP[end_jumps]

                for i in range(0, len(jumps) + 1):
                    x0 = begin_times[i]
                    dx = end_times[i] - begin_times[i]
                    ax.add_patch(Rectangle((x0, lat - 0.015), dx, 0.03, facecolor='green'))

            # True negatives
            TN = np.where((input1 == 0) & (input2 == 0))[0]
            TN0[index] = len(TN)

            if (plot == True) & (len(TN) > 0):
                times_TN = times_stacked[TN]
                difference = np.diff(times_TN)
                jumps = np.where(difference > 1.5 / 365.25)[0]
                begin_jumps = np.insert(jumps + 1, 0, 0)
                end_jumps = np.append(jumps, len(times_TN) - 1)
                begin_times = times_TN[begin_jumps]
                end_times = times_TN[end_jumps]

                for i in range(0, len(jumps) + 1):
                    x0 = begin_times[i]
                    dx = end_times[i] - begin_times[i]
                    ax.add_patch(Rectangle((x0, lat - 0.015), dx, 0.03, facecolor='grey'))

            # False positives
            FP = np.where((input1 * input2 == -1) | ((np.abs(input1) == 1) & (input2 == 0)))[0]
            FP0[index] = len(FP)

            if (plot == True) & (len(FP) > 0):
                times_FP = times_stacked[FP]
                difference = np.diff(times_FP)
                jumps = np.where(difference > 1.5 / 365.25)[0]
                begin_jumps = np.insert(jumps + 1, 0, 0)
                end_jumps = np.append(jumps, len(times_FP) - 1)
                begin_times = times_FP[begin_jumps]
                end_times = times_FP[end_jumps]

                for i in range(0, len(jumps) + 1):
                    x0 = begin_times[i]
                    dx = end_times[i] - begin_times[i]
                    ax.add_patch(Rectangle((x0, lat - 0.015), dx, 0.03, facecolor='red'))

            # False negatives
            FN = np.where((input1 == 0) & (np.abs(input2) == 1))[0]
            FN0[index] = len(FN)

            if (plot == True) & (len(FN) > 0):
                times_FN = times_stacked[FN]
                difference = np.diff(times_FN)
                jumps = np.where(difference > 1.5 / 365.25)[0]
                begin_jumps = np.insert(jumps + 1, 0, 0)
                end_jumps = np.append(jumps, len(times_FN) - 1)
                begin_times = times_FN[begin_jumps]
                end_times = times_FN[end_jumps]

                for i in range(0, len(jumps) + 1):
                    x0 = begin_times[i]
                    dx = end_times[i] - begin_times[i]
                    ax.add_patch(Rectangle((x0, lat - 0.015), dx, 0.03, facecolor='orange'))

    if plot == True:
        plt.xlim([2006.0, 2021.5])
        plt.xlabel('Time (years)', fontsize=24)
        plt.xticks(fontsize=24)
        plt.ylim([min(lats) - 0.15, max(lats) + 0.15])
        plt.ylabel('Latitude', fontsize=24)
        plt.yticks(fontsize=24)
#        plt.title('Signal detected for detail at level {:d}'. \
#            format(J + 1), fontsize=24)
        plt.title('Signal detected for details at levels {}'. \
            format(J), fontsize=24)
#        plt.savefig('signal_detected_' + str(J + 1) + '.eps', format='eps')
        plt.savefig('signal_detected.eps', format='eps')
        plt.close(1)
    
    return(np.sum(TP0), np.sum(TN0), np.sum(FP0), np.sum(FN0))

def plot_ROC_curve(station_file, lats, lons, dataset, direction, \
    radius_GPS, J, thresh_GPS, thresh_tremor, events, chosen_GPS, chosen_tremor):
    """
    """
    TP = np.zeros((len(thresh_GPS), len(thresh_tremor)))
    TN = np.zeros((len(thresh_GPS), len(thresh_tremor)))
    FP = np.zeros((len(thresh_GPS), len(thresh_tremor)))
    FN = np.zeros((len(thresh_GPS), len(thresh_tremor)))
    sensitivity = np.zeros((len(thresh_GPS), len(thresh_tremor)))
    specificity = np.zeros((len(thresh_GPS), len(thresh_tremor)))
    for i in range(0, len(thresh_GPS)):
        for j in range(0, len(thresh_tremor)):
#            (TP[i, j], TN[i, j], FP[i, j], FN[i, j]) = find_false_detections( \
#                station_file, lats, lons, dataset, direction, radius_GPS, \
#                J - 1, thresh_GPS[i], thresh_tremor[j], events, False)
            (TP[i, j], TN[i, j], FP[i, j], FN[i, j]) = find_false_detections( \
                station_file, lats, lons, dataset, direction, radius_GPS, \
                J, thresh_GPS[i], thresh_tremor[j], events, False)
            if TP[i, j] + FN[i, j] == 0:
                sensitivity[i, j] = 0
            else:
                sensitivity[i, j] = TP[i, j] / (TP[i, j] + FN[i, j])
            if TN[i, j] + FP[i, j] == 0:
                specificity[i, j] = 0
            else:
                specificity[i, j] = TN[i, j] / (TN[i, j] + FP[i, j])
    i0 = np.where(np.abs(thresh_GPS - chosen_GPS) < 0.001)[0]
    j0 = np.where(np.abs(thresh_tremor - chosen_tremor) < 0.00001)[0]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    params = {'xtick.labelsize':24,
              'ytick.labelsize':24}
    pylab.rcParams.update(params)
    plt.plot([-0.1, 1.1], [-0.1, 1.1], color='grey')
    plt.plot(1 - specificity, sensitivity, 'ko')
    plt.plot(1 - specificity[i0, j0], sensitivity[i0, j0], 'rx', markersize=20)
    plt.xlabel('False positive rate', fontsize=24)
    plt.xticks(fontsize=24)
    plt.xlim([0, 1])
    plt.ylabel('True positive rate', fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylim([0, 1])
#    plt.title('ROC curve for detail at level {:d}'. format(J), fontsize=24)
    plt.title('ROC curve for details at levels {}'. format(J), fontsize=24)
#    plt.savefig('ROC_' + str(J) + '.eps', format='eps')
    plt.savefig('ROC.eps', format='eps')
    plt.close(1)

    return(sensitivity, specificity)

if __name__ == '__main__':

    station_file = '../data/PANGA/stations.txt'
    lats = [47.20000, 47.30000, 47.40000, 47.50000, 47.60000, 47.70000, \
        47.80000, 47.90000, 48.00000, 48.10000, 48.20000, 48.30000, 48.40000, \
        48.50000, 48.60000, 48.70000]
    lons = [-122.74294, -122.73912, -122.75036, -122.77612, -122.81591, \
        -122.86920, -122.93549, -123.01425, -123.10498, -123.20716, \
        -123.32028, -123.44381, -123.57726, -123.72011, -123.87183, \
        -124.03193]
    dataset = 'cleaned'
    direction = 'lon'
    radius_GPS = 50
    events = [2007.0630, 2008.3566, 2009.3452, 2010.6342, 2011.6041, 2012.7172, \
        2013.7137, 2014.6479, 2014.8904, 2016.1082, 2017.2342, 2018.4932, \
        2019.2329, 2019.8767, 2020.7923, 2020.8552, 2021.0904]
    small_events = [2007.06, 2007.08, 2008.38, 2009.16, 2009.36, 2010.63, \
        2011.66, 2012.69, 2013.74, 2014.69, 2014.93, 2016.03, 2017.13, 2017.22]

#    radius_tremor = 50
#    wavelet = 'LA8'
#    j = 4
#    slowness = np.arange(-0.1, 0.105, 0.005)
#    t0 = 2021.118
#    nt = 16

#    compare_tremor_GPS(station_file, lats, lons, dataset, direction, \
#        radius_GPS, radius_tremor, slowness, wavelet, j - 1, J)

#    correlate_tremor_GPS(station_file, lats, lons, dataset, direction, \
#        radius_GPS, J - 1, t0, nt)

#    plot_correlations()

    # For GPS data
    # Level 10: 0.3 - Level 9: 0.4 - Level 8: 0.4 - Level 7: 0.5 - Level 6: 0.3 - Level 5: 0.3 - Level 4: 0.3
    # Level 7-8: 0.7 - Level 6-7-8: 0.8 - Level 5-6-7-8: 0.8 - Level 5-6-7: 0.6 - Level 5-6: 0.4 - Level 6-7: 0.5
    # For tremor data:
    # Level 10: 0.009 - Level 9: 0.009 - Level 8: 0.003 - Level 7: 0.01 - Level 6: 0.009 - Level 5: 0.01 - Level 4: 0.007
    # Level 7-8: 0.01 - Level 6-7-8: 0.01 - Level 5-6-7-8: 0.009 - Level 5-6-7: 0.01 - Level 5-6: 0.01 - Level 6-7: 0.01

    thresh_GPS = np.arange(0.1, 1.6, 0.1)
    thresh_tremor = np.arange(0.001, 0.011, 0.001)
    chosen_GPS = 0.5
    chosen_tremor = 0.01
    J = [6, 7]

    plot_GPS(station_file, lats, lons, dataset, direction, radius_GPS, J, chosen_GPS, events, small_events)
    plot_tremor(lats, J, chosen_tremor, events, small_events)
    
    (TP, TN, FP, FN) = find_false_detections(station_file, lats, lons, dataset, direction, \
        radius_GPS, J, chosen_GPS, chosen_tremor, events, True)
    
    (sensitivity, specificity) = plot_ROC_curve(station_file, lats, lons, dataset, direction, \
        radius_GPS, J, thresh_GPS, thresh_tremor, events, chosen_GPS, chosen_tremor)
