"""
Script to plot a vespagram-like figure of slow slip
"""

import datetime
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from math import cos, pi, sin, sqrt
from scipy.io import loadmat

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
        # MODWT
        (W, V) = pyramid(disp, wavelet, J)
        (D, S) = get_DS(disp, W, wavelet, J)
        # Save wavelets into file
        filename = 'tmp/' + dataset + '_' + station + '_' + direction + '.pkl'
        pickle.dump([time, disp, W, V, D, S], open(filename, 'wb'))
        
def vespagram(station_file, tremor_file, lats, lons, names, radius_GPS, \
    radius_tremor, direction, dataset, wavelet, J, slowness, xmin, xmax):
    """
    """
    # Read station file
    stations = pd.read_csv(station_file, sep=r'\s{1,}', header=None, engine='python')
    stations.columns = ['name', 'longitude', 'latitude']

    # Read tremor file
    data = loadmat(tremor_file)
    mbbp = data['mbbp_cat_d']

    # Loop on latitude and longitude
    a = 6378.136
    e = 0.006694470
    for (lat, lon, name) in zip(lats, lons, names):

        # Keep only stations in a given radius
        dx = (pi / 180.0) * a * cos(lat * pi / 180.0) / sqrt(1.0 - e * e * \
            sin(lat * pi / 180.0) * sin(lat * pi / 180.0))
        dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * \
            sin(lat * pi / 180.0) * sin(lat * pi / 180.0)) ** 1.5)
        x = dx * (stations['longitude'] - lon)
        y = dy * (stations['latitude'] - lat)
        stations['distance'] = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0))
        mask = stations['distance'] <= radius_GPS
        stations = stations.loc[mask]

        # Keep only tremor in a given radius
        lat_tremor = mbbp[:, 2]
        lon_tremor = mbbp[:, 3]
        x = dx * (lon_tremor - lon)
        y = dy * (lat_tremor - lat)
        distance = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0))
        find = np.where(distance <= radius_tremor)
        tremor = mbbp[find, :][0, :, :]

        # Keep only tremor within time limits
        find = np.where((tremor[:, 0] >= 734139) & (tremor[:, 0] < 734320))
        tremor_sub = tremor[find, :][0, :, :]

        # Number of tremors per day
        nt = np.shape(tremor_sub)[0]
        time_tremor = np.zeros(nt)
        for i in range(0, nt):
            myday = date.matlab2ymdhms(tremor_sub[i, 0])
            t1 = datetime.date(myday[0], myday[1], myday[2])
            t2 = datetime.date(myday[0], 1, 1)
            time_tremor[i] = myday[0] + (t1 - t2).days / 365

        ntremor = np.zeros(734320 - 734139)
        for i in range(0, len(ntremor)):
            for j in range (0, len(time_tremor)):
                if ((time_tremor[j] >= xmin + (i - 0.5) / 365.0) and \
                    (time_tremor[j] <= xmin + (i + 0.5) / 365.0)):
                    ntremor[i] = ntremor[i] + 1

        # Wavelet vectors initialization
        times = []
        disps = []
        Ws = []
        Vs = []
        Ds = []
        Ss = []

        # Read output files from wavelet transform
        for station in stations['name']:
            filename = 'tmp/' + dataset + '_' + station + '_' + direction + '.pkl'
            (time, disp, W, V, D, S) = pickle.load(open(filename, 'rb'))
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
                for (time, D, lat_sta) in zip(times, Ds, stations['latitude']):
                    indices = np.where((time >= tbegin[t]) & (time < tend[t]))[0]
                    if (len(indices) > 0):
                        nsta = nsta + 1
                        tj = time[indices]
                        Dj = D[j][indices]
                        for i in range(0, len(slowness)):
                            Dj_interp = np.interp(time_subset + \
                                slowness[i] * (lat_sta - lat), tj, Dj)
                            vespagram[i, :, j] = vespagram[i, :, j] + Dj_interp
                vespagram[:, :, j] = vespagram[:, :, j] / nsta
                if j == 0:
                    time_sta[2 * t] = tbegin[t]
                    time_sta[2 * t + 1] = tend[t]
                    nb_sta[2 * t] = nsta
                    nb_sta[2 * t + 1] = nsta
            vesps.append(vespagram)

        if len(time_vesps) > 0:
            time_subset = np.concatenate(time_vesps)
            vespagram = np.concatenate(vesps, axis=1)

            # Plot
            for j in range(0, J):
                plt.figure(1, figsize=(15, 10))
                plt.subplot(311)
                plt.plot(time_sta, nb_sta, 'k-')
                plt.ylabel('Number of stations')
                plt.xlim([xmin, xmax])
                plt.subplot(312)
                plt.plot(xmin + np.arange(0, len(ntremor)), ntremor, 'k-')
                plt.ylabel('Number of tremor')
                plt.xlim([xmin, xmax])
                plt.subplot(313)
                plt.contourf(time_subset, slowness * 365.25 / dy, \
                    vespagram[:, :, j], cmap=plt.get_cmap('seismic'), \
                    vmin=-2.0, vmax=2.0)
                plt.xlabel('Time (year)')
                plt.ylabel('Slowness (day / km)')
                plt.xlim([xmin, xmax])
                plt.colorbar(orientation='horizontal')
                plt.savefig('vespagram/D' + str(j + 1) + '_' + name + '.eps', \
                    format='eps')
                plt.close(1)

if __name__ == '__main__':

    station_file = '../data/PANGA/stations.txt'
    tremor_file = '../data/tremor/mbbp_cat_d_forHeidi'
    radius_GPS = 100
    radius_tremor = 20
    direction = 'lon'
    dataset = 'cleaned'
    wavelet = 'LA8'
    J = 6
    slowness = np.arange(-0.1, 0.105, 0.005)
    lats = [47.20000, 47.30000, 47.40000, 47.50000, 47.60000, 47.70000, \
        47.80000, 47.90000, 48.00000, 48.10000, 48.20000, 48.30000, 48.40000, \
        48.50000, 48.60000, 48.70000]
    lons = [-122.74294, -122.73912, -122.75036, -122.77612, -122.81591, \
        -122.86920, -122.93549, -123.01425, -123.10498, -123.20716, \
        -123.32028, -123.44381, -123.57726, -123.72011, -123.87183, \
        -124.03193]
    names = []
    for i in range(0, 16):
        name = str(i)
        names.append(name)
    xmin = 2010
    xmax = 2010.5

#    compute_wavelets(station_file, lats, lons, radius_GPS, direction, dataset, \
#        wavelet, J)
    vespagram(station_file, tremor_file, lats, lons, names, radius_GPS, \
        radius_tremor, direction, dataset, wavelet, J, slowness, xmin, xmax)