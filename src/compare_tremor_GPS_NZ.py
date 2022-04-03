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


def plot_GPS(station_file, lats, lons, direction, radius_GPS, J, threshold):
    """
    """
    # Read station file
    stations = pd.read_csv(station_file, sep=r'\s{1,}', header=None, engine='python')
    stations.columns = ['name', 'longitude', 'latitude']

    # Time limits
    tmin = 2000 + 909 / 365.25
    tmax = 2000 + 8127 / 365.25

    a = 6378.136
    e = 0.006694470

    # Start figure
    fig, ax = plt.subplots(figsize=(16, 8))
    params = {'xtick.labelsize':24,
              'ytick.labelsize':24}
    pylab.rcParams.update(params)

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
            filename = 'MODWT_GPS_NZ/' + station + '_' + direction + '.pkl'
            (time, disp, W, V, D, S) = pickle.load(open(filename, 'rb'))
            if ((np.min(time) <= tmax) and (np.max(time) >= tmin)):
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
        disps_stacked = disps_stacked[(times_stacked  >= tmin) & (times_stacked <= tmax)]
        times_stacked = times_stacked[(times_stacked  >= tmin) & (times_stacked <= tmax)]

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

    plt.xlim([tmin, tmax])
    plt.xlabel('Time (years)', fontsize=24)
    plt.xticks(fontsize=24)
    plt.ylim([min(lats) - 0.15, max(lats) + 0.15])
    plt.ylabel('Latitude', fontsize=24)
    plt.yticks(fontsize=24)
#    plt.title('Detail at level {:d} of MODWT of GPS data'. \
#        format(J + 1), fontsize=24)
    plt.title('Details at levels {} of MODWT of GPS data'. \
        format(J), fontsize=24)
#    plt.savefig('GPS_NZ_detail_' + str(J + 1) + '.pdf', format='pdf')
    plt.savefig('GPS_NZ_detail.pdf', format='pdf')
    plt.close(1)

if __name__ == '__main__':

    station_file = '../data/GeoNet/stations.txt'
    lats = [-39.5000, -39.4000, -39.3000, -39.2000, -39.1000, -39.0000, \
        -38.9000, -38.8000, -38.7000, -38.6000, -38.5000, -38.4000, \
        -38.3000, -38.2000, -38.1000, -38.0000]
    lons = [176.8000, 176.9000, 177.0000, 177.1000, 177.2000, 177.3000, \
        177.4000, 177.5000, 177.6000, 177.7000, 177.8000, 177.9000, \
        178.0000, 178.1000, 178.2000, 178.3000]
    direction = 'e'
    radius_GPS = 50

    # For GPS data
    # Level 10: 0.3 - Level 9: 0.4 - Level 8: 0.4 - Level 7: 0.5 - Level 6: 0.3 - Level 5: 0.3 - Level 4: 0.3
    # Level 7-8: 0.7 - Level 6-7-8: 0.8 - Level 5-6-7-8: 0.8 - Level 5-6-7: 0.6 - Level 5-6: 0.4 - Level 6-7: 0.5

    chosen_GPS = 0.5
    chosen_tremor = 0.01
    J = [6, 7]

    plot_GPS(station_file, lats, lons, direction, radius_GPS, J, chosen_GPS)
