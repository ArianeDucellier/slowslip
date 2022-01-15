"""
Script to make a series of figures for a movie
"""

import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from datetime import datetime
from math import cos, floor, pi, sin, sqrt
from scipy.io import loadmat

import date

def plot_GPS_map_tremor(station_file, lats, lons, dataset, direction, radius_GPS, J, threshold, \
    lonmin, lonmax, latmin, latmax, tmin_GPS, tmax_GPS, tmin_tremor, tmax_tremor, image):
    """
    """
    # Read station file
    stations = pd.read_csv(station_file, sep=r'\s{1,}', header=None, engine='python')
    stations.columns = ['name', 'longitude', 'latitude']

    # Read tremor files (A. Wech)
#    data_2009 = pd.read_csv('../data/tremor/tremor_events-2009-08-06T00 00 00-2009-12-31T23 59 59.csv')
#    data_2009['time '] = pd.to_datetime(data_2009['time '], format='%Y-%m-%d %H:%M:%S')
#    data_2010 = pd.read_csv('../data/tremor/tremor_events-2010-01-01T00 00 00-2010-12-31T23 59 59.csv')
#    data_2010['time '] = pd.to_datetime(data_2010['time '], format='%Y-%m-%d %H:%M:%S')
#    data_2011 = pd.read_csv('../data/tremor/tremor_events-2011-01-01T00 00 00-2011-12-31T23 59 59.csv')
#    data_2011['time '] = pd.to_datetime(data_2011['time '], format='%Y-%m-%d %H:%M:%S')
#    data_2012 = pd.read_csv('../data/tremor/tremor_events-2012-01-01T00 00 00-2012-12-31T23 59 59.csv')
#    data_2012['time '] = pd.to_datetime(data_2012['time '], format='%Y-%m-%d %H:%M:%S')
#    data_2013 = pd.read_csv('../data/tremor/tremor_events-2013-01-01T00 00 00-2013-12-31T23 59 59.csv')
#    data_2013['time '] = pd.to_datetime(data_2013['time '], format='%Y-%m-%d %H:%M:%S')
#    data_2014 = pd.read_csv('../data/tremor/tremor_events-2014-01-01T00 00 00-2014-12-31T23 59 59.csv')
#    data_2014['time '] = pd.to_datetime(data_2014['time '], format='%Y-%m-%d %H:%M:%S')
#    data_2015 = pd.read_csv('../data/tremor/tremor_events-2015-01-01T00 00 00-2015-12-31T23 59 59.csv')
#    data_2015['time '] = pd.to_datetime(data_2015['time '], format='%Y-%m-%d %H:%M:%S')
#    data_2016 = pd.read_csv('../data/tremor/tremor_events-2016-01-01T00 00 00-2016-12-31T23 59 59.csv')
#    data_2016['time '] = pd.to_datetime(data_2016['time '], format='%Y-%m-%d %H:%M:%S')
#    data_2017 = pd.read_csv('../data/tremor/tremor_events-2017-01-01T00 00 00-2017-12-31T23 59 59.csv')
#    data_2017['time '] = pd.to_datetime(data_2017['time '], format='%Y-%m-%d %H:%M:%S')
#    data_2018 = pd.read_csv('../data/tremor/tremor_events-2018-01-01T00 00 00-2018-12-31T23 59 59.csv')
#    data_2018['time '] = pd.to_datetime(data_2018['time '], format='%Y-%m-%d %H:%M:%S')
#    data_2019 = pd.read_csv('../data/tremor/tremor_events-2019-01-01T00 00 00-2019-12-31T23 59 59.csv')
#    data_2019['time '] = pd.to_datetime(data_2019['time '], format='%Y-%m-%d %H:%M:%S')
#    data_2020 = pd.read_csv('../data/tremor/tremor_events-2020-01-01T00 00 00-2020-12-31T23 59 59.csv')
#    data_2020['time '] = pd.to_datetime(data_2020['time '], format='%Y-%m-%d %H:%M:%S')
#    data_2021 = pd.read_csv('../data/tremor/tremor_events-2021-01-01T00 00 00-2021-04-29T23 59 59.csv')
#    data_2021['time '] = pd.to_datetime(data_2020['time '], format='%Y-%m-%d %H:%M:%S')
#    data = pd.concat([data_2009, data_2010, data_2011, data_2012, data_2013, \
#        data_2014, data_2015, data_2016, data_2017, data_2018, data_2019, \
#        data_2020, data_2021])
#    data.reset_index(drop=True, inplace=True)

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

    # Convert begin and end times for tremor
    (year1, month1, day1, hour1, minute1, second1) = date.day2ymdhms(tmin_tremor)
    day_begin = date.ymdhms2matlab(year1, month1, day1, hour1, minute1, second1)
    (year2, month2, day2, hour2, minute2, second2) = date.day2ymdhms(tmax_tremor)
    day_end = date.ymdhms2matlab(year2, month2, day2, hour2, minute2, second2)

    a = 6378.136
    e = 0.006694470

    # Start figure
    fig = plt.figure(1, figsize=(16, 8))
    params = {'xtick.labelsize':20,
              'ytick.labelsize':20}
    pylab.rcParams.update(params)
    
    # First part: MODWT
    ax1 = plt.subplot2grid((1, 2), (0, 0))

    # Stations used for figure
    lon_stas = []
    lat_stas = []

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
                        ax1.add_patch(Rectangle((x0, lat + 0.01), dx, 0.03, facecolor='red'))
                    else:
                        ax1.add_patch(Rectangle((x0, lat - 0.04), dx, 0.03, facecolor='blue'))

            plt.plot(times_stacked, lat + 0.1 * disps_stacked, color='black')

    ax1.add_patch(Rectangle((tmin_tremor, min(lats) - 0.15), \
        tmax_tremor - tmin_tremor, max(lats) - min(lats) + 0.3, \
        fill=False, edgecolor='grey', linewidth=3))
    plt.xlim([tmin_GPS, tmax_GPS])
    plt.xlabel('Time (years)', fontsize=20)
    plt.xticks(fontsize=24)
    plt.ylim([min(lats) - 0.15, max(lats) + 0.15])
    plt.ylabel('Latitude', fontsize=20)
    plt.yticks(fontsize=24)
#    plt.title('Detail at level {:d} of MODWT of GPS data'. \
#        format(J + 1), fontsize=20)
    plt.title('Details at levels {} of MODWT of GPS data'. \
        format(J), fontsize=20)

    # Part 2: map of tremor
    ax2 = plt.subplot2grid((1, 2), (0, 1), projection=ccrs.Mercator())
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

    # Keep only tremor on map
    mask = ((data['lat'] >= latmin) & (data['lat'] <= latmax) \
          & (data['lon'] >= lonmin) & (data['lon'] <= lonmax))
    tremor = data.loc[mask].copy()
    tremor.reset_index(drop=True, inplace=True)

   # Keep only tremor in time interval (A. Wech)
#    mask = ((tremor['time '] >= datetime(year1, month1, day1, hour1, minute1, second1)) \
#          & (tremor['time '] <= datetime(year2, month2, day2, hour2, minute2, second2)))
#    tremor_sub = tremor.loc[mask].copy()
#    tremor_sub.reset_index(drop=True, inplace=True)

    # Keep only tremor in time interval (K. Creager)
    mask = ((tremor['time'] >= day_begin) \
          & (tremor['time'] <= day_end))
    tremor_sub = tremor.loc[mask].copy()
    tremor_sub.reset_index(drop=True, inplace=True)

    # Convert tremor time (A. Wech)
#    nt = len(tremor_sub)
#    time_tremor = np.zeros(nt)
#    for i in range(0, nt):
#        year = tremor_sub['time '].loc[i].year
#        month = tremor_sub['time '].loc[i].month
#        day = tremor_sub['time '].loc[i].day
#        hour = tremor_sub['time '].loc[i].hour
#        minute = tremor_sub['time '].loc[i].minute
#        second = tremor_sub['time '].loc[i].second
#        time_tremor[i] = date.ymdhms2day(year, month, day, hour, minute, second)    

    # Convert tremor time (K. Creager)
    nt = len(tremor_sub)
    time_tremor = np.zeros(nt)
    for i in range(0, nt):
        (year, month, day, hour, minute, second) = date.matlab2ymdhms(tremor_sub['time'].loc[i])
        time_tremor[i] = date.ymdhms2day(year, month, day, hour, minute, second)    

    # Plot tremor on map (A. Wech)
    ax2.scatter(tremor_sub['lon'], tremor_sub['lat'], c='blue', s=2, transform=ccrs.PlateCarree())

    # Plot GPS stations on map
    ax2.scatter(np.array(lon_stas), np.array(lat_stas), c='r', marker='^', s=20, transform=ccrs.PlateCarree())

    # Plot centers at which we look for GPS data on map
    ax2.scatter(np.array(lons), np.array(lats), c='k', marker='x', s=20, transform=ccrs.PlateCarree())

    ax2.set_title('{:04d} tremor in {:02d} days'.format(np.shape(tremor_sub)[0], \
        int(floor(365.25 * (tmax_tremor - tmin_tremor)))), fontsize=20)

    plt.savefig('movie/GPS_zoom_2.pdf', format='pdf')
#    plt.savefig('movie/GPS_{:03d}.png'.format(image), format='png')
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
    dataset = 'cleaned'
    direction = 'lon'
    radius_GPS = 50

    J = [6, 7, 8]
    chosen_GPS = 0.8

    lonmin = -125.4
    lonmax = -121.4
    latmin = 46.3
    latmax = 49.6

#    for image in range(261, 501):

#        tmin_GPS = 2010.25 + image * 0.02 - 1.0
#        tmax_GPS = 2010.25 + image * 0.02 + 1.0
#        tmin_tremor = 2010.25 + image * 0.02 - 7.5 / 365.25
#        tmax_tremor = 2010.25 + image * 0.02 + 7.5 / 365.25

#        plot_GPS_map_tremor(station_file, lats, lons, dataset, direction, radius_GPS, J, chosen_GPS, \
#            lonmin, lonmax, latmin, latmax, tmin_GPS, tmax_GPS, tmin_tremor, tmax_tremor, image)

    tmin_GPS = 2010.22 - 1.0
    tmax_GPS = 2010.22 + 1.0
    tmin_tremor = 2010.22 - 7.5 / 365.25
    tmax_tremor = 2010.22 + 7.5 / 365.25
    image = 0

    plot_GPS_map_tremor(station_file, lats, lons, dataset, direction, radius_GPS, J, chosen_GPS, \
        lonmin, lonmax, latmin, latmax, tmin_GPS, tmax_GPS, tmin_tremor, tmax_tremor, image)
