"""
Script to find automatically the ETS events
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

#  analyze the sum of wavelets for details 6-8
details = '6-8'

# start and end times of wavelets and tremor catalogs
startTime = 2006.8
endTime = 2021.3

# Latitudes and Longitudes for 16 reference points along strike
lats = [47.20000, 47.30000, 47.40000, 47.50000, 47.60000, 47.70000, 47.80000, 47.90000, 48.00000, \
    48.10000, 48.20000, 48.30000, 48.40000, 48.50000, 48.60000, 48.70000]
lons = [-122.74294, -122.73912, -122.75036, -122.77612, -122.81591, -122.86920, -122.93549, \
    -123.01425, -123.10498, -123.20716, -123.32028, -123.44381, -123.57726, -123.72011, -123.87183, -124.03193]
Nlat = len(lats)

# Read in all the GPS wavelets
# Redefine time vectors with errors up to +/- 4 days
# time for 1-4 are the same but have 10 extra samples so remove 10 evenly spaced time samples
# time for 16 has one extra sample so remove the middle one
# time for 5-15 are the same
# time for each latitude point is replaced by time vector for latitude 12
# start and end times are the same for 1-16

Dat_T = np.zeros((4, 16, 5578))
Dat_Y = np.zeros((4, 16, 5578))

for detail in [5, 6, 7, 8]:
    for lat_id in np.arange(0, 16):
        fileName = 'files/GPS_detail_' + str(detail) + '_loc_' + str(lat_id) + '.txt'
        tmp = np.loadtxt(fileName)
        klat = lat_id + 1
        tmp1 = tmp[:, 0]
        tmp2 = tmp[:, 1]
        if klat <= 4:
            krm = np.int_(np.floor(np.arange(0, 10) * len(tmp) / 11))
            tmp1 = np.delete(tmp1, krm)
            tmp2 = np.delete(tmp2, krm)
        if klat == 16:
            krm = np.int_(np.floor(0.5 * len(tmp)))
            tmp1 = np.delete(tmp1, krm)
            tmp2 = np.delete(tmp2, krm)
        Dat_T[detail - 5, klat - 1, :] = tmp1
        Dat_Y[detail - 5, klat - 1, :] = tmp2

T = Dat_T[1, 11, :]

# Cut the time to startTime and endTime
k1 = np.min(np.where(T >= startTime)[0])
k2 = np.max(np.where(T <= endTime)[0])
kk = np.arange(k1, k2 + 1)

T = T[kk]

Dat_T_new = np.zeros((4, 16, len(kk)))
Dat_Y_new = np.zeros((4, 16, len(kk)))

for detail in [5, 6, 7, 8]:
    for klat in np.arange(1, 17):
        Dat_T_new[detail - 5, klat - 1, :] = Dat_T[detail - 5, klat - 1, kk]
        Dat_Y_new[detail - 5, klat - 1, :] = Dat_Y[detail - 5, klat - 1, kk]

Dat_T = Dat_T_new
Dat_Y = Dat_Y_new

det_Y = np.zeros((Nlat, np.shape(Dat_Y)[2]))
det_T = np.zeros((Nlat, np.shape(Dat_T)[2]))

for lat_id in np.arange(0, Nlat):
    if details == '5-8':
        det_Y[lat_id, :] = Dat_Y[0, lat_id, :] + Dat_Y[1, lat_id, :] + Dat_Y[2, lat_id, :] + Dat_Y[3, lat_id, :]
    elif details == '6-8':
        det_Y[lat_id, :] = Dat_Y[1, lat_id, :] + Dat_Y[2, lat_id, :] + Dat_Y[3, lat_id, :]
    det_T[lat_id, :] = Dat_T[1, lat_id, :]

# for each reference latitude/longitude Y(T) is the summed wavelet detail
# find all peaks and their following troughs and store them in Results(16) with
# fields:
# T,     Y     : time vector and summed wavelet vector
# T_top, Y_top : time and wavelet value for each peak
# T_bot, Y_bot : time and wavelet value for each following trough
# T_mid, Y_mid : time and wavelet value for each midpoint
# Y_siz        : peak-trough amplitude of the wavelet function

# Also create R(16) with one field called R contianing a 42x4 matrix containing
# Y_siz, T_mid, T_top, T_bot for the N largest Y_siz values.
for lat_id in np.arange(0, Nlat):
    Y = det_Y[lat_id, :]
    T = det_T[lat_id, :]
    dY = np.diff(Y)
    k_top = np.where((dY[0 : -1] > 0) & (dY[1 :] < 0))[0] + 1 # find all peaks
    k_bot = np.where((dY[0 : -1] < 0) & (dY[1 :] > 0))[0] + 1 # find all troughs