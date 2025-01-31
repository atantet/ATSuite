"""Alexis Tantet's Regridding module"""

#  Import numpy
import numpy as np

def downsample_rect2rect(src_lon, src_lat, src_data, dst_lon, dst_lat):
    # dst_x, dst_y must be monotically increasing
    dst_nlon = dst_lon.shape[0]
    dst_nlat = dst_lat.shape[0]
    dst_dlon = dst_lon[1] - dst_lon[0]
    dst_dlat = dst_lat[1] - dst_lat[0]
    dst_data = np.empty((dst_nlat, dst_nlon))

#    src_dlon = src_lon[1] - src_lon[0]
#    src_dlat = src_lat[1] - src_lat[0]

    # Number of source grid points to average
#    nid_lon = int(dst_dlon / src_dlon / 2)
#    nid_lat = int(dst_dlat / src_dlat / 2)

#    startid_lon = 

    for ii in range(dst_nlat):
        latid = np.logical_and(src_lat >= (dst_lat[ii] - dst_dlat / 2.),
                               src_lat <= (dst_lat[ii] + dst_dlat / 2.))
        for jj in range(dst_nlon):
            # # No area average yet
            # dst_data[ii, jj] = np.mean(src_data[jj-nid_lat:jj+nid_lat+1,
            #                                     ii-nid_lon:ii+nid_lon+1]) 

            lonid = np.logical_and(src_lon >= (dst_lon[jj] - dst_dlon / 2.),
                                   src_lon <= (dst_lon[jj] + dst_dlon / 2.))
            dst_data[ii, jj] = np.mean(src_data[np.ix_(latid, lonid)])
    return dst_data
