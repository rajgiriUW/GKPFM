# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 21:19:26 2018

@author: Raj
"""

import numpy as np
from sklearn import metrics
import pycroscopy as px

def add_position_sets(h5_file, group, fast_x=32e-6, fast_y=8e-6):
    """
    Adds Position_Indices and Position_Value datasets to a folder within the h5_file
    
    Uses the values of fast_x and fast_y to determine the values
    """
    
    hdf = px.ioHDF5(h5_file)
    parms_dict = hdf.file['/Measurement_000'].attrs
    
    if 'FastScanSize' in parms_dict:
        fast_x = parms_dict['FastScanSize']
    
    if 'SlowScanSize' in parms_dict:
        fast_x = parms_dict['SlowScanSize']
        
    num_cols = parms_dict['num_cols']
    num_rows = parms_dict['num_rows']
    
    grp = px.MicroDataGroup(group)
    ds_pos_ind, ds_pos_val = px.io.translators.utils.build_ind_val_dsets([num_cols, num_rows], is_spectral=False,
                                              steps=[1.0 * fast_x / num_cols,
                                                     1.0 * fast_y / num_rows],
                                              labels=['X', 'Y'], units=['m', 'm'], verbose=True)
    
    grp.addChildren([ds_pos_ind, ds_pos_val])
    
    hdf.writeData(grp, print_log=False)
    
    return hdf[group]


def CPD_positions(h5_file, CPD_avg, mask, CPD_loc = '/Measurement_000/Channel_000/Raw_Data-CPD'):
    """
    Uses 1D Mask file (with NaN and 0) and generates CPD of non-grain boundary regions
    
    h5_file : H5Py File
        commonly as hdf.file
        
    CPD : ndarray
        RowsXColumns matrix of CPD average values such as CPD_on_avg
        
    mask : ndarray, 2D
        Unmasked locations (indices) as 1D location
    
    CPD_loc : str, optional
        The path to the dataset within the h5_file
    
    """
        
    hdf = px.ioHDF5(h5_file)

    # We know what folder we are operating in
    pos_ind = hdf.file[CPD_loc+'/Position_Indices']
    pos_val = hdf.file[CPD_loc+'/Position_Values']

    ones = np.where(mask == 1)
    CPD_1D_vals = np.zeros(ones[0].shape[0])

    for r,c,x in zip(ones[0], ones[1], np.arange(CPD_1D.shape[0])):
        CPD_1D_vals[x] = CPD_avg[r][c]

    return CPD_1D_vals

def make_distance_arrays(mask_on_1D, mask_off_1D, pos_vals, 
                         num_rows=64, num_cols=128):
    """
    Generates 1D arrays where the coordinates are scaled to image dimenions
    
    Returns
    -------
    
    mask_on_1D_scaled : ndarray Nx2
        Where mask is applied (e.g. grain boundaries)
        
    mask_off_1D_scaled : ndarray Nx2
        Where mask isn't applied (e.g. grains)
    
    CPD_1D_scaled : ndarray Nx2
        Identical to mask_off_1D_scaled, this exists just for easier bookkeeping
        without fear of overwriting one or the other
    
    """
    
    # If from H5 file
    if 'Dataset' in str(type(pos_val)):
        pos_val = pos_val.value
    
    # the length and width of the image
    csz = pos_val[-1][0] / num_cols # per pixel
    rsz = pos_val[-1][1] / num_rows # per pixel
    
    mask_on_1D_scaled = np.zeros([mask_on_1D.shape[0],2])
    mask_off_1D_scaled = np.zeros([mask_off_1D.shape[0],2])
    
    for x,y in zip(mask_on_1D, np.arange(mask_on_1D_scaled.shape[0])):
        mask_on_1D_scaled[y,0] = x[0] * rsz
        mask_on_1D_scaled[y,1] = x[1] * csz
        
    for x,y in zip(mask_off_1D, np.arange(mask_off_1D_scaled.shape[0])):
        mask_off_1D_scaled[y,0] = x[0] * rsz
        mask_off_1D_scaled[y,1] = x[1] * csz
    
    CPD_1D_scaled = np.copy(mask_off_1D_scaled) # to keep straight, but these are the same
    
    return mask_on_1D_scaled, mask_off_1D_scaled, CPD_1D_scaled

def CPD_distances(CPD_1D, mask_on_1D):
    
    return