# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 13:52:06 2018

@author: Raj
"""

import numpy as np

def pixelCPD(CPDarr, pixel, rows=128):
    '''
    CPDarr: ndarray
    pixel: [int] with [row, col]
    rows: int
    
    Takes [row,col] point in an array and returns the CPD for that points
    This assumes CPD is 1xn and not in a matrix image
    '''

    return CPDarr[rows*pixel[0] + pixel[1]]
    

def averagemask(CPDarr, mask, rows=128, nan_flag = 1, avg_flag = 0):
    '''
    Returns an averaged CPD trace given the Igor mask
    Mask is assumed to be in image form of [row, col] dimensions
    
    CPDarr is the CPD of n-by-samples, typically 8192 x 128 (8192 pixels, 128 CPD points)
    Rows = 128 is default of 128x64 images
    nan_flag = 1 is what value in the mask to set to NaN. These are IGNORED
        1 is "transparent" in Igor mask
        0 is "opaque/masked off" in Igor mask
    avg_flag = 0 is what values in mask to average.
    
    Returns CPDpixels, the averaged CPD of those pixels
    '''
    mask1D = np.reshape(mask, [mask.shape[0]*mask.shape[1]])
    
    CPDpixels = np.array([])
    
    index = [i for i in np.where(mask1D == avg_flag)[0]]
    for i in index:
        CPDpixels = np.append(CPDpixels, CPDarr[i,:])
        
    CPDpixels = np.reshape(CPDpixels, [len(index), CPDarr.shape[1]])
    CPDpixels = np.mean(CPDpixels, axis=0)

    return CPDpixels