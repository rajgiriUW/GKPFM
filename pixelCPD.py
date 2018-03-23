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
    

